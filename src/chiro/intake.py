"""Patient self-service intake: accounts and website inquiries that feed the lead pipeline.

An inquiry is appended to `leads_raw` (the pipeline picks it up on its next run), scored with the
same lead model the daily job trains, and handed to the Lead agent straight away so a draft reaches
the staff review queue in seconds instead of waiting for tomorrow's run.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid

import pandas as pd

from chiro import models, urgency
from chiro.config import Settings
from chiro.sql import SqlRunner
from chiro.workflows import run_named_agent

COMPLAINTS = {
    "low_back_pain": "Lower back pain", "neck_pain": "Neck pain", "headaches": "Headaches",
    "sciatica": "Pain down the leg (sciatica)", "sports_injury": "Sports injury",
    "auto_injury": "Car accident injury", "posture_wellness": "Posture / general wellness",
}
INSURANCE = {"ppo": "PPO insurance", "medicare": "Medicare", "cash": "Paying myself", "unknown": "Not sure yet"}

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SCRYPT = dict(n=2**14, r=8, p=1)


class IntakeError(ValueError):
    """A problem the patient can fix; shown in the form."""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    return f"scrypt${salt.hex()}${hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT).hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), **_SCRYPT).hex()
    except ValueError:
        return False
    return hmac.compare_digest(actual, digest)


class Intake:
    def __init__(self, db: SqlRunner, settings: Settings):
        self.db, self.s = db, settings

    def t(self, name: str) -> str:
        return self.s.table(name)

    # ------------------------------------------------------------------ accounts
    def create_account(self, email: str, password: str, first_name: str, phone: str = "",
                       consent_to_contact: bool = True, consent_sms: bool = False, consent_email: bool = False) -> dict:
        email, first_name, phone = email.strip().lower(), first_name.strip(), phone.strip()
        if not _EMAIL.match(email):
            raise IntakeError("Enter a valid email address.")
        if len(password) < 8:
            raise IntakeError("Password must be at least 8 characters.")
        if not first_name:
            raise IntakeError("Enter your first name.")
        if self.db.query(f"SELECT 1 FROM {self.t('patient_accounts')} WHERE email = :email", {"email": email}):
            raise IntakeError("An account with this email already exists. Log in instead.")
        account = {"account_id": f"ACC-{uuid.uuid4().hex[:10]}", "email": email, "first_name": first_name,
                   "phone": phone, "consent_to_contact": bool(consent_to_contact),
                   "consent_sms": bool(consent_to_contact and consent_sms),
                   "consent_email": bool(consent_to_contact and consent_email)}
        self.db.execute(
            f"""INSERT INTO {self.t('patient_accounts')}
                (account_id, email, password_hash, first_name, phone, consent_to_contact, consent_sms,
                 consent_email, created_at)
                VALUES (:account_id, :email, :pw, :first_name, :phone, :consent_to_contact, :consent_sms,
                        :consent_email, current_timestamp())""",
            {**account, "pw": hash_password(password)})
        return account

    def log_in(self, email: str, password: str) -> dict | None:
        rows = self.db.query(f"SELECT * FROM {self.t('patient_accounts')} WHERE email = :email",
                             {"email": email.strip().lower()})
        if not rows or not verify_password(password, rows[0]["password_hash"]):
            return None
        return {k: v for k, v in rows[0].items() if k != "password_hash"}

    # ------------------------------------------------------------------ inquiries
    def train_lead_model(self):
        """Same model and features as the daily refresh job, trained on historical outcomes."""
        rows = self.db.query(f"SELECT source, complaint, insurance_type, distance_miles, message, converted "
                             f"FROM {self.t('leads')} WHERE converted IS NOT NULL")
        model, _ = models.train_lead_model(pd.DataFrame(rows))
        return model

    def submit_inquiry(self, account: dict, complaint: str, insurance_type: str, distance_miles: float,
                       location_id: str, message: str, lead_model, llm_client=None, llm_model: str = "") -> dict:
        message = message.strip()
        if complaint not in COMPLAINTS:
            raise IntakeError("Choose what you'd like help with.")
        if insurance_type not in INSURANCE:
            raise IntakeError("Choose how you plan to pay.")
        if not 0 <= float(distance_miles) <= 200:
            raise IntakeError("Distance must be between 0 and 200 miles.")
        if len(message) < 10:
            raise IntakeError("Tell us a little more about what's going on.")
        lead = {"lead_id": f"WEB-{uuid.uuid4().hex[:10].upper()}", "first_name": account["first_name"],
                "source": "website_form", "complaint": complaint, "insurance_type": insurance_type,
                "distance_miles": round(float(distance_miles), 1), "message": message,
                "consent_to_contact": bool(account["consent_to_contact"]), "consent_sms": bool(account["consent_sms"]),
                "consent_email": bool(account["consent_email"]), "status": "new", "location_id": location_id}
        self.db.execute(
            f"""INSERT INTO {self.t('leads_raw')}
                (lead_id, created_at, first_name, source, complaint, insurance_type, distance_miles, message,
                 consent_to_contact, consent_sms, consent_email, first_response_hours, status, converted,
                 num_touchpoints, assigned_location_id)
                VALUES (:lead_id, current_timestamp(), :first_name, :source, :complaint, :insurance_type,
                        CAST(:distance_miles AS DECIMAL(5, 1)), :message, :consent_to_contact, :consent_sms,
                        :consent_email, NULL, :status, NULL, 0, :location_id)""", lead)
        self.db.execute(f"INSERT INTO {self.t('account_leads')} VALUES (:account, :lead, current_timestamp())",
                        {"account": account["account_id"], "lead": lead["lead_id"]})
        # The daily refresh rescores every open lead; this row just covers the gap until then.
        llm = None
        if llm_client is not None:  # second opinion on urgency; rules stand alone if it fails
            llm = {lead["lead_id"]: urgency.llm_triage(llm_client, llm_model, message)}
        score = models.score_leads(lead_model, pd.DataFrame([lead]), llm).iloc[0]
        self.db.execute(
            f"""INSERT INTO {self.t('lead_scores')}
                (lead_id, score, reasons, red_flags, urgency, urgency_reasons, respond_within_hours, priority, scored_at)
                VALUES (:lead, :score, :reasons, :flags, :urgency, :urgency_reasons, :within, :priority,
                        current_timestamp())""",
            {"lead": lead["lead_id"], "score": float(score["score"]), "reasons": score["reasons"],
             "flags": score["red_flags"], "urgency": score["urgency"], "urgency_reasons": score["urgency_reasons"],
             "within": float(score["respond_within_hours"]), "priority": float(score["priority"])})
        return {"lead_id": lead["lead_id"], "score": float(score["score"]), "urgency": score["urgency"],
                "respond_within_hours": float(score["respond_within_hours"]),
                "red_flags": [f for f in score["red_flags"].split(", ") if f]}

    def my_inquiries(self, account_id: str) -> list[dict]:
        return self.db.query(
            f"""SELECT a.lead_id, a.submitted_at, r.complaint, r.message,
                       (SELECT max_by(x.status, x.created_at) FROM {self.t('lead_actions')} x
                         WHERE x.lead_id = a.lead_id) AS review_status
                FROM {self.t('account_leads')} a JOIN {self.t('leads_raw')} r USING (lead_id)
                WHERE a.account_id = :id ORDER BY a.submitted_at DESC""", {"id": account_id})


def process_inquiry(lead_id: str, db: SqlRunner, settings: Settings, client):
    """Run the Lead agent on just this inquiry; its draft lands in the staff review queue."""
    return run_named_agent(
        "lead", db, settings, client, max_items=1, max_steps=12,
        instruction=f"A patient just submitted inquiry {lead_id} on the website. Work only this lead: skip "
                    f"list_scored_leads, call get_lead('{lead_id}') and queue exactly one action for it.")
