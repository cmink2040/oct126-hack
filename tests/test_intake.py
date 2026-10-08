"""Patient intake: password hashing, account rules, and how an inquiry becomes a scored lead."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from chiro.config import Settings
from chiro.intake import Intake, IntakeError, hash_password, verify_password
from chiro.tools import ClinicTools


class RecordingDB:
    def __init__(self, rows=None):
        self.rows, self.writes = rows or {}, []

    def query(self, sql, params=None):
        return next((list(v) for k, v in self.rows.items() if k in sql), [])

    def execute(self, sql, params=None):
        self.writes.append((sql, params))


ACCOUNT = dict(account_id="ACC-1", first_name="Sam", consent_to_contact=True, consent_sms=False, consent_email=True)
MODEL = NS(feature_columns_=["distance_miles", "urgent_language"], predict_proba=lambda X: np.array([[0.4, 0.6]]))


def test_password_hash_roundtrip():
    stored = hash_password("correct horse")
    assert stored.startswith("scrypt$") and "correct horse" not in stored
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong", stored) and not verify_password("x", "garbage")


def test_account_validation_and_duplicates():
    intake = Intake(RecordingDB({"patient_accounts": [{"x": 1}]}), Settings())
    with pytest.raises(IntakeError, match="valid email"):
        intake.create_account("nope", "longenough", "Sam")
    with pytest.raises(IntakeError, match="8 characters"):
        intake.create_account("a@b.co", "short", "Sam")
    with pytest.raises(IntakeError, match="already exists"):
        intake.create_account("A@B.co", "longenough", "Sam")


def test_no_contact_consent_means_no_channel_consent():
    db = RecordingDB()
    acct = Intake(db, Settings()).create_account(" Sam@Example.com ", "longenough", "Sam", "", False, True, True)
    assert acct["email"] == "sam@example.com" and not acct["consent_sms"] and not acct["consent_email"]
    assert "longenough" not in str(db.writes) and "password_hash" not in acct


def test_inquiry_lands_in_landing_table_with_score_and_red_flags():
    db = RecordingDB()
    out = Intake(db, Settings()).submit_inquiry(
        ACCOUNT, "headaches", "cash", 3, "LOC001", "Worst headache of my life, came on suddenly an hour ago", MODEL)
    assert out["lead_id"].startswith("WEB-") and out["score"] == 0.6
    assert out["red_flags"] == ["sudden severe headache"]
    tables = [sql.split("INTO ")[1].split()[0] for sql, _ in db.writes]
    assert tables == ["`workspace`.`chiro`.`leads_raw`", "`workspace`.`chiro`.`account_leads`",
                      "`workspace`.`chiro`.`lead_scores`"]
    lead = db.writes[0][1]
    assert lead["source"] == "website_form" and lead["status"] == "new" and lead["consent_email"]


def test_inquiry_validation():
    intake = Intake(RecordingDB(), Settings())
    with pytest.raises(IntakeError):
        intake.submit_inquiry(ACCOUNT, "headaches", "cash", 3, "LOC001", "hi", MODEL)
    with pytest.raises(IntakeError):
        intake.submit_inquiry(ACCOUNT, "brain_surgery", "cash", 3, "LOC001", "a long enough message", MODEL)


def test_agent_sees_leads_the_pipeline_has_not_processed_yet():
    lead = dict(lead_id="WEB-1", message="Neck stiffness for a week", consent_to_contact=True,
                consent_sms=False, consent_email=True, score=0.5, reasons="")
    t = ClinicTools(RecordingDB({"`leads_raw`": [lead]}), Settings(), run_id="t")
    assert t.get_lead("WEB-1")["allowed_channels"] == ["email", "phone"]
