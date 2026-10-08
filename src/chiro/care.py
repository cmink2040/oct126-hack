"""Care guidance reports: staff care notes -> "what to keep doing, and what happens if you don't".

Staff write short care notes about follow-through on the treatment plan (home exercises, posture at
work, keeping the visit rhythm, aftercare), e.g. "Skipping the daily stretches tends to bring the
stiffness back and means extra visits." A model turns the notes plus the patient's visit pattern and
recommended care settings into two drafts:
  * a staff report - summary, signals, talking points;
  * a patient report - plain-language advice items, each with the consequence of skipping it.
Not diagnostic: the drafts may only restate what staff wrote and the attendance facts, and are
checked in code for diagnostic or outcome-promising language. Staff approve before the patient sees it.
"""
from __future__ import annotations

import json
import re
import uuid

from chiro import guardrails
from chiro.config import Settings
from chiro.llm import max_output_tokens
from chiro.sql import SqlRunner

_CLINICAL = {
    r"\bdiagnos": "diagnostic language",
    r"\bprognos": "prognosis",
    r"\bprescri": "prescribing",
    r"\b(dosage|dose|\d+\s?mg)\b": "medication dosing",
    r"\byou (have|suffer from|are suffering from) (a |an )?[\w-]*\s?(disease|disorder|syndrome|condition|tear|fracture)":
        "states a medical condition",
    r"\bwill (definitely|certainly|always)\b|\bcertain to\b": "certainty about outcomes",
    r"\b(stop|change|skip) (taking )?(your )?medication": "medication advice",
}
MAX_ITEMS = 6

REPORT_PROMPT = """You write care-plan guidance for {clinic}, a chiropractic clinic, from staff care notes.
You get JSON with: the patient's first name, attendance facts (visits, gaps, no-shows, plan progress),
their treatment theme and recommended non-medical care settings, and the staff's care notes.

Return ONLY a JSON object:
{{"staff_summary": "3-5 sentences for staff: where the patient is with their plan, the biggest follow-through
   risk, and what to raise at the next visit. Cite the facts.",
  "patient_intro": "1-2 warm sentences addressed to the patient by first name.",
  "items": [{{"advice": "one concrete thing to keep doing", "if_skipped": "the realistic consequence of skipping it"}}]}}

Rules:
- 2 to {max_items} items. Every item must come from a staff note or an attendance fact; add nothing medical of your own.
- This is guidance, not a medical assessment: never diagnose, never name conditions the notes don't name,
  never predict outcomes with certainty ("may", "tends to", "often" - not "will"), no medication advice.
- Consequences are practical: e.g. symptoms coming back, progress slowing, needing extra visits or a re-check.
- Plain words, second person, no jargon, no scare tactics. No "cure", "guarantee", "permanent", "100%", "risk-free".
- If the notes include something that is not appropriate to tell the patient, leave it to the staff summary."""


def check_care_language(text: str) -> list[str]:
    lowered = (text or "").lower()
    problems = [why for pattern, why in _CLINICAL.items() if re.search(pattern, lowered)]
    problems += [p for p in guardrails.check_message(text, "report") if "too short" not in p]
    return problems


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object in the response")
    data = json.loads(match.group(0))
    items = data.get("items") or []
    if not (isinstance(data.get("staff_summary"), str) and isinstance(data.get("patient_intro"), str)):
        raise ValueError("staff_summary and patient_intro must be strings")
    if not 1 <= len(items) <= MAX_ITEMS or not all(
            isinstance(i, dict) and str(i.get("advice", "")).strip() and str(i.get("if_skipped", "")).strip()
            for i in items):
        raise ValueError(f"items must be 1-{MAX_ITEMS} objects with advice and if_skipped")
    return data


def validate(data: dict) -> list[str]:
    texts = [data["patient_intro"], data["staff_summary"]]
    texts += [t for i in data["items"] for t in (i["advice"], i["if_skipped"])]
    return sorted({p for t in texts for p in check_care_language(t)})


def _pct(x) -> str | None:
    return None if x is None else f"{float(x):.0%}"


def render_patient(data: dict) -> str:
    lines = [data["patient_intro"].strip(), ""]
    for i in data["items"]:
        lines.append(f"- **{i['advice'].strip()}**  \n  If this slips: {i['if_skipped'].strip()}")
    return "\n".join(lines)


def render_staff(data: dict, context: dict) -> str:
    facts = context["attendance"]
    recs = [r for r in context.get("care_settings", []) if not r["keep_current"]]
    lines = [data["staff_summary"].strip(), "",
             f"**Theme:** {context.get('theme') or 'n/a'}  ",
             "**Attendance:** " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in facts.items() if v is not None)]
    if recs:
        lines.append("**Suggested care-setting changes:** " + "; ".join(
            f"{r['aspect']}: {r['current']} → {r['recommended']} ({r['evidence']} evidence, +{r['lift']:.0%} active)"
            for r in recs))
    lines += ["", "**Patient version items:**"] + [f"- {i['advice']} — if skipped: {i['if_skipped']}" for i in data["items"]]
    return "\n".join(lines)


class CareReports:
    def __init__(self, db: SqlRunner, settings: Settings, run_id: str = ""):
        self.db, self.s, self.run_id = db, settings, run_id or f"care-{uuid.uuid4().hex[:8]}"

    def t(self, name: str) -> str:
        return self.s.table(name)

    def add_note(self, patient_id: str, author: str, note: str) -> dict:
        note = (note or "").strip()
        if not 10 <= len(note) <= 1000:
            return {"error": "a care note must be 10-1000 characters"}
        if not self.db.query(f"SELECT 1 FROM {self.t('patients')} WHERE patient_id = :id", {"id": patient_id}):
            return {"error": f"patient {patient_id} not found"}
        note_id = f"CN-{uuid.uuid4().hex[:10]}"
        self.db.execute(f"INSERT INTO {self.t('care_notes')} VALUES (:nid, :pid, :author, :note, current_timestamp())",
                        {"nid": note_id, "pid": patient_id, "author": author or "staff", "note": note})
        return {"ok": True, "note_id": note_id}

    def notes(self, patient_id: str, limit: int = 10) -> list[dict]:
        return self.db.query(f"SELECT note_id, author, note, created_at FROM {self.t('care_notes')} "
                             f"WHERE patient_id = :id ORDER BY created_at DESC LIMIT {int(limit)}", {"id": patient_id})

    def context(self, profile: dict, recommendations: dict | None) -> dict:
        visits = profile.get("recent_visits") or []
        return {
            "first_name": profile.get("first_name"),
            "attendance": {
                "days_since_last_visit": profile.get("days_since_last_visit"),
                "visits_planned_in_care_plan": profile.get("care_plan_visits"),
                "share_of_plan_completed": _pct(profile.get("plan_progress")),
                "no_shows_in_last_8_bookings": sum(1 for v in visits if v.get("status") == "no_show"),
                "dropout_risk": _pct(profile.get("churn_risk")),
            },
            "theme": (recommendations or {}).get("theme"),
            "care_settings": (recommendations or {}).get("recommendations", []),
            "staff_notes": [n["note"] for n in self.notes(profile["patient_id"])],
        }

    def draft(self, client, model: str, profile: dict, recommendations: dict | None = None) -> dict:
        """Draft both reports, re-asking once if the output breaks format or language rules."""
        ctx = self.context(profile, recommendations)
        if not ctx["staff_notes"]:
            return {"error": "add at least one care note first; reports are built from staff notes"}
        messages = [{"role": "system", "content": REPORT_PROMPT.format(clinic=self.s.clinic_name, max_items=MAX_ITEMS)},
                    {"role": "user", "content": json.dumps(ctx, default=str)}]
        problems: list[str] = []
        for _ in range(2):
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2,
                                                  max_tokens=max_output_tokens())
            text = resp.choices[0].message.content or ""
            if getattr(resp.choices[0], "finish_reason", None) == "length":
                return {"error": "the model ran out of output tokens before finishing; raise LLM_MAX_TOKENS"}
            try:
                data = _parse(text)
                problems = validate(data)
            except (ValueError, json.JSONDecodeError) as e:
                problems = [str(e)]
            if not problems:
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Fix these problems and return the full JSON again: "
                                                     + "; ".join(problems)}]
        if problems:
            return {"error": "draft rejected: " + "; ".join(problems)}
        report_id = f"CR-{uuid.uuid4().hex[:10]}"
        staff, patient = render_staff(data, ctx), render_patient(data)
        self.db.execute(
            f"""INSERT INTO {self.t('care_reports')}
                (report_id, run_id, patient_id, staff_report, patient_report, status, created_at)
                VALUES (:rid, :run, :pid, :staff, :patient, 'pending_review', current_timestamp())""",
            {"rid": report_id, "run": self.run_id, "pid": profile["patient_id"], "staff": staff, "patient": patient})
        return {"ok": True, "report_id": report_id, "status": "pending_review", "staff_report": staff,
                "patient_report": patient}

    def latest_approved(self, patient_id: str) -> dict | None:
        rows = self.db.query(f"SELECT patient_report, reviewed_at FROM {self.t('care_reports')} "
                             "WHERE patient_id = :id AND status = 'approved' ORDER BY reviewed_at DESC LIMIT 1",
                             {"id": patient_id})
        return rows[0] if rows else None
