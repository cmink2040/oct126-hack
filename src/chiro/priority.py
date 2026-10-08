"""Lead queue prioritisation: which open lead should staff contact next, and why.

Sort order, each step breaking ties in the one before:
  1. urgency tier - emergency, urgent, soon, routine (safety and patient experience come first);
  2. deadline pressure - overdue, due within the hour, later (keeps the response-time promise);
  3. expected value - P(convert) x what a new patient on that payer is typically worth.

P(convert) comes from the lead model but is shrunk toward the base rate in proportion to the model's
measured skill (AUC on held-out leads): with AUC 0.5 every lead gets the base rate, so a model that
can't discriminate never masquerades as signal. Leads older than ACTIVE_LEAD_DAYS are out of the triage
queue - by then they need a nurture sequence, not a first response.
"""
from __future__ import annotations

import datetime as dt

from chiro.urgency import TIERS

ACTIVE_LEAD_DAYS = 30
FULL_TRUST_AUC = 0.75  # AUC at which the model's scores are used as-is
DUE_SOON = dt.timedelta(hours=1)
DEADLINE_STATES = ("overdue", "due_soon", "on_track")


def model_skill(auc: float | None) -> float:
    if auc is None:
        return 0.0
    return min(max((float(auc) - 0.5) / (FULL_TRUST_AUC - 0.5), 0.0), 1.0)


def adjusted_probability(score: float | None, base_rate: float, skill: float) -> float:
    if score is None:
        return base_rate
    return base_rate + (float(score) - base_rate) * skill


def deadline_state(respond_by: dt.datetime | None, now: dt.datetime) -> str:
    if respond_by is None:
        return "on_track"
    if now > respond_by:
        return "overdue"
    return "due_soon" if respond_by - now <= DUE_SOON else "on_track"


def _ts(value) -> dt.datetime | None:
    if value is None or isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def rank(leads: list[dict], now: dt.datetime, base_rate: float, auc: float | None,
         payer_value: dict[str, float]) -> list[dict]:
    """Annotate and sort leads (each needs urgency, score, insurance_type, respond_by)."""
    skill = model_skill(auc)
    default_value = sum(payer_value.values()) / len(payer_value) if payer_value else 0.0
    out = []
    for lead in leads:
        tier = lead.get("urgency") if lead.get("urgency") in TIERS else "routine"
        due = _ts(lead.get("respond_by"))
        p = adjusted_probability(lead.get("score"), base_rate, skill)
        value = payer_value.get(lead.get("insurance_type"), default_value)
        state = deadline_state(due, now)
        minutes = None if due is None else round((due - now).total_seconds() / 60)
        out.append({**lead, "urgency": tier, "deadline_state": state, "minutes_to_deadline": minutes,
                    "p_convert": round(p, 3), "patient_value": round(value, 0), "expected_value": round(p * value, 1),
                    "_key": (TIERS.index(tier), DEADLINE_STATES.index(state), -p * value)})
    out.sort(key=lambda r: r["_key"])
    for i, r in enumerate(out, 1):
        del r["_key"]
        r["queue_position"] = i
    return out
