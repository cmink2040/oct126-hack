"""Deterministic guardrails enforced inside agent tools.

The LLM is told these rules in its prompt, but the tools enforce them in code,
so a non-compliant draft can never reach the review queue.
"""
from __future__ import annotations

import re

# Healthcare-advertising claims a chiropractic clinic must not make.
_BANNED_CLAIMS = {
    r"\bcur(e|es|ed|ing)\b": "claims to cure a condition",
    r"\bguarantee": "guarantees an outcome",
    r"\bpermanent(ly)?\b": "promises permanent results",
    r"\b100\s?%": "absolute claim (100%)",
    r"\bno risk\b|\brisk[- ]free\b": "claims no risk",
    r"\bmiracle": "miracle claim",
    r"\binstead of (your )?(doctor|surgery|medication|physician)": "discourages medical care",
    r"\b(only|last) \d+ (spots?|slots?) left\b|\bact now\b|\bexpires (today|tonight)\b": "high-pressure / false scarcity",
}

# Symptoms that need urgent medical evaluation rather than a marketing reply.
_RED_FLAGS = {
    r"bladder|bowel control|incontinen|(can'?t|trouble|difficulty|problems?) (pee|peeing|urinat)|"
    r"wetting (myself|the bed)|leaking urine|lost control of my (bowels?|bladder)": "bladder/bowel dysfunction",
    r"groin.{0,30}numb|numb.{0,30}groin|saddle (area|numb)|numb(ness)? (down there|in my (private|genital|buttocks?))|"
    r"numb when i (sit|wipe)": "saddle anesthesia",
    r"worst headache|thunderclap|thunderbolt|headache.{0,20}(out of nowhere|came on suddenly|like being hit)":
        "sudden severe headache",
    r"chest (pain|tightness|pressure)|chest (feels|is) (heavy|tight)|pressure in my chest": "chest pain",
    r"\bfever\b|high temperature|running a temperature|\bchills\b|night sweats": "fever with pain",
    r"(legs?|arms?) (keep |are |is )?(giving out|getting weaker)|progressive weakness|"
    r"(arm|leg|hand|foot) is getting weaker|can'?t feel (my )?(legs?|feet|foot)|numb(ness)? in both (legs|feet)|"
    r"(feet|legs) (feel|are|went|going) numb|foot drop": "progressive weakness",
    r"unexplained weight loss|weight loss without trying|lost \d+ ?(pounds|lbs|kg) without trying":
        "unexplained weight loss",
    r"passed out|lost consciousness|unconscious|fainted": "loss of consciousness",
}

# NegEx-style negation: a cue shortly before the symptom, in the same clause, negates it ("no fever",
# "I don't have any numbness"). Hedges ("not sure if it's a fever") are NOT negations - when in doubt,
# keep the flag, because a missed red flag costs far more than a phone call.
# Inability ("can't control my bladder", "couldn't feel my feet") is a symptom, not a negation, so only
# denial forms count.
_NEGATION_CUE = re.compile(r"\b(no|not|never|without|denies|deny|none|nor|neither|"
                           r"(do|does|did|have|has|had|is|are|was|were)(n'?t| not))\b")
_HEDGE = re.compile(r"\b(not sure|unsure|don'?t know|not certain|can'?t tell|maybe|might|possibly|think)\b")
_CLAUSE_BREAK = re.compile(r"[.;!?\n]|\b(but|however|although|though|except|yet)\b")
_NEGATION_WINDOW_WORDS = 5

SMS_MAX_CHARS = 320
EMAIL_MAX_CHARS = 1500
SMS_OPT_OUT = "Reply STOP to opt out."


def is_negated(text: str, start: int) -> bool:
    """True when the term starting at `start` is negated within its clause (lower-cased text)."""
    before = text[:start]
    breaks = list(_CLAUSE_BREAK.finditer(before))
    clause = before[breaks[-1].end():] if breaks else before
    window = " ".join(clause.split()[-_NEGATION_WINDOW_WORDS:])
    return bool(_NEGATION_CUE.search(window)) and not _HEDGE.search(window)


def detect_red_flags(text: str | None) -> list[str]:
    text = (text or "").lower()
    return [label for pattern, label in _RED_FLAGS.items()
            if any(not is_negated(text, m.start()) for m in re.finditer(pattern, text))]


def check_message(message: str | None, channel: str) -> list[str]:
    """Return a list of problems; empty means the draft is acceptable."""
    msg = (message or "").strip()
    problems: list[str] = []
    if len(msg) < 20:
        problems.append("message is empty or too short")
    lowered = msg.lower()
    problems += [f"message {why}" for pattern, why in _BANNED_CLAIMS.items() if re.search(pattern, lowered)]
    limit = SMS_MAX_CHARS - len(SMS_OPT_OUT) - 1 if channel == "sms" else EMAIL_MAX_CHARS
    if channel in ("sms", "email") and len(msg) > limit:
        problems.append(f"message is {len(msg)} chars; {channel} limit is {limit}")
    return problems


def ensure_sms_opt_out(message: str) -> str:
    message = message.strip()
    if "stop" in message.lower():
        return message
    return f"{message} {SMS_OPT_OUT}"
