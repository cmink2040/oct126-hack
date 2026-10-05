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
    r"bladder|bowel control|incontinen": "bladder/bowel dysfunction",
    r"groin.{0,30}numb|numb.{0,30}groin|saddle (area|numb)": "saddle anesthesia",
    r"worst headache|thunderclap": "sudden severe headache",
    r"chest pain": "chest pain",
    r"\bfever\b": "fever with pain",
    r"(legs?|arms?) (keep |are )?(giving out|getting weaker)|progressive weakness": "progressive weakness",
    r"unexplained weight loss": "unexplained weight loss",
    r"passed out|lost consciousness|unconscious": "loss of consciousness",
}

SMS_MAX_CHARS = 320
EMAIL_MAX_CHARS = 1500
SMS_OPT_OUT = "Reply STOP to opt out."


def detect_red_flags(text: str | None) -> list[str]:
    text = (text or "").lower()
    return [label for pattern, label in _RED_FLAGS.items() if re.search(pattern, text)]


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
