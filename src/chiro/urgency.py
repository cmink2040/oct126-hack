"""Urgency triage for inbound leads: how fast the clinic should respond, never what it charges.

Tiers, most urgent first:
  emergency - red-flag symptoms: staff call to direct them to medical care (no booking pitch)
  urgent    - acute onset or severe impact on daily life: reach out within the hour, offer same/next-day
  soon      - worsening, recent or moderate: reach out within a few hours
  routine   - chronic, wellness or information-seeking: same business day

Speed of first response drives conversion, so the tier sets the response deadline and queue order.
Two classifiers: an LLM (primary - it handles new phrasing) and transparent rules (a safety floor that
also works when no LLM is reachable). The final tier is the more severe of the two; see combine().
Measured on a held-out set with scripts/eval_triage.py.
"""
from __future__ import annotations

import re

from chiro.guardrails import detect_red_flags, is_negated

TIERS = ("emergency", "urgent", "soon", "routine")
RESPOND_WITHIN_HOURS = {"emergency": 0.25, "urgent": 1.0, "soon": 4.0, "routine": 24.0}

_ACUTE = re.compile(r"since (yesterday|this morning|last night|today)|yesterday|this morning|last night|an hour ago|"
                    r"few hours|just (happened|now)|out of nowhere|suddenly|threw (out|my)|tweaked|"
                    r"(fell|slipped|accident|crash|fender bender)", re.I)
_SEVERE = re.compile(r"can'?t (walk|sleep|work|sit|stand|move|turn|bend|lift|drive|get out of bed)|can barely|"
                     r"unbearable|excruciating|killing me|severe|agony|in tears|locked up", re.I)
_SUBACUTE = re.compile(r"this week|last week|past (few|couple of) days|few days|getting worse|worse|flare|"
                       r"keeps coming back|(isn'?t|not) (improving|getting better)|started (on )?(mon|tues|wednes|thurs|fri|satur|sun)day|"
                       r"on the weekend|asap|as soon as|hoping to get in|aching more", re.I)
_CHRONIC = re.compile(r"for (months|years)|\bmonths\b|\byears\b|wellness|maintenance|posture|just looking|"
                      r"\binfo\b|information|how much|price|rates?\b", re.I)
_MILD = re.compile(r"\b(a little|a bit|slight(ly)?|mild(ly)?|minor|tiny bit)\b (sore|stiff|achy|tight|pain)", re.I)
_PAIN_SCORE = re.compile(r"\b(10|[0-9])\s*(/|out of)\s*10\b")


def _found(pattern: re.Pattern, text: str) -> bool:
    lowered = text.lower()
    return any(not is_negated(lowered, m.start()) for m in pattern.finditer(lowered))


def classify(message: str | None, complaint: str | None = None, ai_intent: str | None = None) -> tuple[str, list[str]]:
    """Return (tier, reasons) for one lead message."""
    text = message or ""
    flags = detect_red_flags(text)
    if flags:
        return "emergency", [f"red flag: {f}" for f in flags]

    points, reasons = 0, []
    pain = [int(m.group(1)) for m in _PAIN_SCORE.finditer(text)]
    if pain:
        p = max(pain)
        points += 3 if p >= 7 else 1 if p >= 4 else 0
        reasons.append(f"pain {p}/10")
    if _found(_SEVERE, text):
        points += 3
        reasons.append("severe impact on daily life")
    if _found(_ACUTE, text):
        points += 2
        reasons.append("acute onset")
    if complaint == "auto_injury":
        points += 1
        reasons.append("injury from an accident")
    if _found(_SUBACUTE, text):
        points += 1
        reasons.append("recent / worsening")
    if _found(_MILD, text) and not pain:
        points -= 2
        reasons.append("described as mild")
    if ai_intent == "urgent_medical":
        points += 2
        reasons.append("AI intent: urgent")
    elif ai_intent == "ready_to_book":
        points += 1
        reasons.append("AI intent: ready to book")
    if _CHRONIC.search(text) and points < 3:
        reasons.append("long-standing or information-seeking")
        return "routine", reasons

    tier = "urgent" if points >= 4 else "soon" if points >= 1 else "routine"
    return tier, reasons or ["no urgency signals"]


LLM_TRIAGE_PROMPT = """You triage inbound messages to a chiropractic clinic by how fast staff should respond.
Return ONLY JSON: {"tier": "emergency|urgent|soon|routine", "red_flags": ["..."], "reason": "<= 15 words"}
- emergency: possible red-flag symptoms that need medical care first: bladder/bowel changes, numbness in the
  groin/saddle area, sudden worst-ever headache, chest pain/pressure, fever or chills with back/neck pain,
  progressive leg/arm weakness or loss of feeling, fainting, unexplained weight loss. List them in red_flags.
  Denied symptoms ("no fever") are not red flags; uncertain ones ("not sure if it's a fever") are.
- urgent: acute onset in the last ~48h or severe impact on daily life (can't walk/sleep/work, pain >= 7/10).
- soon: recent, worsening or moderate problems, or they ask to be seen this week.
- routine: long-standing, mild, wellness or information-only.
Do not diagnose. Judge only what the message says."""


def llm_triage(client, model: str, message: str) -> dict | None:
    """Second opinion from an LLM; None if the call or its output is unusable (rules then stand alone)."""
    import json

    from chiro.llm import max_output_tokens, request_options

    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=max_output_tokens(),
            messages=[{"role": "system", "content": LLM_TRIAGE_PROMPT}, {"role": "user", "content": message or ""}],
            **request_options(thinking=False))
        text = resp.choices[0].message.content or ""
        out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        return None
    if out.get("tier") not in TIERS:
        return None
    return {"tier": out["tier"], "red_flags": [str(f) for f in out.get("red_flags") or []],
            "reason": str(out.get("reason", ""))[:200]}


def llm_triage_many(client, model: str, messages: dict[str, str], workers: int = 8) -> dict[str, dict | None]:
    """LLM triage for many leads in parallel: {lead_id: result or None}."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(lambda m: llm_triage(client, model, m), messages.values())
        return dict(zip(messages, results))


def triage(message: str | None, complaint: str | None = None, ai_intent: str | None = None,
           llm: dict | None = None, llm_attempted: bool = False) -> tuple[str, list[str]]:
    """Final tier for a lead: rules alone, or rules combined with an LLM second opinion."""
    rules = classify(message, complaint, ai_intent)
    if not llm_attempted:
        return rules
    return combine(rules, llm)


def combine(rules: tuple[str, list[str]], llm: dict | None) -> tuple[str, list[str]]:
    """Ensemble policy: the LLM is the primary classifier (it generalises to new phrasing far better - see
    scripts/eval_triage.py), the rules are a floor. Final tier = the more severe of the two, so neither can
    talk the other out of an emergency. Without an LLM answer the rules stand alone and say so."""
    tier, reasons = rules
    if llm is None:
        return tier, reasons + ["rules only (LLM unavailable) - less reliable"]
    if llm["tier"] == "emergency" and tier != "emergency":
        flags = ", ".join(llm["red_flags"]) or llm["reason"]
        return "emergency", reasons + [f"LLM flagged possible red flag ({flags}) - verify by phone"]
    final = min(tier, llm["tier"], key=TIERS.index)
    note = f"LLM: {llm['tier']} ({llm['reason']})" if llm["reason"] else f"LLM: {llm['tier']}"
    return final, [note] + [r for r in reasons if r != "no urgency signals"]


def priority_key(tier: str, score: float) -> float:
    """Sortable priority: tier dominates, conversion score breaks ties (higher = sooner)."""
    return (len(TIERS) - TIERS.index(tier)) + float(score or 0.0)
