"""Write a corpus of realistic staff care notes (advice + consequence of ignoring it), using an LLM once.

Output src/chiro/data/care_notes.json is committed so the demo scenario is reproducible without an LLM.
Notes that fail the care-language checks are dropped: staff notes are follow-through advice, not diagnoses.

    OPENAI_BASE_URL=... LLM_ENDPOINT=... LLM_NO_THINKING=1 python scripts/gen_care_corpus.py
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chiro.care import CATEGORIES, check_care_language  # noqa: E402
from chiro.llm import get_llm_client, max_output_tokens, request_options  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "chiro", "data", "care_notes.json")
COMPLAINTS = ["low_back_pain", "neck_pain", "headaches", "sciatica", "sports_injury", "auto_injury", "posture_wellness"]
PROMPT = """You are a chiropractor writing short follow-through notes in a patient's chart after a visit.
Complaint: {complaint}. Category: {category} ({desc}).
Write {n} different notes. Each has:
 - "advice": one concrete thing the patient should keep doing (specific: what, how often, when), <= 25 words
 - "if_ignored": the realistic, practical consequence of not doing it (symptoms returning, progress slowing,
   needing extra visits or a re-check, flare-ups) - hedged ("tends to", "often", "may"), <= 25 words
 - "importance": "high", "medium" or "low"
Rules: no diagnoses, no medication advice, no certainty ("will"), no scare tactics, plain words.
Return ONLY a JSON array of objects with keys advice, if_ignored, importance."""


def main() -> None:
    client, model = get_llm_client(), os.environ["LLM_ENDPOINT"]
    corpus, dropped = [], 0
    for category, desc in CATEGORIES.items():
        for complaint in COMPLAINTS:
            resp = client.chat.completions.create(
                model=model, temperature=0.8, max_tokens=max_output_tokens(), **request_options(thinking=False),
                messages=[{"role": "user", "content": PROMPT.format(n=4, complaint=complaint, category=category,
                                                                    desc=desc)}])
            try:
                items = json.loads(re.search(r"\[.*\]", resp.choices[0].message.content or "", re.S).group(0))
            except Exception as e:
                print(f"skip {category}/{complaint}: {e}")
                continue
            for it in items:
                if not isinstance(it, dict) or not it.get("advice") or not it.get("if_ignored"):
                    continue
                if check_care_language(it["advice"]) or check_care_language(it["if_ignored"]):
                    dropped += 1
                    continue
                corpus.append({"category": category, "complaint": complaint, "advice": it["advice"].strip(),
                               "if_ignored": it["if_ignored"].strip(),
                               "importance": it.get("importance") if it.get("importance") in ("high", "medium", "low")
                               else "medium"})
            print(f"{category}/{complaint}: {len(items)}", flush=True)
    with open(OUT, "w") as f:
        json.dump(corpus, f, indent=1)
    print(f"wrote {len(corpus)} notes ({dropped} dropped by the language checks) to {OUT}")


if __name__ == "__main__":
    main()
