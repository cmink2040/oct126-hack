"""Write a corpus of realistic inbound lead messages with an intended urgency tier, using an LLM once.

The output (src/chiro/data/lead_messages.json) is committed so the demo scenario is reproducible
without an LLM. Labels are the tier the message was written to express, not a classifier's output.

    OPENAI_BASE_URL=... LLM_ENDPOINT=... LLM_MAX_TOKENS=8192 [LLM_NO_THINKING=1] python scripts/gen_lead_corpus.py
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chiro.llm import get_llm_client, max_output_tokens, request_options  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "chiro", "data", "lead_messages.json")
PARTIAL = OUT + ".partial.jsonl"
COMPLAINTS = ["low_back_pain", "neck_pain", "headaches", "sciatica", "sports_injury", "auto_injury", "posture_wellness"]
TIERS = {
    "emergency": ("possible red-flag symptoms alongside their pain (bladder/bowel changes, numbness in the groin or "
                  "'down there', sudden worst-ever headache, chest pain or pressure, fever/chills, legs or arms getting "
                  "weaker or numb, fainting, unexplained weight loss). Most people don't realise it's serious.", 3),
    "urgent": ("acute problems from the last day or two, or severe impact (can't walk/sleep/work, pain 7-10/10, "
               "recent accident or fall with strong pain).", 12),
    "soon": ("recent, worsening or moderate problems (days to a couple of weeks), flare-ups, wanting to be seen "
             "this week, mild-to-moderate pain after a minor incident.", 15),
    "routine": ("long-standing or mild issues, wellness/maintenance, posture, price/insurance/hours questions, "
                "people who say nothing hurts much. Some explicitly deny red flags ('no numbness, no fever').", 15),
}
PROMPT = """Write {n} different messages that people send to a chiropractic clinic's website form.
Main complaint: {complaint}. Every message expresses this urgency: {desc}
Vary length (5-60 words), tone, spelling (some typos, lowercase, no punctuation), who is writing (worker,
parent, athlete, retiree), and add practical questions (price, insurance, hours, parking) to some.
Don't use the words emergency, urgent, routine or red flag. Return ONLY a JSON array of strings."""


def main() -> None:
    client, model = get_llm_client(), os.environ["LLM_ENDPOINT"]
    extra = request_options(thinking=False)  # writing varied examples needs no deliberation
    corpus, done = [], set()
    if os.path.exists(PARTIAL):
        with open(PARTIAL) as f:
            for line in f:
                b = json.loads(line)
                done.add((b["tier"], b["complaint"]))
                corpus += b["items"]
    for tier, (desc, n) in TIERS.items():
        for complaint in COMPLAINTS:
            if (tier, complaint) in done:
                continue
            resp = client.chat.completions.create(
                model=model, temperature=0.9, max_tokens=max_output_tokens(),
                messages=[{"role": "user", "content": PROMPT.format(n=n, complaint=complaint, desc=desc)}],
                **extra)
            try:
                msgs = json.loads(re.search(r"\[.*\]", resp.choices[0].message.content or "", re.S).group(0))
            except Exception as e:
                print(f"skip {tier}/{complaint}: {e}")
                continue
            batch = [{"message": m.strip(), "tier": tier, "complaint": complaint}
                     for m in msgs if isinstance(m, str) and m.strip()]
            corpus += batch
            with open(PARTIAL, "a") as f:  # resumable: one line per finished batch
                f.write(json.dumps({"tier": tier, "complaint": complaint, "items": batch}) + "\n")
            print(f"{tier}/{complaint}: {len(msgs)}", flush=True)
    seen, unique = set(), []
    for c in corpus:
        if c["message"].lower() not in seen:
            seen.add(c["message"].lower())
            unique.append(c)
    with open(OUT, "w") as f:
        json.dump(unique, f, indent=1)
    os.remove(PARTIAL)
    print(f"wrote {len(unique)} messages to {OUT}")


if __name__ == "__main__":
    main()
