"""Evaluate urgency triage against the hand-labelled golden set.

    python scripts/eval_triage.py            # rules only
    python scripts/eval_triage.py --llm      # rules, LLM and the ensemble (needs an LLM endpoint configured)

Reports accuracy, per-tier precision/recall, and the confusion matrix. Emergency recall is the number that
matters most: a missed red flag is far worse than an extra phone call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chiro import urgency  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "tests", "data")
SETS = {"golden": "triage_golden.jsonl",  # used while developing the rules
        "holdout": "triage_holdout.jsonl"}  # written separately, never tuned against


def load(name: str = "golden") -> list[dict]:
    with open(os.path.join(DATA, SETS[name])) as f:
        return [json.loads(line) for line in f if line.strip()]


def report(name: str, cases: list[dict], predicted: list[str]) -> dict:
    gold = [c["tier"] for c in cases]
    acc = sum(g == p for g, p in zip(gold, predicted)) / len(gold)
    print(f"\n== {name}: accuracy {acc:.0%} ({len(gold)} cases)")
    print(f"{'tier':<10}{'precision':>10}{'recall':>8}   confusion (rows=gold, cols={' '.join(t[:4] for t in urgency.TIERS)})")
    conf = Counter(zip(gold, predicted))
    stats = {}
    for t in urgency.TIERS:
        tp, fp = conf[(t, t)], sum(conf[(g, t)] for g in urgency.TIERS if g != t)
        fn = sum(conf[(t, p)] for p in urgency.TIERS if p != t)
        prec, rec = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        stats[t] = (prec, rec)
        row = " ".join(f"{conf[(t, p)]:>4}" for p in urgency.TIERS)
        print(f"{t:<10}{prec:>10.0%}{rec:>8.0%}   {row}")
    misses = [(c["message"], c["tier"], p) for c, p in zip(cases, predicted) if c["tier"] != p]
    for msg, g, p in misses:
        print(f"   miss: gold={g:<9} got={p:<9} {msg}")
    return {"accuracy": acc, "per_tier": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also evaluate the LLM triage and the ensemble")
    ap.add_argument("--set", choices=sorted(SETS), default="holdout")
    args = ap.parse_args()
    cases = load(args.set)
    print(f"set: {args.set}")
    report("rules", cases, [urgency.classify(c["message"], c["complaint"])[0] for c in cases])
    if args.llm:
        from chiro.llm import get_llm_client

        client, model = get_llm_client(), os.environ["LLM_ENDPOINT"]
        llm = [urgency.llm_triage(client, model, c["message"]) for c in cases]
        report("llm", cases, [r["tier"] if r else "routine" for r in llm])
        report("ensemble", cases, [urgency.combine(urgency.classify(c["message"], c["complaint"]), r)[0]
                                   for c, r in zip(cases, llm)])


if __name__ == "__main__":
    main()
