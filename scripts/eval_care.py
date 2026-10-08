"""Evaluate care guidance.

    python scripts/eval_care.py                 # language checker vs hand-labelled sentences (a development
                                                # set: the checker's rules were widened against it)
    python scripts/eval_care.py --drafts 10     # + draft quality on real patients (needs Databricks + an LLM)

Drafts are generated for patients with care notes and measured, not saved: first-pass rate, attempts, rejections
and why, reading grade, grounding, items per report, and how many items cite attendance signals.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chiro import care  # noqa: E402

LABELS = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "care_language.jsonl")


def eval_checker() -> None:
    cases = [json.loads(line) for line in open(LABELS) if line.strip()]
    flagged = [bool(care.check_care_language(c["text"])) for c in cases]
    tp = sum(f and not c["ok"] for f, c in zip(flagged, cases))
    fp = sum(f and c["ok"] for f, c in zip(flagged, cases))
    fn = sum(not f and not c["ok"] for f, c in zip(flagged, cases))
    print(f"== language checker on {len(cases)} sentences: precision {tp / max(tp + fp, 1):.0%}, "
          f"recall {tp / max(tp + fn, 1):.0%} ({fp} false alarms, {fn} misses)")
    for f, c in zip(flagged, cases):
        if f == c["ok"]:
            print(f"   {'false alarm' if f else 'missed':<11} {c.get('why', ''):<32} {c['text']}")


class Dry:
    """Wraps a SqlRunner so drafts are measured but never written."""

    def __init__(self, db):
        self.db = db

    def query(self, sql, params=None):
        return self.db.query(sql, params)

    def execute(self, sql, params=None):
        pass


def eval_drafts(n: int) -> None:
    from chiro.config import Settings
    from chiro.llm import get_llm_client
    from chiro.sql import WarehouseSql
    from chiro.tools import ClinicTools

    s = Settings.from_env()
    db = Dry(WarehouseSql(os.environ["DATABRICKS_WAREHOUSE_ID"]))
    tools, reports, client = ClinicTools(db, s, "eval"), care.CareReports(db, s, "eval"), get_llm_client()
    patients = [r["patient_id"] for r in db.query(
        f"SELECT DISTINCT patient_id FROM {s.table('care_notes')} ORDER BY patient_id LIMIT {int(n)}")]
    results, reasons = [], Counter()
    for pid in patients:
        out = reports.draft(client, s.llm_endpoint, tools.get_patient_profile(pid), tools.get_care_recommendations(pid))
        results.append(out)
        if "error" in out:
            reasons.update(r.split(" (")[0].split(":")[0] for r in out["error"].removeprefix("draft rejected: ").split("; "))
        print(f"   {pid}: " + (out["error"][:120] if "error" in out else
                               f"v{out['version']} grade {out['quality']['reading_grade']} "
                               f"grounding {out['quality']['mean_grounding']} attempts {out['quality']['attempts']}"))
    ok = [r for r in results if "ok" in r]
    print(f"\n== drafts for {len(results)} patients: {len(ok)} accepted "
          f"({sum(r['quality']['attempts'] == 1 for r in ok)} on the first attempt), {len(results) - len(ok)} rejected")
    if ok:
        q = [r["quality"] for r in ok]
        print(f"   reading grade median {statistics.median(x['reading_grade'] for x in q)}, "
              f"max {max(x['reading_grade'] for x in q)}; grounding mean "
              f"{statistics.mean(x['mean_grounding'] for x in q):.2f}, min {min(x['min_grounding'] for x in q):.2f}")
    if reasons:
        print("   rejection reasons:", dict(reasons))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", type=int, default=0)
    args = ap.parse_args()
    eval_checker()
    if args.drafts:
        eval_drafts(args.drafts)
