"""Load the demo scenario (chiro/scenario.py) into the operational schema. Dev / demo only; never touches
the source dataset. Safe to re-run: appointments and the slot book are rebuilt, leads and responses are
only appended when missing (leads_raw is the pipeline's append-only streaming source).

    DATABRICKS_CONFIG_PROFILE=... DATABRICKS_WAREHOUSE_ID=... CHIRO_SCHEMA=chiro_dev \
    [OPENAI_BASE_URL=... LLM_ENDPOINT=... LLM_MAX_TOKENS=8192] python scripts/load_scenario.py [--llm] [--pipeline]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from chiro import models, scenario, urgency  # noqa: E402
from chiro.config import Settings  # noqa: E402
from chiro.intake import Intake  # noqa: E402
from chiro.schema import migrate  # noqa: E402
from chiro.sql import WarehouseSql  # noqa: E402

APPOINTMENT_SCHEMA = ("appointment_id STRING, patient_id STRING, provider_id STRING, location_id STRING, "
                      "appointment_date DATE, appointment_type STRING, booked_channel STRING, status STRING, "
                      "lead_time_days INT")
SLOT_SCHEMA = "slot_id STRING, slot_start TIMESTAMP, provider STRING, is_open BOOLEAN"
LEAD_SCHEMA = ("lead_id STRING, created_at TIMESTAMP, first_name STRING, source STRING, complaint STRING, "
               "insurance_type STRING, distance_miles DOUBLE, message STRING, consent_to_contact BOOLEAN, "
               "consent_sms BOOLEAN, consent_email BOOLEAN, first_response_hours DOUBLE, status STRING, "
               "converted BOOLEAN, num_touchpoints INT, assigned_location_id STRING")
ACTION_SCHEMA = ("action_id STRING, run_id STRING, lead_id STRING, action_type STRING, channel STRING, "
                 "priority STRING, message STRING, proposed_slot_id STRING, reasoning STRING, status STRING, "
                 "created_at TIMESTAMP, reviewed_by STRING, reviewed_at TIMESTAMP")
SCORE_SCHEMA = ("lead_id STRING, score DOUBLE, reasons STRING, red_flags STRING, urgency STRING, "
                "urgency_reasons STRING, respond_within_hours DOUBLE, priority DOUBLE")


MAX_PARAM_CHARS = 900_000  # statement parameters are capped at 1 MiB in total


def _chunks(rows: list[dict]):
    chunk, size = [], 2
    for r in rows:
        n = len(json.dumps(r, default=str)) + 1
        if chunk and size + n > MAX_PARAM_CHARS:
            yield chunk
            chunk, size = [], 2
        chunk.append(r)
        size += n
    if chunk:
        yield chunk


def insert(db, table: str, rows: list[dict], schema: str, overwrite: bool = False) -> None:
    cols = ", ".join(c.split()[0] for c in schema.split(", "))
    for i, chunk in enumerate(_chunks(rows)):
        verb = "INSERT OVERWRITE" if overwrite and i == 0 else "INSERT INTO"
        db.execute(f"{verb} {table} ({cols}) SELECT {cols} FROM (SELECT inline(from_json(:j, 'ARRAY<STRUCT<{schema}>>')))",
                   {"j": json.dumps(chunk, default=str)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="LLM second-opinion triage for the scenario leads")
    ap.add_argument("--pipeline", action="store_true", help="start a pipeline update so silver tables pick up the leads")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    s, db = Settings.from_env(), WarehouseSql(os.environ["DATABRICKS_WAREHOUSE_ID"])
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, microsecond=0)
    today = now.date()

    print("schema:", migrate(db, s) or "up to date")

    locs = [{"location_id": r["location_id"], "room_capacity": int(r["cap"]), "providers": 0} for r in db.query(
        f"SELECT location_id, capacity_patients_per_day AS cap FROM {s.source_table('locations')} ORDER BY 1")]
    providers: dict[str, list[str]] = {}
    for r in db.query(f"SELECT location_id, provider_id FROM {s.source_table('providers')} WHERE active_flag ORDER BY 2"):
        providers.setdefault(r["location_id"], []).append(r["provider_id"])
    for loc in locs:
        loc["providers"] = len(providers.get(loc["location_id"], []))
    patients: dict[str, list[str]] = {}
    for r in db.query(f"SELECT patient_id, home_location_id FROM {s.source_table('patients')} ORDER BY 1"):
        patients.setdefault(r["home_location_id"], []).append(r["patient_id"])

    start = today - dt.timedelta(weeks=scenario.SCENARIO_WEEKS)
    appts = scenario.appointments(locs, providers, patients, start, today - dt.timedelta(days=1), args.seed)
    db.execute(f"CREATE OR REPLACE TABLE {s.table('appointments')} AS SELECT * FROM {s.source_table('appointments')} "
               "WHERE appointment_date < :start", {"start": start.isoformat()})
    insert(db, s.table("appointments"), appts, APPOINTMENT_SCHEMA)
    print(f"appointments: linked history before {start}, {len(appts)} scenario rows {start}..{today}")
    for lid, prof in scenario.PROFILES.items():
        print(f"   {lid}: {prof.problem}")

    slots = scenario.slot_book(locs, providers, today, 14, start, args.seed)
    insert(db, s.table("appointment_slots"), slots, SLOT_SCHEMA, overwrite=True)
    print(f"appointment_slots: {len(slots)} slots, {sum(not x['is_open'] for x in slots)} booked")

    leads, actions = scenario.leads(scenario.load_corpus(), now, sorted(providers), seed=args.seed)
    have = {r["lead_id"] for r in db.query(f"SELECT lead_id FROM {s.table('leads_raw')} WHERE lead_id LIKE 'GEN-%'")}
    new_leads = [r for r in leads if r["lead_id"] not in have]
    insert(db, s.table("leads_raw"), new_leads, LEAD_SCHEMA)
    have = {r["action_id"] for r in db.query(f"SELECT action_id FROM {s.table('lead_actions')} WHERE run_id = :r",
                                             {"r": scenario.RUN_ID})}
    insert(db, s.table("lead_actions"), [a for a in actions if a["action_id"] not in have], ACTION_SCHEMA)
    print(f"leads_raw: +{len(new_leads)} scenario leads ({len(leads)} total); lead_actions: {len(actions)} responses")

    # Triage + score every scenario lead so queue and SLA reports use the tier the system would assign.
    llm = None
    if args.llm:
        from chiro.llm import get_llm_client
        llm = urgency.llm_triage_many(get_llm_client(), s.llm_endpoint, {r["lead_id"]: r["message"] for r in leads})
        print(f"LLM triage: {sum(v is not None for v in llm.values())}/{len(llm)} answered")
    frame = pd.DataFrame(leads).assign(status="new")  # score_leads only scores open leads; score all here
    scores = models.score_leads(Intake(db, s).train_lead_model(), frame, llm).drop(columns=["scored_at"])
    db.execute(f"DELETE FROM {s.table('lead_scores')} WHERE lead_id LIKE 'GEN-%'")
    insert(db, s.table("lead_scores"), scores.to_dict("records"), SCORE_SCHEMA)
    db.execute(f"UPDATE {s.table('lead_scores')} SET scored_at = current_timestamp() WHERE scored_at IS NULL")
    print("lead_scores:", scores["urgency"].value_counts().to_dict())

    if args.pipeline:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        p = next((p for p in w.pipelines.list_pipelines() if "chiro-lakehouse" in (p.name or "")), None)
        if p is None:
            print("pipeline not found; run it from the workspace to refresh silver tables")
        else:
            print("pipeline update started:", w.pipelines.start_update(p.pipeline_id).update_id, f"({p.name})")


if __name__ == "__main__":
    main()
