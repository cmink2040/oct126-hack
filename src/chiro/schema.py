"""Unity Catalog DDL for the tables the agents write. Base/gold tables are written from DataFrames."""
from __future__ import annotations

from chiro.config import Settings

ACTION_TABLES = {
    "patient_accounts": """
        account_id STRING, email STRING, password_hash STRING, first_name STRING, phone STRING,
        consent_to_contact BOOLEAN, consent_sms BOOLEAN, consent_email BOOLEAN, created_at TIMESTAMP,
        patient_id STRING""",
    "capacity_actions": """
        action_id STRING, run_id STRING, location_id STRING, action_type STRING, details STRING,
        expected_weekly_visits DOUBLE, expected_weekly_revenue DOUBLE, reasoning STRING, status STRING,
        created_at TIMESTAMP, reviewed_by STRING, reviewed_at TIMESTAMP, targets STRING, target_date DATE,
        baseline_utilization DOUBLE""",
    "campaign_targets": """
        action_id STRING, patient_id STRING, location_id STRING, target_date DATE, status STRING,
        retention_action_id STRING, created_at TIMESTAMP""",
    "care_notes": """
        note_id STRING, patient_id STRING, author STRING, note STRING, created_at TIMESTAMP, category STRING,
        advice STRING, if_ignored STRING, importance STRING""",
    "care_reports": """
        report_id STRING, run_id STRING, patient_id STRING, staff_report STRING, patient_report STRING,
        status STRING, created_at TIMESTAMP, reviewed_by STRING, reviewed_at TIMESTAMP, items STRING,
        version INT, quality STRING, viewed_at TIMESTAMP""",
    "care_report_responses": """
        response_id STRING, report_id STRING, patient_id STRING, item_key STRING, response STRING,
        comment STRING, created_at TIMESTAMP""",
    "account_leads": """
        account_id STRING, lead_id STRING, submitted_at TIMESTAMP""",
    "lead_actions": """
        action_id STRING, run_id STRING, lead_id STRING, action_type STRING, channel STRING,
        priority STRING, message STRING, proposed_slot_id STRING, reasoning STRING,
        status STRING, created_at TIMESTAMP, reviewed_by STRING, reviewed_at TIMESTAMP""",
    "retention_actions": """
        action_id STRING, run_id STRING, patient_id STRING, channel STRING, offer_code STRING,
        message STRING, reasoning STRING, churn_risk DOUBLE, status STRING, created_at TIMESTAMP,
        reviewed_by STRING, reviewed_at TIMESTAMP, proposed_slot_id STRING, campaign_action_id STRING""",
    "price_recommendations": """
        rec_id STRING, run_id STRING, service_id STRING, current_price DOUBLE, proposed_price DOUBLE,
        projected_weekly_margin_delta DOUBLE, projected_weekly_volume_delta DOUBLE, rationale STRING,
        status STRING, created_at TIMESTAMP, reviewed_by STRING, reviewed_at TIMESTAMP""",
    "agent_runs": """
        run_id STRING, agent STRING, started_at TIMESTAMP, finished_at TIMESTAMP, status STRING,
        steps INT, tool_calls INT, failed_tool_calls INT, summary STRING""",
    "daily_briefings": """
        briefing_date DATE, run_id STRING, content STRING, created_at TIMESTAMP""",
}

COMMENTS = {
    "patient_accounts": "Patient portal accounts (scrypt password hashes). Consent here is copied onto each inquiry.",
    "account_leads": "Which portal account submitted which lead.",
    "capacity_actions": "Corrective actions for under-used capacity drafted by the Capacity agent; pending review.",
    "campaign_targets": "Patients in approved fill campaigns, worked by the Retention agent (to_contact -> drafted).",
    "care_notes": "Staff care notes: follow-through advice and consequences of skipping it. Not diagnostic.",
    "care_reports": "Care guidance drafted from care notes and attendance; shown to the patient only after approval. "
                    "A newer approved version supersedes the old one.",
    "care_report_responses": "Patient answers per guidance item (on_it / need_help / not_relevant), from the portal.",
    "lead_actions": "Outreach drafted by the Lead agent. Nothing is sent until a human approves it in the App.",
    "retention_actions": "Re-engagement drafted by the Retention agent; pending human review.",
    "price_recommendations": "Cash-pay price changes proposed by the Pricing agent; applied only on approval.",
    "agent_runs": "Audit log of every agent run.",
    "daily_briefings": "Morning briefing written by the Briefing agent.",
}


CLUSTER_KEYS = {
    "patient_accounts": "email",
    "account_leads": "account_id",
    "capacity_actions": "status, location_id",
    "campaign_targets": "status, location_id",
    "care_notes": "patient_id",
    "care_reports": "status, patient_id",
    "care_report_responses": "patient_id",
    "lead_actions": "status, lead_id",
    "retention_actions": "status, patient_id",
    "price_recommendations": "status, service_id",
    "agent_runs": "started_at",
    "daily_briefings": "briefing_date",
}

# Landing tables written by seed / ingestion; the Lakeflow pipeline builds the silver tables from them.
RAW_TABLES = {"leads": "leads_raw", "patients": "patients_raw", "visits": "visits_raw"}


def ddl_statements(s: Settings) -> list[str]:
    stmts = [f"CREATE SCHEMA IF NOT EXISTS `{s.catalog}`.`{s.schema}` "
             f"COMMENT 'Chiropractic clinic agentic AI (linked from workspace.chiro_hackathon)'"]
    for name, cols in ACTION_TABLES.items():
        # Liquid clustering for queue lookups; Change Data Feed gives a replayable audit trail of
        # every agent draft and human decision (table_changes()).
        stmts.append(
            f"CREATE TABLE IF NOT EXISTS {s.table(name)} ({cols.strip()}) "
            f"CLUSTER BY ({CLUSTER_KEYS[name]}) COMMENT '{COMMENTS[name]}' "
            "TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
    return stmts


# Columns added to tables written from DataFrames, for schemas created before the column existed.
EXTRA_COLUMNS = {
    "lead_scores": "urgency STRING, urgency_reasons STRING, respond_within_hours DOUBLE, priority DOUBLE",
}


def migrate(db, s: Settings) -> list[str]:
    """Bring an existing schema up to date: create missing action tables, add missing columns. Idempotent."""
    done = []
    for stmt in ddl_statements(s)[1:]:
        db.execute(stmt)
    wanted = {name: cols for name, cols in ACTION_TABLES.items()} | EXTRA_COLUMNS
    for name, cols in wanted.items():
        try:
            have = {r["col_name"].lower() for r in db.query(f"DESCRIBE {s.table(name)}")}
        except Exception:  # table written by a job that hasn't run yet
            continue
        missing = [c.strip() for c in cols.split(",") if c.strip() and c.split()[0].lower() not in have]
        if missing:
            db.execute(f"ALTER TABLE {s.table(name)} ADD COLUMNS ({', '.join(missing)})")
            done.append(f"{name}: +{', '.join(c.split()[0] for c in missing)}")
    return done
