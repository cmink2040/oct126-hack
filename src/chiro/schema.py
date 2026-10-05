"""Unity Catalog DDL for the tables the agents write. Base/gold tables are written from DataFrames."""
from __future__ import annotations

from chiro.config import Settings

ACTION_TABLES = {
    "lead_actions": """
        action_id STRING, run_id STRING, lead_id STRING, action_type STRING, channel STRING,
        priority STRING, message STRING, proposed_slot_id STRING, reasoning STRING,
        status STRING, created_at TIMESTAMP, reviewed_by STRING, reviewed_at TIMESTAMP""",
    "retention_actions": """
        action_id STRING, run_id STRING, patient_id STRING, channel STRING, offer_code STRING,
        message STRING, reasoning STRING, churn_risk DOUBLE, status STRING, created_at TIMESTAMP,
        reviewed_by STRING, reviewed_at TIMESTAMP""",
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
    "lead_actions": "Outreach drafted by the Lead agent. Nothing is sent until a human approves it in the App.",
    "retention_actions": "Re-engagement drafted by the Retention agent; pending human review.",
    "price_recommendations": "Cash-pay price changes proposed by the Pricing agent; applied only on approval.",
    "agent_runs": "Audit log of every agent run.",
    "daily_briefings": "Morning briefing written by the Briefing agent.",
}


CLUSTER_KEYS = {
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
