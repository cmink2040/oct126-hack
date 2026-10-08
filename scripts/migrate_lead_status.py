"""One-off: apply the Contacted / Qualified -> 'contacted' status mapping to leads already landed, without
re-running setup (which would rebuild leads_raw and drop anything appended since, e.g. the demo scenario).

Updates leads_raw in place, then fully refreshes the pipeline's `leads` streaming table - required, because a
streaming source that was updated (not appended to) fails the next incremental run.

    DATABRICKS_CONFIG_PROFILE=... DATABRICKS_WAREHOUSE_ID=... CHIRO_SCHEMA=chiro_dev python scripts/migrate_lead_status.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databricks.sdk import WorkspaceClient  # noqa: E402

from chiro.config import Settings  # noqa: E402
from chiro.link import contacted_status_fix  # noqa: E402
from chiro.sql import WarehouseSql  # noqa: E402


def counts(db, s) -> dict:
    return {r["status"]: r["n"] for r in db.query(f"SELECT status, count(*) AS n FROM {s.table('leads_raw')} GROUP BY 1")}


def main() -> None:
    s, db, w = Settings.from_env(), WarehouseSql(os.environ["DATABRICKS_WAREHOUSE_ID"]), WorkspaceClient()
    print("leads_raw before:", counts(db, s))
    db.execute(contacted_status_fix(s))
    print("leads_raw after: ", counts(db, s))

    pipeline = next((p for p in w.pipelines.list_pipelines() if "chiro-lakehouse" in (p.name or "")), None)
    if pipeline is None:
        sys.exit("pipeline not found: fully refresh the `leads` table before the next pipeline run")
    update = w.pipelines.start_update(pipeline.pipeline_id, full_refresh_selection=["leads"]).update_id
    print(f"full refresh of `leads` started ({pipeline.name}, update {update})")
    while True:
        state = w.pipelines.get_update(pipeline.pipeline_id, update).update.state.value
        if state in ("COMPLETED", "FAILED", "CANCELED"):
            break
        time.sleep(20)
    print("pipeline:", state)
    if state == "COMPLETED":
        print("silver leads:", {r["status"]: r["n"] for r in db.query(
            f"SELECT status, count(*) AS n FROM {s.table('leads')} GROUP BY 1")})


if __name__ == "__main__":
    main()
