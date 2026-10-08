# Databricks notebook source
# MAGIC %md
# MAGIC # 11 · Draft care guidance for patients whose notes changed
# MAGIC Staff care notes + attendance signals -> a staff version and a plain-language patient version per patient,
# MAGIC grounded in their sources and checked in code. Drafts wait for staff approval in Clinic Copilot.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from chiro.care import CareReports
from chiro.config import Settings
from chiro.llm import get_llm_client
from chiro.sql import SparkSql
from chiro.tools import ClinicTools
from chiro.workflows import new_run_id

settings = Settings.from_widgets(dbutils)
dbutils.widgets.text("max_items", "10")
db = SparkSql(spark)
run_id = new_run_id("care")
results = CareReports(db, settings, run_id).draft_due(get_llm_client(), settings.llm_endpoint,
                                                       ClinicTools(db, settings, run_id),
                                                       limit=int(dbutils.widgets.get("max_items")))
for r in results:
    print(r)
drafted = sum(1 for r in results if r.get("report_id"))
print(f"drafted {drafted} of {len(results)} due reports")
dbutils.notebook.exit(f"drafted {drafted}/{len(results)}")
