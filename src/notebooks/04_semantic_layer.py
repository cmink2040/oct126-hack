# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Semantic layer: Unity Catalog metric views
# MAGIC Creates the governed metric views (`clinic_visit_metrics`, `clinic_lead_metrics`) from `chiro/semantic.py`,
# MAGIC tags tables in Unity Catalog, and smoke-tests a `MEASURE()` query. The agents, the AI/BI dashboard and
# MAGIC Genie all read these views, so every surface reports identical numbers.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from chiro import semantic
from chiro.config import Settings
from chiro.sql import SparkSql

settings = Settings.from_widgets(dbutils)

for mv in semantic.METRIC_VIEWS.values():
    spark.sql(mv.ddl(settings))
    try:
        spark.sql(f"COMMENT ON TABLE {settings.table(mv.name)} IS '{mv.comment}'")
    except Exception as e:
        print(f"comment skipped on {mv.name}: {e}")
    print(f"created metric view {mv.name}")

# COMMAND ----------

# Unity Catalog tags make the data discoverable and classify it clearly.
for table, tags in {
    "patients": {"domain": "clinical_ops", "data_class": "sample_phi"},
    "visits": {"domain": "clinical_ops", "data_class": "sample_phi"},
    "leads": {"domain": "growth", "data_class": "sample_pii"},
    "gold_visit_facts": {"domain": "analytics", "layer": "gold"},
    "gold_lead_facts": {"domain": "analytics", "layer": "gold"},
}.items():
    pairs = ", ".join(f"'{k}' = '{v}'" for k, v in tags.items())
    try:
        spark.sql(f"ALTER TABLE {settings.table(table)} SET TAGS ({pairs})")
    except Exception as e:  # pipeline-owned tables may need tags set by the pipeline owner
        print(f"tags skipped on {table}: {str(e).splitlines()[0]}")

# COMMAND ----------

sql, params = semantic.build_metric_query(
    settings, "clinic_visit_metrics", ["revenue", "contribution_margin", "active_patients", "no_show_rate"],
    ["visit_month"], last_n_days=180)
print(sql)
display(spark.sql(sql, args=params))

sql, params = semantic.build_metric_query(
    settings, "clinic_lead_metrics", ["leads", "conversion_rate"], ["source"], order_by="conversion_rate")
display(SparkSql(spark).query(sql, params))
