# Databricks notebook source
# MAGIC %md
# MAGIC # 30 · Grant the Clinic Copilot app access to the schema
# MAGIC Databricks Apps run as their own service principal. This grants it least-privilege access in Unity Catalog.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from databricks.sdk import WorkspaceClient

from chiro.config import Settings

settings = Settings.from_widgets(dbutils)
dbutils.widgets.text("app_name", "")
app_name = dbutils.widgets.get("app_name")

try:
    sp = WorkspaceClient().apps.get(app_name).service_principal_client_id
except Exception as e:
    dbutils.notebook.exit(f"App '{app_name}' not found yet ({e}); run `databricks bundle deploy` first, then rerun.")

schema = f"`{settings.catalog}`.`{settings.schema}`"
grants = [
    f"GRANT USE CATALOG ON CATALOG `{settings.catalog}` TO `{sp}`",
    f"GRANT USE SCHEMA, SELECT ON SCHEMA {schema} TO `{sp}`",
    # Writes are limited to the review queues and the tables an approval updates.
    *[f"GRANT MODIFY ON TABLE {settings.table(t)} TO `{sp}`"
      for t in ("lead_actions", "retention_actions", "price_recommendations",
                "services", "service_pricing_stats", "price_history")],
]
for g in grants:
    try:
        spark.sql(g)
        print("OK  ", g)
    except Exception as e:  # e.g. USE CATALOG already granted to all workspace users
        print("SKIP", g, "->", str(e).splitlines()[0])
