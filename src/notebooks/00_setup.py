# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the Unity Catalog schema and the agent action/audit tables. Set `reset=true` to drop everything first.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from chiro.config import Settings
from chiro.schema import ddl_statements

settings = Settings.from_widgets(dbutils)
dbutils.widgets.dropdown("reset", "false", ["false", "true"])

if dbutils.widgets.get("reset") == "true":
    spark.sql(f"DROP SCHEMA IF EXISTS `{settings.catalog}`.`{settings.schema}` CASCADE")

for stmt in ddl_statements(settings):
    spark.sql(stmt)
display(spark.sql(f"SHOW TABLES IN `{settings.catalog}`.`{settings.schema}`"))
