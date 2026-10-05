# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Run an agent
# MAGIC One notebook for every agent (`agent` = lead | retention | pricing | briefing).
# MAGIC The agent reasons with a Foundation Model API endpoint, acts only through governed tools over Unity Catalog,
# MAGIC writes *pending_review* actions, and is fully traced in MLflow.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-jobs.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from chiro.config import Settings
from chiro.llm import get_llm_client
from chiro.sql import SparkSql
from chiro.tracing import setup_experiment
from chiro.workflows import run_named_agent

settings = Settings.from_widgets(dbutils)
dbutils.widgets.dropdown("agent", "lead", ["lead", "retention", "pricing", "briefing"])
dbutils.widgets.text("max_items", "10")
dbutils.widgets.text("experiment", "")
setup_experiment(dbutils.widgets.get("experiment"))

agent = dbutils.widgets.get("agent")
result = run_named_agent(agent, SparkSql(spark), settings, get_llm_client(),
                         max_items=int(dbutils.widgets.get("max_items")))
print(f"[{agent}] status={result.status} steps={result.steps} tool_calls={len(result.tool_calls)}")
print(result.final_text)
dbutils.notebook.exit(result.status)
