# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Refresh predictive models → gold tables
# MAGIC Reads the pipeline's silver tables, trains lead-scoring and churn models, estimates price elasticity, logs everything to MLflow
# MAGIC (models registered in Unity Catalog), and writes the gold tables the agents read.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import datetime as dt

import mlflow
import pandas as pd

from chiro import models, urgency
from chiro.config import Settings
from chiro.llm import get_llm_client
from chiro.priority import ACTIVE_LEAD_DAYS
from chiro.tracing import setup_experiment

settings = Settings.from_widgets(dbutils)
dbutils.widgets.text("experiment", "")
setup_experiment(dbutils.widgets.get("experiment"))
mlflow.set_registry_uri("databricks-uc")


def read(name):
    return spark.table(settings.table(name).replace("`", "")).toPandas()


def write(name, pdf, mode="overwrite"):
    (spark.createDataFrame(pdf).write.mode(mode).option("overwriteSchema", "true")
     .saveAsTable(settings.table(name).replace("`", "")))


today = dt.date.today()
leads, patients, visits = read("leads"), read("patients"), read("visits")
services, price_history = read("services"), read("price_history")

# COMMAND ----------


def log_model(model, X_example, name, metrics):
    """Log + register in UC. Registration is best-effort so a registry hiccup never blocks the agents."""
    mlflow.log_metrics(metrics)
    kwargs = dict(input_example=X_example.head(5),
                  registered_model_name=f"{settings.catalog}.{settings.schema}.{name}")
    try:
        try:
            mlflow.sklearn.log_model(model, name=name, **kwargs)  # MLflow 3
        except TypeError:
            mlflow.sklearn.log_model(model, artifact_path=name, **kwargs)  # MLflow 2
    except Exception as e:
        print(f"Model registration skipped for {name}: {e}")


with mlflow.start_run(run_name=f"refresh-{today}"):
    lead_model, lead_metrics = models.train_lead_model(leads)
    log_model(lead_model, models.lead_features(leads)[lead_model.feature_columns_], "lead_scoring", lead_metrics)
    # Tiers are assigned once and kept, so only leads never triaged before get the LLM second opinion.
    previous = read("lead_scores") if spark.catalog.tableExists(settings.table("lead_scores").replace("`", "")) else None
    todo = models.leads_needing_triage(previous, leads, ACTIVE_LEAD_DAYS)
    llm = urgency.llm_triage_many(get_llm_client(), settings.llm_endpoint, dict(zip(todo["lead_id"], todo["message"])))
    mlflow.log_metrics({"llm_triaged": len(llm), "llm_triage_failures": sum(v is None for v in llm.values())})
    # Materialise before overwriting the table we just read from.
    write("lead_scores", models.refresh_lead_scores(previous, lead_model, leads, llm, ACTIVE_LEAD_DAYS).copy())

    churn_model, churn_metrics = models.train_churn_model(patients, visits, today)
    churn_X = models.churn_features(patients, visits, today)[models.CHURN_FEATURES].astype(float)
    log_model(churn_model, churn_X, "churn_risk", churn_metrics)
    write("patient_churn_risk", models.score_churn(churn_model, patients, visits, today))

    elasticity = models.estimate_elasticity(services, price_history, visits)
    mlflow.log_table(elasticity, "elasticity.json")
    write("service_pricing_stats", models.pricing_stats(services, visits, elasticity, today))

    metrics = {**lead_metrics, **churn_metrics}
    write("model_metrics", pd.DataFrame([{"trained_at": pd.Timestamp.now(), "metric": k, "value": float(v)}
                                         for k, v in metrics.items()]), mode="append")
print(metrics)
display(elasticity)
