# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Simulate today's inbound activity (demo feed)
# MAGIC Stands in for real ingestion (web forms, ads, phone system, EHR/PM export) so the daily agent loop always has fresh work.
# MAGIC Skipped when `data_source=real` (the linked clinic dataset is the source of truth), or when
# MAGIC `simulate_activity=false`.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import datetime as dt

import numpy as np

from chiro import datagen
from chiro.config import Settings

settings = Settings.from_widgets(dbutils)
dbutils.widgets.dropdown("data_source", "real", ["real", "synthetic"])
dbutils.widgets.dropdown("simulate_activity", "true", ["true", "false"])

if dbutils.widgets.get("data_source") == "real":
    print("data_source=real: skipping simulated activity (real clinic dataset is the source of truth)")
elif dbutils.widgets.get("simulate_activity") == "true":
    now = dt.datetime.now()
    seed = int(now.strftime("%Y%m%d%H"))
    raw = settings.table("leads_raw").replace("`", "")
    n_existing = spark.table(raw).count()
    leads = datagen.simulate_new_leads(now, n=int(np.random.default_rng(seed).integers(10, 20)),
                                       seed=seed, id_offset=n_existing)
    spark.createDataFrame(leads).write.mode("append").saveAsTable(raw)  # streaming source: append only
    slots = datagen.appointment_slots(np.random.default_rng(seed), now)
    spark.createDataFrame(slots).write.mode("overwrite").saveAsTable(settings.table("appointment_slots").replace("`", ""))
    print(f"Appended {len(leads)} leads; refreshed {len(slots)} appointment slots")
else:
    print("Simulation disabled")
