# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Link the clinic data into the landing layer
# MAGIC
# MAGIC Default (`data_source=real`) projects the real dataset in `source_catalog.source_schema`
# MAGIC (default `workspace.chiro_hackathon`) into the landing tables the pipeline reads, via `chiro/link.py`.
# MAGIC Switch to `data_source=synthetic` to fall back to the self-contained generator (`chiro/datagen.py`).

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import datetime as dt

from chiro import datagen
from chiro.config import Settings
from chiro.link import link_statements
from chiro.schema import RAW_TABLES

settings = Settings.from_widgets(dbutils)
dbutils.widgets.dropdown("data_source", "real", ["real", "synthetic"])
dbutils.widgets.text("seed", "42")

data_source = dbutils.widgets.get("data_source") or "real"

if data_source == "real":
    print(f"Linking {settings.source_catalog}.{settings.source_schema} -> "
          f"{settings.catalog}.{settings.schema}")
    for name, stmt in link_statements(settings):
        spark.sql(stmt)
        print(f"{name:<20} {spark.table(settings.table(name).replace('`', '')).count():>7,} rows")
else:
    print("Generating synthetic clinic data (no real PHI)")
    tables = datagen.generate(today=dt.date.today(), seed=int(dbutils.widgets.get("seed")))
    # leads / patients / visits land in *_raw; the pipeline builds governed silver tables from them.
    # services, price_history, appointment_slots, retention_offers are operational tables the App updates.
    for name, pdf in tables.items():
        target = RAW_TABLES.get(name, name)
        (spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true")
         .saveAsTable(settings.table(target).replace("`", "")))
        print(f"{target:<20} {len(pdf):>7,} rows")

# Offer catalogue is clinic policy (static), whichever source the data came from.
(spark.createDataFrame(datagen.RETENTION_OFFERS).write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(settings.table("retention_offers").replace("`", "")))

spark.sql(f"ALTER TABLE {settings.table('services')} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
spark.sql(f"COMMENT ON TABLE {settings.table('leads_raw')} IS "
          "'Landing: inbound leads (linked from the clinic dataset). Append-only.'")
