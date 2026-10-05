-- Lakeflow Declarative Pipeline · Silver layer
-- Raw landing tables (*_raw) are written by the seed / ingestion notebooks. Everything below is
-- declarative: Databricks manages dependencies, incremental processing and data-quality metrics.

-- Leads: a STREAMING table, so each lead is processed exactly once. That makes the AI Functions
-- enrichment incremental - every new lead is classified once, never re-billed on refresh.
CREATE OR REFRESH STREAMING TABLE leads (
  CONSTRAINT valid_lead_id   EXPECT (lead_id IS NOT NULL)                ON VIOLATION DROP ROW,
  CONSTRAINT has_message     EXPECT (length(trim(message)) > 0)          ON VIOLATION DROP ROW,
  CONSTRAINT known_source    EXPECT (source IN ('social_media_ad', 'referral', 'walk_in', 'phone_inquiry',
                                                  'website_form', 'community_event', 'insurance_directory')),
  CONSTRAINT sane_distance   EXPECT (distance_miles BETWEEN 0 AND 200)
)
CLUSTER BY (created_at)
COMMENT 'Inbound leads, enriched with Databricks AI Functions (intent + extracted details) for recent leads.'
AS SELECT
  *,
  -- AI enrichment only for recent leads keeps Free Edition token usage small on the first full refresh.
  CASE WHEN created_at >= current_timestamp() - INTERVAL 90 DAYS THEN
    ai_classify(message, ARRAY('ready_to_book', 'price_shopping', 'insurance_question', 'general_info', 'urgent_medical'))
  END AS ai_intent,
  CASE WHEN created_at >= current_timestamp() - INTERVAL 90 DAYS THEN
    ai_extract(message, ARRAY('body_area', 'symptom_duration', 'insurance_question'))
  END AS ai_details
FROM STREAM(leads_raw);

CREATE OR REFRESH MATERIALIZED VIEW patients (
  CONSTRAINT valid_patient_id EXPECT (patient_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT medicare_age     EXPECT (insurance_type <> 'medicare' OR age >= 65),
  CONSTRAINT known_payer      EXPECT (insurance_type IN ('cash', 'ppo', 'medicare'))
)
COMMENT 'Patient master (synthetic; first names only).'
AS SELECT * FROM patients_raw;

CREATE OR REFRESH MATERIALIZED VIEW visits (
  CONSTRAINT valid_visit_id   EXPECT (visit_id IS NOT NULL)                          ON VIOLATION DROP ROW,
  CONSTRAINT known_patient    EXPECT (patient_id IS NOT NULL)                        ON VIOLATION DROP ROW,
  CONSTRAINT valid_status     EXPECT (status IN ('completed', 'no_show', 'cancelled')) ON VIOLATION DROP ROW,
  CONSTRAINT non_negative_pay EXPECT (price_paid >= 0)                               ON VIOLATION DROP ROW,
  CONSTRAINT not_in_future    EXPECT (visit_date <= current_date())
)
CLUSTER BY (visit_date, patient_id)
COMMENT 'Visit ledger. price_paid = cash price or insurer allowed amount.'
AS SELECT * FROM visits_raw;
