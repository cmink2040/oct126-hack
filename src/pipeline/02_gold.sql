-- Lakeflow Declarative Pipeline · Gold layer (analytics-ready facts that back the UC metric views,
-- the AI/BI dashboard, Genie and the agents' analytics tools).

CREATE OR REFRESH MATERIALIZED VIEW gold_visit_facts
CLUSTER BY (visit_date)
COMMENT 'One row per visit, denormalised with service economics and patient cohort attributes.'
AS SELECT
  v.visit_id, v.visit_date, v.patient_id, v.service_id, s.name AS service_name, s.category,
  v.payer, v.provider, v.status, v.price_paid,
  CASE WHEN v.status = 'completed' THEN v.price_paid - s.unit_cost ELSE 0 END AS margin,
  p.acquisition_channel, p.is_member, p.insurance_type,
  DATE_TRUNC('MONTH', p.first_visit_date) AS cohort_month,
  CAST(FLOOR(MONTHS_BETWEEN(v.visit_date, p.first_visit_date)) AS INT) AS months_since_first_visit
FROM visits v
JOIN services s USING (service_id)
JOIN patients p USING (patient_id);

CREATE OR REFRESH MATERIALIZED VIEW gold_lead_facts
COMMENT 'One row per lead with funnel attributes and AI-classified intent.'
AS SELECT
  lead_id, CAST(created_at AS DATE) AS created_date, source, complaint, insurance_type, status, converted,
  first_response_hours, ai_intent,
  CASE WHEN distance_miles < 5 THEN '0-5 mi' WHEN distance_miles < 10 THEN '5-10 mi'
       WHEN distance_miles < 20 THEN '10-20 mi' ELSE '20+ mi' END AS distance_band,
  CASE WHEN first_response_hours IS NULL THEN 'not yet'
       WHEN first_response_hours < 1 THEN '< 1h' WHEN first_response_hours < 4 THEN '1-4h'
       WHEN first_response_hours < 24 THEN '4-24h' ELSE '24h+' END AS response_band
FROM leads;

CREATE OR REFRESH MATERIALIZED VIEW gold_weekly_service_demand
COMMENT 'Weekly completed volume and average realised price per service and payer group (forecasting input).'
AS SELECT
  DATE_TRUNC('WEEK', visit_date) AS week, service_id,
  CASE WHEN payer = 'cash' THEN 'cash' ELSE 'insured' END AS payer_group,
  COUNT(*) AS qty, SUM(price_paid) AS revenue, AVG(price_paid) AS avg_price
FROM visits
WHERE status = 'completed' AND visit_date < DATE_TRUNC('WEEK', current_date())
GROUP BY ALL;

CREATE OR REFRESH MATERIALIZED VIEW gold_cohort_retention
COMMENT 'Share of each monthly first-visit cohort still visiting N months later.'
AS WITH cohort AS (
  SELECT DATE_TRUNC('MONTH', first_visit_date) AS cohort_month, COUNT(*) AS cohort_size
  FROM patients GROUP BY ALL
), activity AS (
  SELECT cohort_month, months_since_first_visit, COUNT(DISTINCT patient_id) AS active_patients
  FROM gold_visit_facts WHERE status = 'completed' GROUP BY ALL
)
SELECT a.cohort_month, a.months_since_first_visit, c.cohort_size, a.active_patients,
       ROUND(a.active_patients / c.cohort_size, 4) AS retention_rate
FROM activity a JOIN cohort c USING (cohort_month);
