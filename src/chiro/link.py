"""Link the real clinic dataset into the shape the rest of the repo expects.

`workspace.chiro_hackathon` is a real, well-formed clinic dataset (see the README). Its schema
differs from the internal contract the agents/pipeline/models were built against, so this module
projects it into the landing and operational tables the repo reads:

  source.leads        -> leads_raw      (adds a synthetic message + consent/insurance/distance
                                         so the AI Functions and guardrails still have inputs)
  source.patients     -> patients_raw   (age band -> age, payer/member derived from visit mix)
  source.appointments -> visits_raw     (appointment spine, status mapped to completed/no_show/cancelled,
     + source.visits                     revenue -> price_paid, payment_type -> payer)
  source.providers    -> appointment_slots (a forward book, since the source is historical)
  derived             -> services, price_history

Every assumption (unit cost, competitor band, consent, distance, the lead message) is deterministic
and clearly a stand-in for a field the source does not carry. Nothing here writes to the source.
"""
from __future__ import annotations

from chiro.config import Settings

# Stable vocabularies reused from the synthetic generator so downstream code keeps working.
_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn", "Drew",
          "Sam", "Robin", "Maria", "Wei", "Priya", "Diego", "Aisha", "Noah", "Emma", "Liam", "Olivia",
          "Mateo", "Sofia", "Kenji"]
_COMPLAINTS = ["low_back_pain", "neck_pain", "headaches", "sciatica", "sports_injury",
               "posture_wellness", "auto_injury"]
_COMPLAINT_BODIES = ["my lower back has been sore every morning", "a stiff neck I can't turn to the left",
                     "tension headaches most afternoons", "pain shooting down my right leg",
                     "a tweaked shoulder from pickleball", "general posture and wellness",
                     "neck and back pain since a fender bender"]
_RED_FLAGS = ["lower back pain and now numbness in my groin area",
              "the worst headache of my life that came on suddenly",
              "back pain with a high fever and chills", "chest pain that spreads into my left arm",
              "back pain and my legs keep giving out"]


def _array(values: list[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _element(values: list[str], key: str, salt: str = "") -> str:
    k = f"concat({key}, '{salt}')" if salt else key
    return f"element_at(array({_array(values)}), pmod(hash({k}), {len(values)}) + 1)"


def _service_id(col: str) -> str:
    return f"""CASE {col}
        WHEN 'Initial Consultation' THEN 'NEW_EXAM'
        WHEN 'Spinal Adjustment' THEN 'ADJ'
        WHEN 'Follow-Up Adjustment' THEN 'ADJ_ST'
        WHEN 'Re-Evaluation' THEN 'RE_EVAL'
        WHEN 'Physical Therapy' THEN 'PT'
        WHEN 'Therapeutic Massage' THEN 'MASSAGE60'
        ELSE replace(lower({col}), ' ', '_') END"""


def link_statements(s: Settings) -> list[tuple[str, str]]:
    """Return (target_table, sql) pairs that (re)build the repo's landing/operational tables."""
    S, T = s.source_table, s.table

    leads = f"""
    CREATE OR REPLACE TABLE {T('leads_raw')} AS
    WITH mapped AS (
      SELECT lead_id, created_date, status, first_response_hours, num_touchpoints, assigned_location_id,
             replace(replace(lower(source), ' ', '_'), '-', '_') AS source,
             CASE source WHEN 'Referral' THEN 'low_back_pain' WHEN 'Phone Inquiry' THEN 'sciatica'
                         WHEN 'Walk-In' THEN 'neck_pain' WHEN 'Website Form' THEN 'posture_wellness'
                         WHEN 'Social Media Ad' THEN 'sports_injury' WHEN 'Community Event' THEN 'headaches'
                         ELSE 'auto_injury' END AS complaint,
             CASE WHEN pmod(hash(lead_id), 100) < 40 THEN 'cash'
                  WHEN pmod(hash(lead_id), 100) < 78 THEN 'ppo'
                  WHEN pmod(hash(lead_id), 100) < 90 THEN 'medicare'
                  ELSE 'unknown' END AS insurance_type,
             round(0.5 + 24.5 * pmod(hash(concat(lead_id, 'd')), 100) / 100.0, 1) AS distance_miles
      FROM {S('leads')}
    )
    SELECT
      lead_id,
      to_timestamp(created_date) AS created_at,
      {_element(_NAMES, 'lead_id')} AS first_name,
      source, complaint, insurance_type, distance_miles,
      concat(
        CASE WHEN pmod(hash(lead_id), 100) = 0
             THEN concat('Hi, I have ', {_element(_RED_FLAGS, 'lead_id', 'r')}, '.')
             ELSE concat('Hi, I have ',
                element_at(array({_array(_COMPLAINT_BODIES)}),
                           cast(array_position(array({_array(_COMPLAINTS)}), complaint) AS INT)), '.') END,
        ' ',
        CASE WHEN pmod(hash(concat(lead_id, 'u')), 100) < 40 THEN 'Hoping to get in ASAP.'
             WHEN pmod(hash(concat(lead_id, 'u')), 100) < 70 THEN 'Can barely sit at work today.'
             ELSE 'Just looking for info.' END,
        ' ',
        CASE insurance_type WHEN 'cash' THEN 'What is your self-pay rate?'
             WHEN 'ppo' THEN 'Do you take PPO insurance?'
             WHEN 'medicare' THEN 'Do you accept Medicare?' ELSE '' END
      ) AS message,
      (pmod(hash(concat(lead_id, 'c')), 100) < 93) AS consent_to_contact,
      (pmod(hash(concat(lead_id, 'c')), 100) < 93 AND pmod(hash(concat(lead_id, 's')), 100) < 80) AS consent_sms,
      (pmod(hash(concat(lead_id, 'c')), 100) < 93 AND pmod(hash(concat(lead_id, 'e')), 100) < 90) AS consent_email,
      first_response_hours,
      CASE status WHEN 'Converted' THEN 'booked' WHEN 'Lost' THEN 'lost' ELSE 'new' END AS status,
      CASE status WHEN 'Converted' THEN true WHEN 'Lost' THEN false ELSE CAST(NULL AS BOOLEAN) END AS converted,
      num_touchpoints,
      assigned_location_id
    FROM mapped"""

    patients = f"""
    CREATE OR REPLACE TABLE {T('patients_raw')} AS
    WITH mix AS (
      SELECT patient_id,
             sum(CASE WHEN payment_type = 'Insurance' THEN 1 ELSE 0 END) AS ins,
             sum(CASE WHEN payment_type = 'Package Plan' THEN 1 ELSE 0 END) AS pkg,
             count(*) AS n
      FROM {S('visits')} GROUP BY patient_id
    )
    SELECT
      p.patient_id,
      {_element(_NAMES, 'p.patient_id')} AS first_name,
      CASE p.age_band WHEN '18-24' THEN 21 WHEN '25-34' THEN 30 WHEN '35-44' THEN 40
                      WHEN '45-54' THEN 50 WHEN '55-64' THEN 60 WHEN '65+' THEN 72 ELSE 45 END AS age,
      CASE WHEN coalesce(m.n, 0) = 0 THEN 'cash'
           WHEN m.ins * 1.0 / m.n > 0.5 THEN (CASE WHEN p.age_band = '65+' THEN 'medicare' ELSE 'ppo' END)
           ELSE 'cash' END AS insurance_type,
      coalesce(m.pkg, 0) > 0 AS is_member,
      round(0.5 + 24.5 * pmod(hash(concat(p.patient_id, 'd')), 100) / 100.0, 1) AS distance_miles,
      replace(replace(lower(p.acquisition_source), ' ', '_'), '-', '_') AS acquisition_channel,
      {_element(_COMPLAINTS, 'p.patient_id', 'c')} AS primary_complaint,
      element_at(array(6, 8, 12), pmod(hash(concat(p.patient_id, 'p')), 3) + 1) AS care_plan_visits,
      p.first_visit_date,
      (pmod(hash(concat(p.patient_id, 's')), 100) < 80) AS consent_sms,
      (pmod(hash(concat(p.patient_id, 'e')), 100) < 90) AS consent_email,
      CASE WHEN pmod(hash(concat(p.patient_id, 'ch')), 100) < 70 THEN 'sms' ELSE 'email' END AS preferred_channel
    FROM {S('patients')} p LEFT JOIN mix m USING (patient_id)"""

    visits = f"""
    CREATE OR REPLACE TABLE {T('visits_raw')} AS
    WITH mix AS (
      SELECT patient_id,
             sum(CASE WHEN payment_type = 'Insurance' THEN 1 ELSE 0 END) AS ins, count(*) AS n
      FROM {S('visits')} GROUP BY patient_id
    )
    SELECT
      a.appointment_id AS visit_id,
      a.patient_id,
      a.appointment_date AS visit_date,
      {_service_id('a.appointment_type')} AS service_id,
      a.provider_id AS provider,
      CASE WHEN v.payment_type = 'Insurance'
                THEN (CASE WHEN pat.age_band = '65+' THEN 'medicare' ELSE 'ppo' END)
           WHEN v.payment_type IS NOT NULL THEN 'cash'
           WHEN coalesce(m.ins, 0) * 1.0 / nullif(m.n, 0) > 0.5
                THEN (CASE WHEN pat.age_band = '65+' THEN 'medicare' ELSE 'ppo' END)
           ELSE 'cash' END AS payer,
      CASE a.status WHEN 'Completed' THEN 'completed' WHEN 'No-Show' THEN 'no_show'
                    ELSE 'cancelled' END AS status,
      CASE WHEN a.status = 'Completed' THEN coalesce(v.revenue, 0.0) ELSE 0.0 END AS price_paid
    FROM {S('appointments')} a
    LEFT JOIN {S('visits')} v ON v.appointment_id = a.appointment_id
    LEFT JOIN {S('patients')} pat ON pat.patient_id = a.patient_id
    LEFT JOIN mix m ON m.patient_id = a.patient_id"""

    services = f"""
    CREATE OR REPLACE TABLE {T('services')} AS
    WITH agg AS (
      SELECT service_id,
             avg(CASE WHEN status = 'completed' THEN price_paid END) AS avg_price,
             avg(CASE WHEN status = 'completed' AND payer <> 'cash' THEN price_paid END) AS allowed
      FROM {T('visits_raw')} GROUP BY service_id
    )
    SELECT
      service_id,
      CASE service_id WHEN 'NEW_EXAM' THEN 'New patient exam + X-ray'
           WHEN 'ADJ' THEN 'Chiropractic adjustment'
           WHEN 'ADJ_ST' THEN 'Adjustment + soft tissue'
           WHEN 'RE_EVAL' THEN 'Re-evaluation'
           WHEN 'PT' THEN 'Physical therapy session'
           WHEN 'MASSAGE60' THEN '60-min therapeutic massage' ELSE service_id END AS name,
      CASE WHEN service_id = 'MASSAGE60' THEN 'addon' ELSE 'core' END AS category,
      round(avg_price, 2) AS ref_price,
      round(avg_price * 0.40, 2) AS unit_cost,
      CASE service_id WHEN 'NEW_EXAM' THEN 45 WHEN 'ADJ' THEN 15 WHEN 'ADJ_ST' THEN 30
           WHEN 'RE_EVAL' THEN 30 WHEN 'PT' THEN 45 WHEN 'MASSAGE60' THEN 60 ELSE 30 END AS duration_min,
      round(avg_price * 0.85, 2) AS competitor_low,
      round(avg_price * 1.15, 2) AS competitor_high,
      round(avg_price * 0.75, 2) AS min_price,
      round(avg_price * 1.25, 2) AS max_price,
      round(coalesce(allowed, avg_price * 0.85), 2) AS insurance_allowed,
      true AS is_cash_pay,
      round(avg_price, 2) AS current_price
    FROM agg"""

    price_history = f"""
    CREATE OR REPLACE TABLE {T('price_history')} AS
    SELECT service_id, date_sub(current_date(), 360) AS effective_date, round(current_price * 0.92, 2) AS price
      FROM {T('services')}
    UNION ALL SELECT service_id, date_sub(current_date(), 270), round(current_price * 0.97, 2) FROM {T('services')}
    UNION ALL SELECT service_id, date_sub(current_date(), 180), round(current_price * 1.03, 2) FROM {T('services')}
    UNION ALL SELECT service_id, date_sub(current_date(), 90), round(current_price * 1.08, 2) FROM {T('services')}"""

    slots = f"""
    CREATE OR REPLACE TABLE {T('appointment_slots')} AS
    SELECT
      concat('S', date_format(d, 'yyyyMMdd'), provider_id, lpad(cast(slot AS STRING), 2, '0')) AS slot_id,
      timestamp(concat(date_format(d, 'yyyy-MM-dd'), ' ',
                       lpad(cast(8 + floor((slot - 1) / 2) AS INT), 2, '0'), ':',
                       CASE WHEN pmod(slot, 2) = 1 THEN '00' ELSE '30' END, ':00')) AS slot_start,
      provider_id AS provider,
      (pmod(hash(concat(provider_id, d, slot)), 100) < 30) AS is_open
    FROM (SELECT date_add(current_date(), d + 1) AS d FROM (SELECT explode(sequence(0, 13)) AS d)),
         (SELECT provider_id FROM {S('providers')} WHERE active_flag),
         (SELECT explode(sequence(1, 16)) AS slot)"""

    return [("leads_raw", leads), ("patients_raw", patients), ("visits_raw", visits),
            ("services", services), ("price_history", price_history), ("appointment_slots", slots)]
