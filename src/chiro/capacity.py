"""Capacity: find unused capacity, forecast it, and turn it into booked visits.

Definitions
  room capacity     - capacity_patients_per_day of the location (what the space can take)
  staffed slots     - provider slots actually scheduled (SLOTS_PER_PROVIDER_DAY per active provider,
                      half on Saturday, none on Sunday)
  effective capacity- min(room capacity, staffed slots): the most visits a day can hold
  utilization       - completed visits / effective capacity on open days
A location that fills its staffed slots while rooms sit empty is staffing-constrained (add hours); one
with empty staffed slots has a demand or attendance problem (fill, reduce no-shows, market, or trim hours).

Benchmarks are the network's own top quartile, an achievable target. Forecasts combine the forward
booking book with the location's booking lead-time curve: bookings still to come for a day =
typical final bookings x share of bookings usually made within the remaining days.
"""
from __future__ import annotations

import datetime as dt
import math

from chiro.config import Settings

SLOTS_PER_PROVIDER_DAY = 16
REACTIVATION_RESPONSE_RATE = 0.15  # assumption: share of contacted lapsed patients who book; replace with measured
MAX_OVERBOOK_RISK = 0.05  # overbook only while P(more arrivals than staffed slots) stays under this

ACTION_TYPES = {
    "reactivation_campaign": "Invite lapsed patients of this location into specific open days",
    "no_show_reduction": "Confirmations, reminders and a standby list to backfill late cancellations",
    "overbooking_policy": "Book a few extra slots on reliably full days, sized to the no-show rate",
    "schedule_rebalance": "Shift provider hours from low-demand weekdays to busier ones",
    "add_provider_hours": "Staffed slots are full while rooms sit empty: add provider hours",
    "local_marketing": "Targeted new-patient campaign for this location",
    "reduce_hours": "Close or shorten consistently empty sessions to cut cost",
}


def _staffed_slots_expr(providers: str, day: str) -> str:
    return (f"CASE dayofweek({day}) WHEN 1 THEN 0 WHEN 7 THEN {providers} * {SLOTS_PER_PROVIDER_DAY // 2} "
            f"ELSE {providers} * {SLOTS_PER_PROVIDER_DAY} END")


def utilization_sql(s: Settings) -> str:
    """Per location over the trailing window."""
    staffed = _staffed_slots_expr("st.active_providers", "d.appointment_date")
    return f"""
    WITH daily AS (
      SELECT location_id, appointment_date,
             count(*) AS booked,
             count_if(status = 'Completed') AS completed,
             count_if(status = 'No-Show') AS no_shows,
             count_if(status = 'Cancelled') AS cancelled
      FROM {s.table('appointments')}
      WHERE appointment_date > date_sub(current_date(), :days) AND appointment_date <= current_date()
      GROUP BY ALL
    ),
    rev AS (
      SELECT location_id, avg(revenue) AS revenue_per_visit
      FROM {s.source_table('visits')} WHERE visit_date > date_sub(current_date(), 365) GROUP BY ALL
    ),
    staff AS (
      SELECT location_id, count_if(active_flag) AS active_providers FROM {s.source_table('providers')} GROUP BY ALL
    ),
    lapsed AS (
      SELECT home_location_id AS location_id, count(*) AS lapsed_patients
      FROM {s.source_table('patients')} WHERE status = 'Lapsed' GROUP BY ALL
    ),
    weekday_mix AS (  -- spread of average bookings across Mon-Fri, relative to their mean
      SELECT location_id, (max(avg_booked) - min(avg_booked)) / nullif(avg(avg_booked), 0) AS weekday_spread
      FROM (SELECT location_id, dayofweek(appointment_date) AS dow, avg(booked) AS avg_booked
            FROM daily WHERE dayofweek(appointment_date) BETWEEN 2 AND 6 GROUP BY ALL)
      GROUP BY ALL
    ),
    per_day AS (
      SELECT d.*, l.capacity_patients_per_day AS room_capacity, {staffed} AS staffed_slots,
             least(l.capacity_patients_per_day, {staffed}) AS effective_capacity
      FROM daily d JOIN {s.source_table('locations')} l USING (location_id)
      JOIN staff st USING (location_id)
    )
    SELECT l.location_id, l.location_name, l.city, l.capacity_patients_per_day AS room_capacity,
           coalesce(st.active_providers, 0) AS active_providers,
           count(p.appointment_date) AS open_days,
           round(count(p.appointment_date) / (:days / 7.0), 2) AS open_days_per_week,
           round(avg(p.staffed_slots), 1) AS staffed_slots_per_day,
           round(avg(p.effective_capacity), 1) AS effective_capacity_per_day,
           round(avg(p.booked), 1) AS booked_per_day,
           round(avg(p.completed), 1) AS completed_per_day,
           round(sum(p.completed) / nullif(sum(p.effective_capacity), 0), 3) AS utilization,
           round(sum(p.booked) / nullif(sum(p.staffed_slots), 0), 3) AS staffed_slot_fill,
           round(sum(p.completed) / nullif(sum(p.room_capacity), 0), 3) AS room_utilization,
           round(avg(CASE WHEN p.booked >= 0.95 * p.staffed_slots THEN 1.0 ELSE 0.0 END), 3) AS share_days_fully_booked,
           round(sum(p.no_shows) / nullif(sum(p.booked), 0), 3) AS no_show_rate,
           round(sum(p.no_shows + p.cancelled) / nullif(sum(p.booked), 0), 3) AS lost_slot_rate,
           round(avg(p.no_shows + p.cancelled), 1) AS lost_visits_per_day,
           round(r.revenue_per_visit, 2) AS revenue_per_visit,
           round(l.monthly_lease_cost / nullif(avg(p.completed) * 26, 0), 2) AS lease_cost_per_visit,
           coalesce(lp.lapsed_patients, 0) AS lapsed_patients,
           round(wm.weekday_spread, 2) AS weekday_spread,
           -- Weekly trend in completed visits over the window (visits per week, per week).
           round(regr_slope(p.completed, datediff(p.appointment_date, current_date())) * 7
                 * count(p.appointment_date) / (:days / 7.0), 2) AS weekly_trend
    FROM {s.source_table('locations')} l
    LEFT JOIN per_day p USING (location_id)
    LEFT JOIN rev r USING (location_id)
    LEFT JOIN staff st USING (location_id)
    LEFT JOIN lapsed lp USING (location_id)
    LEFT JOIN weekday_mix wm USING (location_id)
    GROUP BY l.location_id, l.location_name, l.city, l.capacity_patients_per_day, l.monthly_lease_cost,
             st.active_providers, r.revenue_per_visit, lp.lapsed_patients, wm.weekday_spread"""


def diagnose(row: dict) -> list[str]:
    """Plain-language problems a location's numbers point to, most important first."""
    out = []
    if (row.get("share_days_fully_booked") or 0) >= 0.5 and (row.get("room_utilization") or 0) < 0.8:
        out.append("staffing-constrained: staffed slots are full on most days while rooms have space")
    if (row.get("no_show_rate") or 0) >= 0.15:
        out.append(f"high no-shows ({row['no_show_rate']:.0%} of bookings)")
    if (row.get("weekly_trend") or 0) <= -0.03 * (row.get("completed_per_day") or 0) * (row.get("open_days_per_week") or 0):
        out.append(f"declining: {row['weekly_trend']:+.1f} completed visits/week each week")
    if (row.get("weekday_spread") or 0) >= 0.6:
        out.append(f"weekday imbalance: busiest weekday books {row['weekday_spread']:.0%} more than the quietest "
                   "(relative to the average)")
    if (row.get("utilization") or 1) < 0.6 and (row.get("share_days_fully_booked") or 0) < 0.2:
        out.append(f"low demand: {row['utilization']:.0%} of effective capacity used")
    return out


def add_revenue_gap(rows: list[dict], benchmark_quantile: float = 0.75) -> dict:
    """Benchmark every location against the network's own top quartile (achievable, unlike nominal capacity
    many clinics never reach), size the weekly revenue left on the table, and diagnose each location."""
    utils = sorted(float(r["utilization"]) for r in rows if r.get("utilization") is not None)
    if not utils:
        return {"benchmark_utilization": None, "locations": rows}
    bench = utils[min(len(utils) - 1, int(benchmark_quantile * len(utils)))]
    for r in rows:
        r["problems"] = diagnose(r)
        if r.get("utilization") is None:
            r["visits_gap_per_day"], r["weekly_revenue_gap"] = None, None
            continue
        gap = max(bench * float(r["effective_capacity_per_day"]) - float(r["completed_per_day"]), 0.0)
        r["visits_gap_per_day"] = round(gap, 1)
        r["weekly_revenue_gap"] = round(gap * float(r["open_days_per_week"]) * float(r["revenue_per_visit"] or 0), 0)
    rows.sort(key=lambda r: -(r["weekly_revenue_gap"] or 0))
    return {"benchmark_utilization": bench, "locations": rows}


def weekday_sql(s: Settings) -> str:
    return f"""
    SELECT date_format(appointment_date, 'EEEE') AS weekday, dayofweek(appointment_date) AS dow,
           count(DISTINCT appointment_date) AS days,
           round(count(*) / count(DISTINCT appointment_date), 1) AS booked_per_day,
           round(count_if(status = 'Completed') / count(DISTINCT appointment_date), 1) AS completed_per_day,
           round(count_if(status = 'No-Show') / count(*), 3) AS no_show_rate,
           round(count_if(status IN ('No-Show', 'Cancelled')) / count(*), 3) AS lost_slot_rate
    FROM {s.table('appointments')}
    WHERE location_id = :loc AND appointment_date > date_sub(current_date(), :days)
      AND appointment_date <= current_date()
    GROUP BY ALL ORDER BY dow"""


def provider_sql(s: Settings) -> str:
    return f"""
    SELECT p.provider_id, p.specialty, p.employment_type,
           count(DISTINCT a.appointment_date) AS working_days,
           round(count_if(a.status = 'Completed') / nullif(count(DISTINCT a.appointment_date), 0), 1)
             AS completed_per_working_day,
           round(count_if(a.status IN ('No-Show', 'Cancelled')) / nullif(count(a.appointment_id), 0), 3)
             AS lost_slot_rate
    FROM {s.source_table('providers')} p
    LEFT JOIN {s.table('appointments')} a
      ON a.provider_id = p.provider_id AND a.appointment_date > date_sub(current_date(), :days)
         AND a.appointment_date <= current_date()
    WHERE p.location_id = :loc AND p.active_flag
    GROUP BY ALL ORDER BY completed_per_working_day"""


def lead_time_sql(s: Settings) -> str:
    """Share of a day's final bookings made at least k days ahead, k = 0..30 (location, trailing window)."""
    return f"""
    SELECT k, round(avg(CASE WHEN lead_time_days >= k THEN 1.0 ELSE 0.0 END), 4) AS share_booked_k_days_ahead
    FROM {s.table('appointments')} LATERAL VIEW explode(sequence(0, 30)) t AS k
    WHERE location_id = :loc AND appointment_date > date_sub(current_date(), :days)
      AND appointment_date <= current_date()
    GROUP BY k ORDER BY k"""


def book_sql(s: Settings) -> str:
    """Forward book per day: staffed slots and how many are already taken."""
    return f"""
    SELECT to_date(sl.slot_start) AS day, count(*) AS staffed_slots, count_if(NOT sl.is_open) AS booked_now
    FROM {s.table('appointment_slots')} sl JOIN {s.source_table('providers')} p ON p.provider_id = sl.provider
    WHERE p.location_id = :loc AND sl.slot_start > current_timestamp()
      AND sl.slot_start < timestampadd(DAY, :days, current_timestamp())
    GROUP BY ALL ORDER BY day"""


def forecast(book: list[dict], by_weekday: list[dict], lead_curve: list[dict], room_capacity: int,
             today: dt.date) -> list[dict]:
    """Projected final bookings, attended visits and idle slots for each forward day.

    expected_final = booked_now + typical_final x (1 - share usually booked >= days_ahead days out);
    `typical_final` is the weekday's trailing mean; attended = expected_final x weekday show rate."""
    curve = {int(r["k"]): float(r["share_booked_k_days_ahead"]) for r in lead_curve}
    wd = {int(r["dow"]): r for r in by_weekday}
    out = []
    for b in book:
        day = b["day"] if isinstance(b["day"], dt.date) else dt.date.fromisoformat(str(b["day"])[:10])
        dow = (day.isoweekday() % 7) + 1  # Spark dayofweek: Sunday=1
        ahead = (day - today).days
        staffed = int(b["staffed_slots"])
        capacity = min(staffed, int(room_capacity))
        hist = wd.get(dow, {})
        typical = float(hist.get("booked_per_day") or 0.0)
        still_to_come = typical * (1.0 - curve.get(min(max(ahead, 0), 30), 0.0))
        expected_final = min(float(b["booked_now"]) + still_to_come, capacity)
        show = 1.0 - float(hist.get("lost_slot_rate") or 0.0)
        out.append({"day": day.isoformat(), "weekday": day.strftime("%A"), "days_ahead": ahead,
                    "effective_capacity": capacity, "booked_now": int(b["booked_now"]),
                    "expected_final_bookings": round(expected_final, 1),
                    "expected_attended": round(expected_final * show, 1),
                    "projected_idle_slots": round(capacity - expected_final, 1),
                    "projected_utilization": round(expected_final * show / capacity, 3) if capacity else None})
    return out


def _binom_tail(n: int, p: float, k: int) -> float:
    """P(X > k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1, n + 1))


def no_show_plan(by_weekday: list[dict], staffed_by_dow: dict[int, int]) -> list[dict]:
    """Per weekday: on reliably full days, how many extra bookings keep overflow risk under MAX_OVERBOOK_RISK;
    on days with spare slots, how big a same-day standby list should be to backfill expected losses."""
    plan = []
    for r in by_weekday:
        dow = int(r["dow"])
        staffed = staffed_by_dow.get(dow, 0)
        if not staffed:
            continue
        booked, lost = float(r["booked_per_day"]), float(r["lost_slot_rate"] or 0)
        expected_lost = booked * lost
        if booked >= 0.95 * staffed:
            show, extra = 1.0 - lost, 0
            while extra < staffed and _binom_tail(staffed + extra + 1, show, staffed) <= MAX_OVERBOOK_RISK:
                extra += 1
            plan.append({"weekday": r["weekday"], "full": True, "expected_lost_slots": round(expected_lost, 1),
                         "recommended_overbook": extra,
                         "overflow_risk": round(_binom_tail(staffed + extra, show, staffed), 3)})
        else:
            plan.append({"weekday": r["weekday"], "full": False, "expected_lost_slots": round(expected_lost, 1),
                         "recommended_standby_list": math.ceil(expected_lost * 2)})  # ~half of standbys can come
    return plan


def fill_candidates_sql(s: Settings) -> str:
    """Lapsed-but-recoverable patients of a location, ranked for a target day.

    Eligible: last completed visit 45-365 days ago, consented to SMS or email, not contacted in the last
    30 days, not already in an approved campaign. Score favours 60-180 days since the last visit (still
    recoverable), more past visits, and a usual weekday that matches the target day."""
    return f"""
    WITH hist AS (
      SELECT patient_id, max(appointment_date) AS last_visit, count(*) AS past_visits,
             mode(dayofweek(appointment_date)) AS usual_dow
      FROM {s.table('appointments')} WHERE status = 'Completed' AND location_id = :loc GROUP BY ALL
    )
    SELECT h.patient_id, p.first_name, h.last_visit, datediff(current_date(), h.last_visit) AS days_since_last_visit,
           h.past_visits, date_format(date_add(DATE'2023-01-01', h.usual_dow - 1), 'EEEE') AS usual_weekday,  -- a Sunday
           (h.usual_dow = dayofweek(CAST(:day AS DATE))) AS weekday_match,
           round((CASE WHEN datediff(current_date(), h.last_visit) BETWEEN 60 AND 180 THEN 1.0 ELSE 0.6 END)
                 * ln(1 + h.past_visits) * (CASE WHEN h.usual_dow = dayofweek(CAST(:day AS DATE)) THEN 1.5 ELSE 1.0 END),
                 3) AS fit_score
    FROM hist h JOIN {s.table('patients')} p USING (patient_id)
    WHERE datediff(current_date(), h.last_visit) BETWEEN 45 AND 365
      AND (p.consent_sms OR p.consent_email)
      AND NOT EXISTS (SELECT 1 FROM {s.table('retention_actions')} r
                      WHERE r.patient_id = h.patient_id AND r.created_at > current_timestamp() - INTERVAL 30 DAYS)
      AND NOT EXISTS (SELECT 1 FROM {s.table('campaign_targets')} c
                      WHERE c.patient_id = h.patient_id AND c.status IN ('to_contact', 'drafted'))
    ORDER BY fit_score DESC, h.last_visit DESC
    LIMIT {{limit}}"""


def impact_sql(s: Settings) -> str:
    """Utilization after each approved action versus the baseline captured at approval."""
    staffed = _staffed_slots_expr("st.active_providers", "a.appointment_date")
    return f"""
    WITH staff AS (
      SELECT location_id, count_if(active_flag) AS active_providers FROM {s.source_table('providers')} GROUP BY ALL
    ),
    post AS (
      SELECT c.action_id, count(DISTINCT a.appointment_date) AS days_measured,
             sum(CASE WHEN a.status = 'Completed' THEN 1 ELSE 0 END) AS completed
      FROM {s.table('capacity_actions')} c
      JOIN {s.table('appointments')} a
        ON a.location_id = c.location_id AND a.appointment_date > to_date(c.reviewed_at)
           AND a.appointment_date <= current_date()
      WHERE c.status = 'approved' GROUP BY ALL
    ),
    cap AS (
      SELECT c.action_id, sum(least(l.capacity_patients_per_day, {staffed})) AS capacity
      FROM {s.table('capacity_actions')} c
      JOIN (SELECT DISTINCT location_id, appointment_date FROM {s.table('appointments')}) a
        ON a.location_id = c.location_id AND a.appointment_date > to_date(c.reviewed_at)
           AND a.appointment_date <= current_date()
      JOIN {s.source_table('locations')} l ON l.location_id = c.location_id
      JOIN staff st ON st.location_id = c.location_id
      WHERE c.status = 'approved' GROUP BY ALL
    )
    SELECT c.action_id, c.location_id, c.action_type, to_date(c.reviewed_at) AS approved_on,
           c.baseline_utilization, coalesce(post.days_measured, 0) AS days_measured,
           round(post.completed / nullif(cap.capacity, 0), 3) AS utilization_since,
           round(post.completed / nullif(cap.capacity, 0) - c.baseline_utilization, 3) AS change,
           c.expected_weekly_visits
    FROM {s.table('capacity_actions')} c
    LEFT JOIN post USING (action_id) LEFT JOIN cap USING (action_id)
    WHERE c.status = 'approved' ORDER BY c.reviewed_at DESC"""
