"""Agent tools over Unity Catalog tables.

Read tools give the LLM grounded facts; write tools only ever create *pending_review*
rows. Guardrails (consent, red flags, advertising claims, offer eligibility, price bands)
are enforced here in code, not just in prompts. `review()` is deliberately NOT exposed
to any agent - approvals are human-only, through the Databricks App.
"""
from __future__ import annotations

import uuid

import datetime as dt

from chiro import capacity, guardrails, pricing, priority, semantic, themes
from chiro.agent import Tool, ToolRegistry, params
from chiro.config import Settings
from chiro.sql import SqlRunner
from chiro.urgency import RESPOND_WITHIN_HOURS

LEAD_ACTIONS = ("outreach", "refer_out", "disqualify")
CHANNELS = ("sms", "email", "phone")


def _err(msg: str) -> dict:
    return {"error": msg}


class ClinicTools:
    def __init__(self, db: SqlRunner, settings: Settings, run_id: str):
        self.db, self.s, self.run_id = db, settings, run_id

    def t(self, name: str) -> str:
        return self.s.table(name)

    # ------------------------------------------------------------------ shared
    def get_service_prices(self) -> list[dict]:
        return self.db.query(
            f"SELECT service_id, name, category, current_price FROM {self.t('services')} ORDER BY current_price")

    def get_open_slots(self, days_ahead: int = 5, limit: int = 8) -> list[dict]:
        return self.db.query(
            f"""SELECT slot_id, date_format(slot_start, 'EEE MMM d, h:mm a') AS starts, provider
                FROM {self.t('appointment_slots')}
                WHERE is_open AND slot_start > current_timestamp()
                  AND slot_start < timestampadd(DAY, :days, current_timestamp())
                ORDER BY slot_start LIMIT {int(min(limit, 20))}""",
            {"days": int(days_ahead)})

    def get_kpis(self) -> dict:
        t = self.t
        rows = self.db.query(f"""
            SELECT
              (SELECT count(*) FROM {t('leads')} WHERE created_at >= current_timestamp() - INTERVAL 30 DAYS) AS leads_30d,
              (SELECT round(avg(CASE WHEN converted THEN 1.0 ELSE 0.0 END), 3) FROM {t('leads')}
                 WHERE converted IS NOT NULL AND created_at >= current_timestamp() - INTERVAL 180 DAYS) AS lead_conversion_rate_180d,
              -- anti join, not NOT EXISTS: a correlated NOT EXISTS next to another scalar subquery on leads
              -- crashes the Databricks SQL optimizer with INTERNAL_ERROR
              (SELECT count(*) FROM {t('leads_raw')} l LEFT ANTI JOIN {t('lead_actions')} a
                 ON a.lead_id = l.lead_id AND a.status <> 'rejected'
                 WHERE l.status = 'new' AND l.first_response_hours IS NULL
                   AND l.created_at > current_timestamp() - INTERVAL {priority.ACTIVE_LEAD_DAYS} DAYS)
                 AS open_leads,
              (SELECT count(*) FROM {t('leads_raw')} WHERE status = 'new'
                 AND created_at <= current_timestamp() - INTERVAL {priority.ACTIVE_LEAD_DAYS} DAYS) AS stale_open_leads,
              (SELECT count(DISTINCT patient_id) FROM {t('visits')}
                 WHERE status = 'completed' AND visit_date >= current_date() - INTERVAL 45 DAYS) AS active_patients,
              (SELECT count(*) FROM {t('patient_churn_risk')} WHERE churn_risk >= 0.5) AS high_risk_patients,
              (SELECT round(sum(churn_risk * ltv_annual), 0) FROM {t('patient_churn_risk')}) AS annual_revenue_at_risk,
              (SELECT round(sum(price_paid), 0) FROM {t('visits')}
                 WHERE status = 'completed' AND visit_date >= current_date() - INTERVAL 30 DAYS) AS revenue_30d,
              (SELECT round(avg(CASE WHEN status = 'no_show' THEN 1.0 ELSE 0.0 END), 3) FROM {t('visits')}
                 WHERE visit_date >= current_date() - INTERVAL 30 DAYS) AS no_show_rate_30d,
              (SELECT count(*) FROM {t('lead_actions')} WHERE status = 'pending_review') AS pending_lead_actions,
              (SELECT count(*) FROM {t('retention_actions')} WHERE status = 'pending_review') AS pending_retention_actions,
              (SELECT count(*) FROM {t('price_recommendations')} WHERE status = 'pending_review') AS pending_price_changes
        """)
        return rows[0] if rows else {}

    # ------------------------------------------------------------------ leads
    def _priority_context(self) -> dict:
        """Lead-model base rate and AUC (for shrinkage) and typical 12-month revenue per new patient by payer."""
        if getattr(self, "_prio", None) is None:
            m = {r["metric"]: r["value"] for r in self.db.query(
                f"SELECT metric, max_by(value, trained_at) AS value FROM {self.t('model_metrics')} "
                "WHERE metric IN ('lead_auc', 'lead_base_rate') GROUP BY metric")}
            values = {r["insurance_type"]: float(r["value"]) for r in self.db.query(
                f"""SELECT p.insurance_type, avg(v.rev) AS value
                    FROM {self.t('patients')} p
                    JOIN (SELECT patient_id, sum(price_paid) AS rev FROM {self.t('visits')}
                          WHERE status = 'completed' GROUP BY patient_id) v USING (patient_id)
                    GROUP BY p.insurance_type""") if r["value"] is not None}
            self._prio = {"auc": m.get("lead_auc"), "base_rate": float(m.get("lead_base_rate") or 0.25),
                          "payer_value": values}
        return self._prio

    def list_scored_leads(self, limit: int = 10, min_score: float = 0.0) -> list[dict]:
        """Open leads in the active window, ranked by tier, deadline pressure and expected value."""
        rows = self.db.query(
            f"""SELECT r.lead_id, r.created_at, r.source, r.complaint, r.insurance_type, r.distance_miles,
                       l.ai_intent, s.score, s.reasons, s.red_flags, s.urgency, s.urgency_reasons,
                       s.respond_within_hours,
                       timestampadd(MINUTE, CAST(s.respond_within_hours * 60 AS INT), r.created_at) AS respond_by
                FROM {self.t('leads_raw')} r JOIN {self.t('lead_scores')} s USING (lead_id)
                LEFT JOIN {self.t('leads')} l USING (lead_id)
                WHERE r.status = 'new' AND r.consent_to_contact AND s.score >= :min_score
                  AND r.first_response_hours IS NULL  -- already answered in the clinic's own system
                  AND r.created_at > timestampadd(DAY, -1 * :days, current_timestamp())
                  AND NOT EXISTS (SELECT 1 FROM {self.t('lead_actions')} a
                                  WHERE a.lead_id = r.lead_id AND a.status IN ('pending_review', 'approved'))""",
            {"min_score": float(min_score), "days": priority.ACTIVE_LEAD_DAYS})
        ctx = self._priority_context()
        ranked = priority.rank(rows, dt.datetime.now(dt.timezone.utc), ctx["base_rate"], ctx["auc"], ctx["payer_value"])
        return ranked[:int(min(limit, 50))]

    def get_sla_report(self, days: int = 14) -> dict:
        """Response-time performance by urgency tier for leads created in the last N days.
        A lead counts as responded when staff approved its first action (or, for leads imported from
        the clinic's system, at created_at + first_response_hours)."""
        rows = self.db.query(
            f"""WITH l AS (
                  SELECT r.lead_id, r.created_at, r.status, coalesce(s.urgency, 'routine') AS urgency,
                         timestampadd(MINUTE, CAST(coalesce(s.respond_within_hours, 24) * 60 AS INT), r.created_at)
                           AS respond_by,
                         CASE WHEN r.first_response_hours IS NOT NULL
                              THEN timestampadd(MINUTE, CAST(r.first_response_hours * 60 AS INT), r.created_at)
                         END AS imported_response
                  FROM {self.t('leads_raw')} r LEFT JOIN {self.t('lead_scores')} s USING (lead_id)
                  WHERE r.created_at > timestampadd(DAY, -1 * :days, current_timestamp())),
                a AS (
                  SELECT lead_id, min(created_at) AS first_draft,
                         min(CASE WHEN status = 'approved' THEN reviewed_at END) AS first_approval
                  FROM {self.t('lead_actions')} GROUP BY lead_id),
                j AS (
                  SELECT l.*, a.first_draft, coalesce(a.first_approval, l.imported_response) AS responded_at
                  FROM l LEFT JOIN a USING (lead_id))
                SELECT urgency, count(*) AS leads, count(responded_at) AS responded,
                       round(avg(CASE WHEN responded_at <= respond_by THEN 1.0
                                      WHEN responded_at IS NOT NULL THEN 0.0 END), 3) AS within_sla_rate,
                       round(percentile(timestampdiff(MINUTE, created_at, responded_at) / 60.0, 0.5), 2)
                         AS median_hours_to_response,
                       round(percentile(timestampdiff(MINUTE, created_at, responded_at) / 60.0, 0.9), 2)
                         AS p90_hours_to_response,
                       count_if(responded_at IS NULL AND status = 'new' AND current_timestamp() > respond_by)
                         AS overdue_now,
                       count_if(responded_at IS NULL AND status = 'new' AND first_draft IS NOT NULL
                                AND current_timestamp() > respond_by) AS overdue_waiting_on_approval
                FROM j GROUP BY urgency""", {"days": max(1, min(int(days), 90))})
        order = {t: i for i, t in enumerate(priority.TIERS)}
        rows.sort(key=lambda r: order.get(r["urgency"], 9))
        return {"window_days": days, "respond_within_hours": dict(RESPOND_WITHIN_HOURS), "by_tier": rows}

    def get_lead(self, lead_id: str) -> dict:
        # Fall back to the landing table for leads that arrived since the last pipeline run (web inquiries).
        for table in ("leads", "leads_raw"):
            rows = self.db.query(
                f"""SELECT l.*, s.score, s.reasons, s.urgency, s.urgency_reasons, s.respond_within_hours
                    FROM {self.t(table)} l
                    LEFT JOIN {self.t('lead_scores')} s USING (lead_id) WHERE l.lead_id = :id""", {"id": lead_id})
            if rows:
                break
        if not rows:
            return _err(f"lead {lead_id} not found")
        lead = rows[0]
        lead["red_flags_detected"] = guardrails.detect_red_flags(lead.get("message"))
        lead["allowed_channels"] = [c for c, ok in (("sms", lead["consent_sms"]), ("email", lead["consent_email"]),
                                                    ("phone", lead["consent_to_contact"])) if ok]
        return lead

    def queue_lead_action(self, lead_id: str, action_type: str, channel: str, message: str,
                          reasoning: str, priority: str = "normal", proposed_slot_id: str = "") -> dict:
        lead = self.get_lead(lead_id)
        if "error" in lead:
            return lead
        if action_type not in LEAD_ACTIONS:
            return _err(f"action_type must be one of {LEAD_ACTIONS}")
        if channel not in CHANNELS:
            return _err(f"channel must be one of {CHANNELS}")
        if priority not in ("high", "normal", "low"):
            return _err("priority must be high, normal or low")
        flags = lead["red_flags_detected"]
        if flags and action_type != "refer_out":
            return _err(f"Lead reports red-flag symptoms ({', '.join(flags)}). Use action_type='refer_out' "
                        "with channel='phone', advising prompt evaluation by a physician / urgent care or 911.")
        if action_type == "refer_out":
            priority, proposed_slot_id = "high", ""
            if channel != "phone":
                return _err("refer_out must use channel='phone' so staff speak to the person directly")
        elif action_type == "outreach" and channel not in lead["allowed_channels"]:
            return _err(f"no consent for {channel}; allowed channels: {lead['allowed_channels']}")
        if action_type != "disqualify":
            problems = guardrails.check_message(message, channel)
            if problems:
                return _err("message rejected: " + "; ".join(problems))
            if channel == "sms":
                message = guardrails.ensure_sms_opt_out(message)
        if proposed_slot_id:
            ok = self.db.query(f"SELECT 1 FROM {self.t('appointment_slots')} WHERE slot_id = :id AND is_open "
                               "AND slot_start > current_timestamp()", {"id": proposed_slot_id})
            if not ok:
                return _err(f"slot {proposed_slot_id} is not open; call get_open_slots")
        if self.db.query(f"SELECT 1 FROM {self.t('lead_actions')} WHERE lead_id = :id AND status = 'pending_review'",
                         {"id": lead_id}):
            return _err(f"lead {lead_id} already has a pending action")
        action_id = f"LA-{uuid.uuid4().hex[:10]}"
        self.db.execute(
            f"""INSERT INTO {self.t('lead_actions')}
                (action_id, run_id, lead_id, action_type, channel, priority, message, proposed_slot_id, reasoning,
                 status, created_at)
                VALUES (:aid, :run, :lead, :atype, :channel, :priority, :msg, NULLIF(:slot, ''), :why,
                        'pending_review', current_timestamp())""",
            {"aid": action_id, "run": self.run_id, "lead": lead_id, "atype": action_type, "channel": channel,
             "priority": priority, "msg": message, "slot": proposed_slot_id, "why": reasoning})
        return {"ok": True, "action_id": action_id, "status": "pending_review", "final_message": message}

    # ------------------------------------------------------------------ retention
    def list_at_risk_patients(self, limit: int = 10, min_risk: float = 0.5) -> list[dict]:
        return self.db.query(
            f"""SELECT r.patient_id, p.first_name, r.churn_risk, r.ltv_annual,
                       round(r.churn_risk * r.ltv_annual, 0) AS expected_annual_loss,
                       r.days_since_last_visit, r.visits_90d, r.plan_progress, r.risk_reasons,
                       p.insurance_type, p.is_member
                FROM {self.t('patient_churn_risk')} r JOIN {self.t('patients')} p USING (patient_id)
                WHERE r.churn_risk >= :min_risk
                  AND NOT EXISTS (SELECT 1 FROM {self.t('retention_actions')} a
                                  WHERE a.patient_id = r.patient_id AND a.status <> 'rejected'
                                    AND a.created_at > current_timestamp() - INTERVAL 30 DAYS)
                ORDER BY expected_annual_loss DESC
                LIMIT {int(min(limit, 50))}""",
            {"min_risk": float(min_risk)})

    def get_patient_profile(self, patient_id: str) -> dict:
        rows = self.db.query(
            f"""SELECT p.patient_id, p.first_name, p.age, p.insurance_type, p.is_member, p.primary_complaint,
                       p.care_plan_visits, p.first_visit_date, p.preferred_channel, p.consent_sms, p.consent_email,
                       r.churn_risk, r.risk_reasons, r.ltv_annual, r.days_since_last_visit, r.plan_progress
                FROM {self.t('patients')} p LEFT JOIN {self.t('patient_churn_risk')} r USING (patient_id)
                WHERE p.patient_id = :id""", {"id": patient_id})
        if not rows:
            return _err(f"patient {patient_id} not found")
        profile = rows[0]
        profile["recent_visits"] = self.db.query(
            f"""SELECT v.visit_date, s.name AS service, v.status, v.provider
                FROM {self.t('visits')} v JOIN {self.t('services')} s USING (service_id)
                WHERE v.patient_id = :id ORDER BY v.visit_date DESC LIMIT 8""", {"id": patient_id})
        profile["allowed_channels"] = [c for c, ok in (("sms", profile["consent_sms"]),
                                                       ("email", profile["consent_email"]), ("phone", True)) if ok]
        return profile

    def get_retention_offers(self, patient_id: str) -> dict:
        payer = self.db.query(f"SELECT insurance_type FROM {self.t('patients')} WHERE patient_id = :id",
                              {"id": patient_id})
        if not payer:
            return _err(f"patient {patient_id} not found")
        offers = self.db.query(f"SELECT * FROM {self.t('retention_offers')}")
        p = payer[0]["insurance_type"]
        eligible = [o for o in offers if p in o["eligible_payers"].split(",")]
        note = ("Medicare beneficiary: only non-monetary offers are allowed (beneficiary-inducement rules)."
                if p == "medicare" else "")
        return {"payer": p, "eligible_offers": eligible, "note": note}

    def queue_retention_action(self, patient_id: str, channel: str, message: str, offer_code: str,
                               reasoning: str, proposed_slot_id: str = "", campaign_action_id: str = "") -> dict:
        profile = self.get_patient_profile(patient_id)
        if "error" in profile:
            return profile
        if channel not in profile["allowed_channels"]:
            return _err(f"no consent for {channel}; allowed: {profile['allowed_channels']}")
        offers = self.get_retention_offers(patient_id)
        if offer_code and offer_code not in {o["offer_code"] for o in offers["eligible_offers"]}:
            return _err(f"offer {offer_code} not eligible for payer {offers['payer']}. {offers['note']}")
        problems = guardrails.check_message(message, channel)
        if problems:
            return _err("message rejected: " + "; ".join(problems))
        if channel == "sms":
            message = guardrails.ensure_sms_opt_out(message)
        if self.db.query(f"SELECT 1 FROM {self.t('retention_actions')} WHERE patient_id = :id "
                         "AND status = 'pending_review'", {"id": patient_id}):
            return _err(f"patient {patient_id} already has a pending action")
        if proposed_slot_id and not self.db.query(
                f"SELECT 1 FROM {self.t('appointment_slots')} WHERE slot_id = :id AND is_open "
                "AND slot_start > current_timestamp()", {"id": proposed_slot_id}):
            return _err(f"slot {proposed_slot_id} is not open")
        action_id = f"RA-{uuid.uuid4().hex[:10]}"
        self.db.execute(
            f"""INSERT INTO {self.t('retention_actions')}
                (action_id, run_id, patient_id, channel, offer_code, message, reasoning, churn_risk, status, created_at,
                 proposed_slot_id, campaign_action_id)
                VALUES (:aid, :run, :pid, :channel, NULLIF(:offer, ''), :msg, :why, :risk,
                        'pending_review', current_timestamp(), NULLIF(:slot, ''), NULLIF(:camp, ''))""",
            {"aid": action_id, "run": self.run_id, "pid": patient_id, "channel": channel, "offer": offer_code or "",
             "msg": message, "why": reasoning, "risk": float(profile.get("churn_risk") or 0.0),
             "slot": proposed_slot_id or "", "camp": campaign_action_id or ""})
        if campaign_action_id:
            self.db.execute(f"UPDATE {self.t('campaign_targets')} SET status = 'drafted', retention_action_id = :aid "
                            "WHERE action_id = :camp AND patient_id = :pid",
                            {"aid": action_id, "camp": campaign_action_id, "pid": patient_id})
        return {"ok": True, "action_id": action_id, "status": "pending_review", "final_message": message}

    # ------------------------------------------------------------------ treatment themes / care settings
    def get_care_recommendations(self, patient_id: str) -> dict:
        return themes.get_model(self.db, self.s).recommend(patient_id)

    def get_treatment_themes(self) -> list[dict]:
        return themes.get_model(self.db, self.s).theme_summary()

    # ------------------------------------------------------------------ capacity
    def get_utilization(self, days: int = 84) -> dict:
        days = max(14, min(int(days), 365))
        out = capacity.add_revenue_gap(self.db.query(capacity.utilization_sql(self.s), {"days": days}))
        return {"window_days": days, **out}

    def _location(self, location_id: str) -> dict | None:
        rows = self.db.query(
            f"""SELECT l.location_id, l.capacity_patients_per_day AS room_capacity,
                       (SELECT count_if(active_flag) FROM {self.s.source_table('providers')} p
                         WHERE p.location_id = l.location_id) AS active_providers,
                       (SELECT avg(revenue) FROM {self.s.source_table('visits')} v
                         WHERE v.location_id = l.location_id AND v.visit_date > date_sub(current_date(), 365))
                         AS revenue_per_visit
                FROM {self.s.source_table('locations')} l WHERE l.location_id = :loc""", {"loc": location_id})
        return rows[0] if rows else None

    def get_location_detail(self, location_id: str, days: int = 84) -> dict:
        """Weekday and provider breakdown, a 14-day forecast of idle slots, and a no-show plan."""
        loc = self._location(location_id)
        if loc is None:
            return _err(f"unknown location {location_id}")
        p = {"loc": location_id, "days": max(14, min(int(days), 365))}
        by_day = self.db.query(capacity.weekday_sql(self.s), p)
        if not by_day:
            return _err(f"no appointments for location {location_id}")
        book = self.db.query(capacity.book_sql(self.s), {"loc": location_id, "days": 14})
        fc = capacity.forecast(book, by_day, self.db.query(capacity.lead_time_sql(self.s), p),
                               int(loc["room_capacity"]), dt.date.today())
        staffed = {int(r["dow"]): min(int(loc["room_capacity"]), capacity.SLOTS_PER_PROVIDER_DAY
                                      * int(loc["active_providers"]) // (2 if int(r["dow"]) == 7 else 1))
                   for r in by_day if int(r["dow"]) != 1}
        return {"location_id": location_id, "room_capacity": loc["room_capacity"],
                "active_providers": loc["active_providers"], "by_weekday": by_day,
                "providers": self.db.query(capacity.provider_sql(self.s), p),
                "forecast_next_14_days": fc,
                "projected_idle_slots_14d": round(sum(max(f["projected_idle_slots"], 0) for f in fc), 0),
                "no_show_plan": capacity.no_show_plan(by_day, staffed)}

    def get_fill_candidates(self, location_id: str, day: str, limit: int = 25) -> dict:
        """Lapsed-but-recoverable patients of a location to invite into a specific forward day."""
        try:
            target = dt.date.fromisoformat(day)
        except ValueError:
            return _err("day must be YYYY-MM-DD")
        rows = self.db.query(capacity.fill_candidates_sql(self.s).replace("{limit}", str(int(min(limit, 100)))),
                             {"loc": location_id, "day": target.isoformat()})
        return {"location_id": location_id, "day": target.isoformat(), "candidates": rows,
                "assumed_response_rate": capacity.REACTIVATION_RESPONSE_RATE,
                "expected_bookings": round(len(rows) * capacity.REACTIVATION_RESPONSE_RATE, 1)}

    def queue_capacity_action(self, location_id: str, action_type: str, details: str,
                              expected_weekly_visits: float, reasoning: str, target_patient_ids: list | None = None,
                              target_date: str = "") -> dict:
        if action_type not in capacity.ACTION_TYPES:
            return _err(f"action_type must be one of {sorted(capacity.ACTION_TYPES)}")
        loc = self._location(location_id)
        if loc is None:
            return _err(f"unknown location {location_id}")
        visits = float(expected_weekly_visits)
        if action_type == "reduce_hours":
            if visits > 0:
                return _err("reduce_hours frees capacity; expected_weekly_visits must be 0 or negative")
        elif not 0 < visits <= 7 * float(loc["room_capacity"]):
            return _err("expected_weekly_visits must be positive and within weekly capacity")
        if len((details or "").strip()) < 40 or len((reasoning or "").strip()) < 20:
            return _err("details must be a concrete plan (who, which days, how many slots) and reasoning must cite numbers")
        targets = [str(t) for t in (target_patient_ids or [])]
        if targets:
            if action_type != "reactivation_campaign" or not target_date:
                return _err("target_patient_ids need action_type reactivation_campaign and a target_date")
            eligible = {r["patient_id"] for r in self.get_fill_candidates(location_id, target_date, 100)
                        .get("candidates", [])}
            if not set(targets) <= eligible:
                return _err(f"not eligible fill candidates for {location_id}: {sorted(set(targets) - eligible)[:5]}")
            visits = min(visits, round(len(targets) * capacity.REACTIVATION_RESPONSE_RATE * 2, 1))
        if self.db.query(f"SELECT 1 FROM {self.t('capacity_actions')} WHERE location_id = :loc "
                         "AND action_type = :a AND status = 'pending_review'", {"loc": location_id, "a": action_type}):
            return _err(f"{location_id} already has a pending {action_type}")
        action_id = f"CA-{uuid.uuid4().hex[:10]}"
        revenue = round(visits * float(loc["revenue_per_visit"] or 0), 0)
        self.db.execute(
            f"""INSERT INTO {self.t('capacity_actions')}
                (action_id, run_id, location_id, action_type, details, expected_weekly_visits,
                 expected_weekly_revenue, reasoning, status, created_at, targets, target_date)
                VALUES (:aid, :run, :loc, :a, :details, :visits, :rev, :why, 'pending_review', current_timestamp(),
                        NULLIF(:targets, ''), CAST(NULLIF(:tdate, '') AS DATE))""",
            {"aid": action_id, "run": self.run_id, "loc": location_id, "a": action_type, "details": details,
             "visits": visits, "rev": revenue, "why": reasoning, "targets": ",".join(targets),
             "tdate": target_date or ""})
        return {"ok": True, "action_id": action_id, "status": "pending_review", "expected_weekly_visits": visits,
                "expected_weekly_revenue": revenue, "targets": len(targets)}

    def get_capacity_impact(self) -> list[dict]:
        """Approved actions: utilization since approval versus the baseline captured at approval."""
        return self.db.query(capacity.impact_sql(self.s))

    def _location_utilization(self, location_id: str, days: int = 28) -> float | None:
        out = self.get_utilization(days)
        return next((r["utilization"] for r in out["locations"] if r["location_id"] == location_id), None)

    # ------------------------------------------------------------------ fill campaigns (capacity -> retention)
    def list_campaign_targets(self, limit: int = 10) -> list[dict]:
        return self.db.query(
            f"""SELECT c.action_id AS campaign_action_id, c.patient_id, c.location_id, l.location_name, l.city,
                       c.target_date, p.first_name, p.preferred_channel
                FROM {self.t('campaign_targets')} c JOIN {self.t('patients')} p USING (patient_id)
                JOIN {self.s.source_table('locations')} l ON l.location_id = c.location_id
                WHERE c.status = 'to_contact' ORDER BY c.target_date, c.created_at LIMIT {int(min(limit, 50))}""")

    def get_location_open_slots(self, location_id: str, day: str, limit: int = 8) -> list[dict]:
        return self.db.query(
            f"""SELECT sl.slot_id, date_format(sl.slot_start, 'EEE MMM d, h:mm a') AS starts, sl.provider
                FROM {self.t('appointment_slots')} sl JOIN {self.s.source_table('providers')} p
                  ON p.provider_id = sl.provider
                WHERE p.location_id = :loc AND sl.is_open AND to_date(sl.slot_start) = CAST(:day AS DATE)
                ORDER BY sl.slot_start LIMIT {int(min(limit, 20))}""", {"loc": location_id, "day": day})

    # ------------------------------------------------------------------ pricing
    def get_pricing_overview(self) -> list[dict]:
        return self.db.query(f"SELECT * EXCEPT (updated_at) FROM {self.t('service_pricing_stats')} ORDER BY service_id")

    def _pricing(self, service_id: str) -> pricing.ServicePricing | None:
        rows = self.db.query(f"SELECT * FROM {self.t('service_pricing_stats')} WHERE service_id = :id",
                             {"id": service_id})
        return pricing.ServicePricing.from_row(rows[0]) if rows else None

    def simulate_price_change(self, service_id: str, new_price: float) -> dict:
        sp = self._pricing(service_id)
        if sp is None:
            return _err(f"unknown service {service_id}")
        out = pricing.project(sp, float(new_price))
        out["guardrail_violations"] = pricing.violations(sp, float(new_price))
        out["allowed_band"] = [round(x, 2) for x in pricing.bounds(sp)]
        return out

    def optimize_price(self, service_id: str) -> dict:
        sp = self._pricing(service_id)
        return pricing.optimize(sp) if sp else _err(f"unknown service {service_id}")

    def propose_price_change(self, service_id: str, new_price: float, rationale: str) -> dict:
        sp = self._pricing(service_id)
        if sp is None:
            return _err(f"unknown service {service_id}")
        new_price = round(float(new_price), 2)
        bad = pricing.violations(sp, new_price)
        if bad:
            return _err("; ".join(bad))
        if abs(new_price - sp.current_price) < 1:
            return _err("change is under $1 - hold the price instead of proposing")
        if len((rationale or "").strip()) < 40:
            return _err("rationale must explain the evidence (elasticity, volume, competitors, retention risk)")
        if self.db.query(f"SELECT 1 FROM {self.t('price_recommendations')} WHERE service_id = :id "
                         "AND status = 'pending_review'", {"id": service_id}):
            return _err(f"{service_id} already has a pending recommendation")
        proj = pricing.project(sp, new_price)
        rec_id = f"PR-{uuid.uuid4().hex[:10]}"
        self.db.execute(
            f"""INSERT INTO {self.t('price_recommendations')}
                (rec_id, run_id, service_id, current_price, proposed_price, projected_weekly_margin_delta,
                 projected_weekly_volume_delta, rationale, status, created_at)
                VALUES (:rid, :run, :sid, :cur, :new, :dm, :dv, :why, 'pending_review', current_timestamp())""",
            {"rid": rec_id, "run": self.run_id, "sid": service_id, "cur": float(sp.current_price), "new": new_price,
             "dm": float(proj["weekly_margin_delta"]),
             "dv": float(proj["weekly_volume_projected"] - proj["weekly_volume_now"]), "why": rationale})
        return {"ok": True, "rec_id": rec_id, "status": "pending_review", "projection": proj}

    # ------------------------------------------------------------------ analytics (Databricks-native)
    def describe_metrics(self) -> str:
        return semantic.catalog_description()

    def query_metrics(self, metric_view: str, measures: list[str], dimensions: list[str] | None = None,
                      filters: list[dict] | None = None, last_n_days: int | None = None,
                      order_by: str | None = None, descending: bool = True, limit: int = 100) -> dict:
        try:
            sql, p = semantic.build_metric_query(self.s, metric_view, measures, dimensions, filters,
                                                 last_n_days, order_by, descending, limit)
        except semantic.MetricQueryError as e:
            return _err(str(e))
        rows = self.db.query(sql, p)
        return {"rows": rows, "row_count": len(rows), "sql": sql}

    def forecast_demand(self, service_id: str, weeks: int = 8) -> dict:
        """Weekly cash demand forecast with Databricks `ai_forecast`; seasonal-naive fallback."""
        weeks = max(1, min(int(weeks), 26))
        p = {"sid": service_id, "days": weeks * 7}
        try:
            rows = self.db.query(
                f"""SELECT * FROM ai_forecast(
                      TABLE(SELECT week, service_id, CAST(qty AS DOUBLE) AS qty
                            FROM {self.t('gold_weekly_service_demand')}
                            WHERE payer_group = 'cash' AND service_id = :sid),
                      horizon => date_add(current_date(), :days),
                      time_col => 'week', value_col => 'qty', group_col => 'service_id', frequency => 'week')
                    ORDER BY week""", p)
            return {"method": "ai_forecast", "service_id": service_id, "forecast": rows}
        except Exception as e:  # ai_forecast needs a serverless/pro SQL warehouse; degrade gracefully
            hist = self.db.query(
                f"""SELECT round(avg(qty), 1) AS mean_weekly_qty, round(stddev(qty), 1) AS sd_weekly_qty,
                           round(avg(avg_price), 2) AS avg_price
                    FROM (SELECT * FROM {self.t('gold_weekly_service_demand')}
                          WHERE payer_group = 'cash' AND service_id = :sid ORDER BY week DESC LIMIT 8)""",
                {"sid": service_id})
            return {"method": "trailing_8_week_mean (ai_forecast unavailable: " + str(e).splitlines()[0][:120] + ")",
                    "service_id": service_id, "weeks": weeks, "baseline": hist[0] if hist else {}}

    def get_cohort_retention(self, months_since_first_visit: int = 3, last_n_cohorts: int = 12) -> list[dict]:
        return self.db.query(
            f"""SELECT cohort_month, cohort_size, active_patients, retention_rate
                FROM {self.t('gold_cohort_retention')}
                WHERE months_since_first_visit = :m
                  AND cohort_month <= add_months(current_date(), -1 * :m)
                ORDER BY cohort_month DESC LIMIT {int(min(last_n_cohorts, 36))}""",
            {"m": int(months_since_first_visit)})

    def get_decision_audit(self, kind: str = "price", limit: int = 20) -> list[dict] | dict:
        """Every draft and human decision, replayed from the Delta Change Data Feed."""
        if kind not in self._REVIEW:
            return _err(f"kind must be one of {sorted(self._REVIEW)}")
        table, key = self._REVIEW[kind]
        name = self.t(table).replace("`", "")
        return self.db.query(
            f"""SELECT {key}, status, reviewed_by, _change_type, _commit_version, _commit_timestamp
                FROM table_changes('{name}', 0)
                WHERE _change_type IN ('insert', 'update_postimage')
                ORDER BY _commit_timestamp DESC LIMIT {int(min(limit, 100))}""")

    def _analytics_tools(self) -> list[Tool]:
        views = sorted(semantic.METRIC_VIEWS)
        all_measures = sorted({m for v in semantic.METRIC_VIEWS.values() for m in v.measures})
        all_dims = sorted({d for v in semantic.METRIC_VIEWS.values() for d in v.dimensions})
        return [
            Tool("query_metrics",
                 "Query the governed Unity Catalog metric views (same numbers as the dashboards). "
                 "Pick measures/dimensions from the chosen view; filters are structured. Semantic layer:\n"
                 + semantic.catalog_description(),
                 params(["metric_view", "measures"],
                        metric_view={"type": "string", "enum": views},
                        measures={"type": "array", "items": {"type": "string", "enum": all_measures}},
                        dimensions={"type": "array", "items": {"type": "string", "enum": all_dims}},
                        filters={"type": "array", "description": "e.g. [{dimension: payer, op: '=', value: cash}]",
                                 "items": {"type": "object", "properties": {
                                     "dimension": {"type": "string"},
                                     "op": {"type": "string", "enum": ["=", "!=", ">", ">=", "<", "<=", "in"]},
                                     "value": {}}, "required": ["dimension", "value"]}},
                        last_n_days=("integer", "Restrict to the last N days of the view's time dimension"),
                        order_by=("string", "A requested measure or dimension"),
                        descending=("boolean", "Sort descending (default true)"),
                        limit=("integer", "Max rows (default 100)")),
                 self.query_metrics),
            Tool("forecast_demand", "Forecast weekly cash demand for a service (Databricks ai_forecast) "
                 "with prediction intervals.",
                 params(["service_id"], service_id=("string", "Service id"), weeks=("integer", "Horizon in weeks")),
                 self.forecast_demand),
            Tool("get_cohort_retention", "Share of each monthly new-patient cohort still visiting N months later.",
                 params(months_since_first_visit=("integer", "N months after first visit (default 3)"),
                        last_n_cohorts=("integer", "How many recent cohorts (default 12)")),
                 self.get_cohort_retention),
            Tool("get_decision_audit", "Audit trail of agent drafts and human approvals (Delta Change Data Feed).",
                 params(kind={"type": "string", "enum": ["lead", "retention", "price", "capacity", "care"]},
                        limit=("integer", "Max rows")),
                 self.get_decision_audit),
        ]

    # ------------------------------------------------------------------ briefing / audit
    def recent_agent_activity(self, hours: int = 24) -> dict:
        p = {"h": int(hours)}
        since = "created_at > timestampadd(HOUR, -1 * :h, current_timestamp())"
        return {
            "lead_actions": self.db.query(
                f"SELECT lead_id, action_type, channel, priority, reasoning FROM {self.t('lead_actions')} "
                f"WHERE {since} ORDER BY priority, created_at LIMIT 30", p),
            "retention_actions": self.db.query(
                f"SELECT patient_id, channel, offer_code, churn_risk, reasoning FROM {self.t('retention_actions')} "
                f"WHERE {since} ORDER BY churn_risk DESC LIMIT 30", p),
            "price_recommendations": self.db.query(
                f"SELECT service_id, current_price, proposed_price, projected_weekly_margin_delta, rationale "
                f"FROM {self.t('price_recommendations')} WHERE {since}", p),
            "capacity_actions": self.db.query(
                f"SELECT location_id, action_type, expected_weekly_visits, expected_weekly_revenue "
                f"FROM {self.t('capacity_actions')} WHERE {since}", p),
            "agent_runs": self.db.query(
                f"SELECT agent, status, steps, tool_calls, failed_tool_calls FROM {self.t('agent_runs')} "
                f"WHERE started_at > timestampadd(HOUR, -1 * :h, current_timestamp())", p),
        }

    def log_run(self, agent: str, result, started_at: str) -> None:
        failed = sum(1 for c in result.tool_calls if not c["ok"])
        self.db.execute(
            f"""INSERT INTO {self.t('agent_runs')}
                (run_id, agent, started_at, finished_at, status, steps, tool_calls, failed_tool_calls, summary)
                VALUES (:run, :agent, CAST(:started AS TIMESTAMP), current_timestamp(), :status,
                        :steps, :calls, :failed, :summary)""",
            {"run": self.run_id, "agent": agent, "started": started_at, "status": result.status,
             "steps": int(result.steps), "calls": len(result.tool_calls), "failed": failed,
             "summary": (result.final_text or "")[:4000]})

    # ------------------------------------------------------------------ human review (App only)
    _REVIEW = {
        "lead": ("lead_actions", "action_id"),
        "retention": ("retention_actions", "action_id"),
        "price": ("price_recommendations", "rec_id"),
        "capacity": ("capacity_actions", "action_id"),
        "care": ("care_reports", "report_id"),
    }

    def pending(self, kind: str) -> list[dict]:
        table, _ = self._REVIEW[kind]
        return self.db.query(f"SELECT * FROM {self.t(table)} WHERE status = 'pending_review' ORDER BY created_at DESC")

    def review(self, kind: str, item_id: str, decision: str, reviewer: str) -> dict:
        if kind not in self._REVIEW or decision not in ("approved", "rejected"):
            return _err("invalid review request")
        table, key = self._REVIEW[kind]
        rows = self.db.query(f"SELECT * FROM {self.t(table)} WHERE {key} = :id AND status = 'pending_review'",
                             {"id": item_id})
        if not rows:
            return _err("item not found or already reviewed")
        self.db.execute(
            f"UPDATE {self.t(table)} SET status = :d, reviewed_by = :who, reviewed_at = current_timestamp() "
            f"WHERE {key} = :id", {"d": decision, "who": reviewer, "id": item_id})
        if decision == "approved":
            item = rows[0]
            if kind == "care":  # the newest approved guidance replaces the patient's previous one
                self.db.execute(f"UPDATE {self.t('care_reports')} SET status = 'superseded' WHERE patient_id = :pid "
                                "AND status = 'approved' AND report_id <> :id",
                                {"pid": item["patient_id"], "id": item_id})
            if kind == "capacity":
                self.db.execute(f"UPDATE {self.t('capacity_actions')} SET baseline_utilization = :u WHERE action_id = :id",
                                {"u": self._location_utilization(item["location_id"]) or 0.0, "id": item_id})
                if item.get("targets"):
                    self.db.execute(
                        f"""INSERT INTO {self.t('campaign_targets')}
                            SELECT :id, t.patient_id, :loc, CAST(:day AS DATE), 'to_contact', NULL, current_timestamp()
                            FROM (SELECT explode(split(:targets, ',')) AS patient_id) t""",
                        {"id": item_id, "loc": item["location_id"], "day": str(item["target_date"])[:10],
                         "targets": item["targets"]})
            if kind == "price":
                p = {"sid": item["service_id"], "price": float(item["proposed_price"])}
                self.db.execute(f"UPDATE {self.t('services')} SET current_price = :price WHERE service_id = :sid", p)
                self.db.execute(f"UPDATE {self.t('service_pricing_stats')} SET current_price = :price "
                                "WHERE service_id = :sid", p)
                self.db.execute(f"INSERT INTO {self.t('price_history')} (service_id, effective_date, price) "
                                "VALUES (:sid, current_date(), :price)", p)
        return {"ok": True}

    # ------------------------------------------------------------------ registries per agent
    def _shared(self) -> list[Tool]:
        return [
            Tool("get_service_prices", "Current cash price list for all services.", params(), self.get_service_prices),
            Tool("get_open_slots", "Open appointment slots in the next N days.",
                 params(days_ahead=("integer", "Days to look ahead (default 5)"),
                        limit=("integer", "Max slots (default 8)")), self.get_open_slots),
        ]

    def lead_registry(self) -> ToolRegistry:
        return ToolRegistry(self._shared() + [
            Tool("list_scored_leads", "New, uncontacted, consented leads from the last 30 days, ranked by urgency "
                 "tier (emergency > urgent > soon > routine), then deadline (overdue first), then expected value "
                 "(skill-adjusted P(convert) x patient value). Includes respond_by and deadline_state.",
                 params(limit=("integer", "Max leads"), min_score=("number", "Minimum score 0-1")),
                 self.list_scored_leads),
            Tool("get_lead", "Full lead details incl. their message, red flags and consented channels.",
                 params(["lead_id"], lead_id=("string", "Lead id")), self.get_lead),
            Tool("queue_lead_action", "Queue a drafted action for human review. Never sends anything.",
                 params(["lead_id", "action_type", "channel", "message", "reasoning"],
                        lead_id=("string", "Lead id"),
                        action_type={"type": "string", "enum": list(LEAD_ACTIONS)},
                        channel={"type": "string", "enum": list(CHANNELS)},
                        message=("string", "Message text (for phone: the call script)"),
                        reasoning=("string", "Why this action, citing score/reasons"),
                        priority={"type": "string", "enum": ["high", "normal", "low"]},
                        proposed_slot_id=("string", "Optional slot_id from get_open_slots")),
                 self.queue_lead_action),
        ])

    def retention_registry(self) -> ToolRegistry:
        return ToolRegistry(self._shared() + [
            Tool("list_at_risk_patients", "Active patients ranked by expected annual revenue loss "
                 "(churn risk x LTV), excluding anyone contacted in the last 30 days.",
                 params(limit=("integer", "Max patients"), min_risk=("number", "Minimum churn risk 0-1")),
                 self.list_at_risk_patients),
            Tool("get_patient_profile", "Patient profile, recent visits, risk reasons and consented channels.",
                 params(["patient_id"], patient_id=("string", "Patient id")), self.get_patient_profile),
            Tool("get_care_recommendations", "Treatment theme plus non-medical care settings (weekday, cadence, "
                 "payment option, booking channel, add-ons, provider) that similar patients who stayed in care used, "
                 "with evidence strength.", params(["patient_id"], patient_id=("string", "Patient id")),
                 self.get_care_recommendations),
            Tool("get_retention_offers", "Offers this patient is eligible for (depends on payer).",
                 params(["patient_id"], patient_id=("string", "Patient id")), self.get_retention_offers),
            Tool("list_campaign_targets", "Patients in owner-approved fill campaigns still to be invited, with "
                 "their location and target day. Work these first.",
                 params(limit=("integer", "Max patients")), self.list_campaign_targets),
            Tool("get_location_open_slots", "Open slots at a location on a given day.",
                 params(["location_id", "day"], location_id=("string", "Location id"),
                        day=("string", "Day YYYY-MM-DD"), limit=("integer", "Max slots")),
                 self.get_location_open_slots),
            Tool("queue_retention_action", "Queue a re-engagement message for human review. Never sends.",
                 params(["patient_id", "channel", "message", "offer_code", "reasoning"],
                        patient_id=("string", "Patient id"),
                        channel={"type": "string", "enum": list(CHANNELS)},
                        message=("string", "Message text (for phone: the call script)"),
                        offer_code=("string", "Eligible offer_code, or '' for none"),
                        reasoning=("string", "Why, citing risk reasons"),
                        proposed_slot_id=("string", "Optional open slot_id offered in the message"),
                        campaign_action_id=("string", "The campaign_action_id when inviting a campaign target")),
                 self.queue_retention_action),
        ])

    def pricing_registry(self) -> ToolRegistry:
        return ToolRegistry(self._analytics_tools() + [
            Tool("get_pricing_overview", "Per-service price, cost, weekly cash volume, estimated elasticity "
                 "(with standard error), policy floor/ceiling and competitor range.", params(),
                 self.get_pricing_overview),
            Tool("simulate_price_change", "Project weekly volume, revenue and margin at a candidate price, "
                 "plus guardrail violations.",
                 params(["service_id", "new_price"], service_id=("string", "Service id"),
                        new_price=("number", "Candidate price")), self.simulate_price_change),
            Tool("optimize_price", "Margin-maximising whole-dollar price inside the guardrail band.",
                 params(["service_id"], service_id=("string", "Service id")), self.optimize_price),
            Tool("propose_price_change", "Submit a price change for owner approval. Not applied until approved.",
                 params(["service_id", "new_price", "rationale"], service_id=("string", "Service id"),
                        new_price=("number", "Proposed price"),
                        rationale=("string", "Evidence-based rationale with numbers")),
                 self.propose_price_change),
        ])

    def capacity_registry(self) -> ToolRegistry:
        return ToolRegistry([
            Tool("get_utilization", "Per location: utilization of effective capacity (min of rooms and staffed "
                 "slots), staffed-slot fill, share of fully booked days, no-show and lost-slot rates, weekly trend, "
                 "lapsed patients, diagnosed problems, and weekly revenue gap vs the network's top quartile.",
                 params(days=("integer", "Trailing window in days (default 84)")), self.get_utilization),
            Tool("get_location_detail", "One location: weekday and provider breakdown, 14-day forecast of idle "
                 "slots (forward book + booking lead-time curve), and a no-show plan (overbooking on full days, "
                 "standby list size on others).",
                 params(["location_id"], location_id=("string", "Location id"),
                        days=("integer", "Trailing window in days (default 84)")), self.get_location_detail),
            Tool("get_fill_candidates", "Lapsed-but-recoverable patients of a location ranked for a forward day "
                 "(recency, past visits, usual weekday match), consented and not recently contacted.",
                 params(["location_id", "day"], location_id=("string", "Location id"),
                        day=("string", "Target day YYYY-MM-DD"), limit=("integer", "Max candidates (default 25)")),
                 self.get_fill_candidates),
            Tool("queue_capacity_action", "Queue a corrective action for owner approval. For a fill campaign use "
                 "reactivation_campaign with target_patient_ids from get_fill_candidates and target_date; once "
                 "approved, the Retention agent drafts each invitation with a slot on that day.",
                 params(["location_id", "action_type", "details", "expected_weekly_visits", "reasoning"],
                        location_id=("string", "Location id"),
                        action_type={"type": "string", "enum": sorted(capacity.ACTION_TYPES)},
                        details=("string", "Concrete plan: who, which days, how many slots"),
                        expected_weekly_visits=("number", "Conservative extra completed visits per week "
                                                          "(0 or negative for reduce_hours)"),
                        reasoning=("string", "Why, citing the numbers"),
                        target_patient_ids={"type": "array", "items": {"type": "string"},
                                            "description": "Fill-campaign patients (reactivation_campaign only)"},
                        target_date=("string", "Fill-campaign day YYYY-MM-DD")),
                 self.queue_capacity_action),
            Tool("get_capacity_impact", "Approved actions with utilization since approval vs baseline.", params(),
                 self.get_capacity_impact),
        ])

    def briefing_registry(self) -> ToolRegistry:
        return ToolRegistry(self._analytics_tools() + [
            Tool("get_kpis", "Clinic KPIs: leads, conversion, active patients, revenue, risk, pending approvals.",
                 params(), self.get_kpis),
            Tool("get_treatment_themes", "Treatment themes (patient clusters by service mix and visit rhythm) "
                 "with size and share still active.", params(), self.get_treatment_themes),
            Tool("get_sla_report", "Lead response-time performance by urgency tier: share answered within the "
                 "target, median/p90 hours to response, overdue now and overdue because a draft awaits approval.",
                 params(days=("integer", "Lookback days (default 14)")), self.get_sla_report),
            Tool("recent_agent_activity", "What the agents queued in the last N hours.",
                 params(hours=("integer", "Lookback hours (default 24)")), self.recent_agent_activity),
        ])

    def copilot_registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        for r in (self.lead_registry(), self.retention_registry(), self.pricing_registry(), self.briefing_registry(),
                  self.capacity_registry()):
            for tool in r:
                reg.add(tool)
        return reg
