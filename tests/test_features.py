"""Urgency triage, capacity sizing, treatment themes/recommendations and care guidance reports."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from chiro import capacity, care, themes, urgency
from chiro.config import Settings
from chiro.tools import ClinicTools


class DB:
    def __init__(self, rows=None):
        self.rows, self.writes = rows or {}, []

    def query(self, sql, params=None):
        return next((list(v) for k, v in self.rows.items() if k in sql), [])

    def execute(self, sql, params=None):
        self.writes.append((sql, params))


# ---------------------------------------------------------------- urgency
def test_urgency_tiers():
    assert urgency.classify("worst headache of my life, came on suddenly")[0] == "emergency"
    tier, why = urgency.classify("Threw out my back this morning, can't stand up straight, pain 8/10")
    assert tier == "urgent" and "pain 8/10" in why
    assert urgency.classify("My neck has been getting worse this week")[0] == "soon"
    assert urgency.classify("Stiff back for years, looking for maintenance and your rates")[0] == "routine"
    assert urgency.classify("Lower back pain", ai_intent="urgent_medical")[0] == "soon"


def test_priority_orders_by_tier_then_score():
    keys = [urgency.priority_key("routine", 0.99), urgency.priority_key("soon", 0.01),
            urgency.priority_key("urgent", 0.5), urgency.priority_key("emergency", 0.0)]
    assert keys == sorted(keys)


def test_negation_and_inability():
    from chiro.guardrails import detect_red_flags
    assert detect_red_flags("no fever, no numbness, just a stiff back") == []
    assert detect_red_flags("I don't have any bladder problems") == []
    assert detect_red_flags("I can't control my bladder") == ["bladder/bowel dysfunction"]  # inability != negation
    assert detect_red_flags("not sure if it's a fever") == ["fever with pain"]  # hedge keeps the flag
    assert detect_red_flags("no fever but my legs keep giving out") == ["progressive weakness"]
    assert urgency.classify("Need an appointment as soon as possible")[0] == "soon"  # 'appointment' is not "n't"


def test_rules_never_miss_an_emergency_in_the_golden_set():
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "data", "triage_golden.jsonl")
    cases = [json.loads(line) for line in open(path)]
    for c in cases:
        if c["tier"] == "emergency":
            assert urgency.classify(c["message"], c["complaint"])[0] == "emergency", c["message"]
        else:
            assert urgency.classify(c["message"], c["complaint"])[0] != "emergency", c["message"]


def test_ensemble_takes_the_more_severe_tier_and_rules_floor():
    rules_urgent = ("urgent", ["acute onset"])
    assert urgency.combine(rules_urgent, {"tier": "routine", "red_flags": [], "reason": "x"})[0] == "urgent"
    assert urgency.combine(("routine", []), {"tier": "urgent", "red_flags": [], "reason": "x"})[0] == "urgent"
    tier, why = urgency.combine(("soon", []), {"tier": "emergency", "red_flags": ["saddle numbness"], "reason": ""})
    assert tier == "emergency" and "verify by phone" in why[-1]
    assert urgency.combine(("emergency", ["red flag"]), {"tier": "routine", "red_flags": [], "reason": ""})[0] == "emergency"
    assert "rules only" in urgency.combine(("soon", []), None)[1][-1]


def test_llm_triage_parses_and_fails_safe():
    reply = lambda text: NS(chat=NS(completions=NS(create=lambda **kw: NS(choices=[NS(message=NS(content=text))]))))
    out = urgency.llm_triage(reply('ok: {"tier": "urgent", "red_flags": [], "reason": "fell today"}'), "m", "x")
    assert out == {"tier": "urgent", "red_flags": [], "reason": "fell today"}
    assert urgency.llm_triage(reply("no idea"), "m", "x") is None
    assert urgency.llm_triage(reply('{"tier": "panic"}'), "m", "x") is None


def test_priority_rank_tier_then_deadline_then_value_with_skill_shrinkage():
    import datetime as dt

    from chiro import priority
    now = dt.datetime(2026, 10, 8, 12, tzinfo=dt.timezone.utc)
    later, past = (now + dt.timedelta(hours=5)).isoformat(), (now - dt.timedelta(hours=1)).isoformat()
    leads = [
        {"lead_id": "routine-overdue", "urgency": "routine", "score": 0.9, "insurance_type": "cash", "respond_by": past},
        {"lead_id": "soon-cash", "urgency": "soon", "score": 0.2, "insurance_type": "cash", "respond_by": later},
        {"lead_id": "soon-ppo", "urgency": "soon", "score": 0.9, "insurance_type": "ppo", "respond_by": later},
        {"lead_id": "soon-overdue", "urgency": "soon", "score": 0.1, "insurance_type": "ppo", "respond_by": past},
        {"lead_id": "urgent", "urgency": "urgent", "score": 0.1, "insurance_type": "medicare", "respond_by": later},
    ]
    values = {"cash": 430.0, "ppo": 350.0, "medicare": 345.0}
    useless = priority.rank(leads, now, base_rate=0.5, auc=0.51, payer_value=values)
    assert [r["lead_id"] for r in useless] == ["urgent", "soon-overdue", "soon-cash", "soon-ppo", "routine-overdue"]
    assert useless[2]["p_convert"] == 0.488  # AUC 0.51: scores barely move off the base rate
    skilled = priority.rank(leads, now, base_rate=0.5, auc=0.75, payer_value=values)
    assert [r["lead_id"] for r in skilled][2:4] == ["soon-ppo", "soon-cash"]  # a skilled model's score counts
    assert skilled[1]["deadline_state"] == "overdue" and skilled[0]["queue_position"] == 1


def test_migrate_adds_missing_columns_only():
    from chiro.schema import migrate

    class Desc(DB):
        def query(self, sql, params=None):
            if "DESCRIBE" in sql and "lead_scores" in sql:
                return [{"col_name": c} for c in ("lead_id", "score", "urgency")]
            if "DESCRIBE" in sql:
                raise RuntimeError("table not found")
            return []
    db = Desc()
    done = migrate(db, Settings())
    alters = [q for q, _ in db.writes if q.startswith("ALTER")]
    assert len(alters) == 1 and "urgency_reasons" in alters[0] and "urgency STRING" not in alters[0]
    assert done == ["lead_scores: +urgency_reasons, respond_within_hours, priority"]


def test_score_leads_adds_triage_columns():
    from chiro import models
    leads = pd.DataFrame([{"lead_id": "L1", "status": "new", "source": "website_form", "complaint": "neck_pain",
                           "insurance_type": "ppo", "distance_miles": 3.0, "message": "Can't turn my neck since this morning"}])
    model = NS(feature_columns_=["distance_miles"], predict_proba=lambda X: np.array([[0.5, 0.5]]))
    out = models.score_leads(model, leads)
    assert list(out.columns) == models.LEAD_SCORE_COLUMNS
    assert out.loc[0, "urgency"] == "urgent" and out.loc[0, "respond_within_hours"] == 1.0


# ---------------------------------------------------------------- capacity
def _loc(lid, util, cap=100, done=None):
    return {"location_id": lid, "utilization": util, "effective_capacity_per_day": cap,
            "completed_per_day": done or util * cap, "open_days_per_week": 6, "revenue_per_visit": 100.0}


def test_revenue_gap_uses_peer_benchmark():
    out = capacity.add_revenue_gap([_loc("A", 0.10), _loc("B", 0.20), _loc("C", 0.30), _loc("D", 0.40)])
    assert out["benchmark_utilization"] == 0.40
    a = out["locations"][0]
    assert a["location_id"] == "A" and a["visits_gap_per_day"] == 30 and a["weekly_revenue_gap"] == 18000
    assert out["locations"][-1]["weekly_revenue_gap"] == 0


def test_diagnosis_names_the_problem():
    staffed = {"share_days_fully_booked": 0.9, "room_utilization": 0.3, "no_show_rate": 0.08, "utilization": 0.95,
               "weekly_trend": 0, "completed_per_day": 15, "open_days_per_week": 6}
    assert capacity.diagnose(staffed)[0].startswith("staffing-constrained")
    leaky = dict(staffed, share_days_fully_booked=0.1, no_show_rate=0.2, utilization=0.7)
    assert any("no-shows" in d for d in capacity.diagnose(leaky))
    declining = dict(leaky, no_show_rate=0.05, weekly_trend=-4.0)
    assert any("declining" in d for d in capacity.diagnose(declining))
    lopsided = dict(leaky, no_show_rate=0.05, weekday_spread=1.1)
    assert any("weekday imbalance" in d for d in capacity.diagnose(lopsided))


CAP_ROW = {"room_capacity": 50, "active_providers": 3, "revenue_per_visit": 100.0, "location_id": "L1"}


def test_capacity_action_guardrails():
    t = ClinicTools(DB({"AS room_capacity": [CAP_ROW]}), Settings(), run_id="t")
    plan = "Call the 900 lapsed patients of this location and offer Tue/Thu afternoon slots; two front-desk hours/day."
    assert "action_type" in t.queue_capacity_action("L1", "fire_everyone", plan, 5, "utilization 10% vs 15%")["error"]
    assert "positive" in t.queue_capacity_action("L1", "reactivation_campaign", plan, 999, "utilization 10% vs 15%")["error"]
    assert "reduce_hours" in t.queue_capacity_action("L1", "reduce_hours", plan, 5, "utilization 10% vs 15%")["error"]
    assert "target_date" in t.queue_capacity_action("L1", "local_marketing", plan, 5, "utilization 10% vs 15%",
                                                    ["P1"])["error"]
    ok = t.queue_capacity_action("L1", "reactivation_campaign", plan, 12, "utilization 10% vs benchmark 15%")
    assert ok["ok"] and ok["expected_weekly_revenue"] == 1200


def test_fill_campaign_targets_must_be_eligible_and_cap_the_estimate():
    db = DB({"AS room_capacity": [CAP_ROW], "fit_score": [{"patient_id": f"P{i}"} for i in range(10)]})
    t = ClinicTools(db, Settings(), run_id="t")
    plan = "Invite ten lapsed Tuesday regulars into next Tuesday's empty afternoon; front desk follows up by phone."
    assert "not eligible" in t.queue_capacity_action("L1", "reactivation_campaign", plan, 5, "30 idle slots forecast Tuesday",
                                                     ["P1", "X9"], "2026-10-13")["error"]
    ok = t.queue_capacity_action("L1", "reactivation_campaign", plan, 50, "30 idle slots forecast Tuesday",
                                 [f"P{i}" for i in range(10)], "2026-10-13")
    assert ok["ok"] and ok["targets"] == 10 and ok["expected_weekly_visits"] == 3.0  # capped at 2x response rate


def test_approving_a_campaign_creates_targets_and_records_baseline():
    item = {"action_id": "CA-1", "location_id": "L1", "targets": "P1,P2", "target_date": "2026-10-13",
            "status": "pending_review"}
    db = DB({"`capacity_actions` WHERE action_id": [item]})
    t = ClinicTools(db, Settings(), run_id="t")
    t.get_utilization = lambda days=84: {"locations": [{"location_id": "L1", "utilization": 0.41}]}
    assert t.review("capacity", "CA-1", "approved", "owner")["ok"]
    sqls = [w[0] for w in db.writes]
    assert any("baseline_utilization" in q for q in sqls) and any("campaign_targets" in q for q in sqls)
    assert db.writes[1][1]["u"] == 0.41


def test_forecast_adds_expected_late_bookings_and_caps_at_capacity():
    import datetime as dt
    today = dt.date(2026, 10, 8)  # Thursday
    book = [{"day": "2026-10-09", "staffed_slots": 48, "booked_now": 20},   # Friday, 1 day ahead
            {"day": "2026-10-19", "staffed_slots": 48, "booked_now": 10}]   # Monday, 11 days ahead
    by_weekday = [{"dow": 6, "booked_per_day": 30, "lost_slot_rate": 0.1},
                  {"dow": 2, "booked_per_day": 60, "lost_slot_rate": 0.2}]
    curve = [{"k": k, "share_booked_k_days_ahead": max(0.0, 1 - k / 20)} for k in range(31)]
    fri, mon = capacity.forecast(book, by_weekday, curve, room_capacity=40, today=today)
    assert fri["effective_capacity"] == 40 and fri["expected_final_bookings"] == 21.5  # 20 + 30 * (1 - 0.95)
    assert mon["expected_final_bookings"] == 40 and mon["projected_idle_slots"] == 0  # 10 + 60 * 0.55 = 43 -> capped
    assert fri["projected_idle_slots"] == 18.5


def test_no_show_plan_overbooks_only_full_days_within_risk():
    plan = capacity.no_show_plan([{"dow": 2, "weekday": "Monday", "booked_per_day": 32, "lost_slot_rate": 0.15},
                                  {"dow": 6, "weekday": "Friday", "booked_per_day": 12, "lost_slot_rate": 0.15}],
                                 {2: 32, 6: 32})
    mon, fri = plan
    assert mon["full"] and mon["recommended_overbook"] >= 1 and mon["overflow_risk"] <= capacity.MAX_OVERBOOK_RISK
    assert not fri["full"] and fri["recommended_standby_list"] == 4


# ---------------------------------------------------------------- scenario data
def _scenario_inputs():
    locs = [{"location_id": "LOC002", "room_capacity": 48, "providers": 1},
            {"location_id": "LOC019", "room_capacity": 60, "providers": 4},
            {"location_id": "LOC001", "room_capacity": 60, "providers": 4}]
    provs = {l["location_id"]: [f"{l['location_id']}-P{i}" for i in range(l["providers"])] for l in locs}
    pats = {l["location_id"]: [f"{l['location_id']}-PT{i}" for i in range(200)] for l in locs}
    return locs, provs, pats


def test_scenario_plants_documented_problems():
    import datetime as dt

    from chiro import scenario
    locs, provs, pats = _scenario_inputs()
    rows = pd.DataFrame(scenario.appointments(locs, provs, pats, dt.date(2026, 7, 13), dt.date(2026, 10, 4)))
    rows["day"] = pd.to_datetime(rows["appointment_date"])
    assert (rows["day"].dt.weekday != 6).all()  # Sundays closed
    per_day = rows[rows.day.dt.weekday < 5].groupby(["location_id", "day"]).size()
    assert per_day["LOC002"].median() == 16  # one provider: staffed slots always full
    ns = rows.groupby("location_id")["status"].apply(lambda s: (s == "No-Show").mean())
    assert ns["LOC019"] > ns["LOC001"] + 0.08
    assert rows["lead_time_days"].between(0, 30).all()


def test_slot_book_is_consistent_with_staffing():
    import datetime as dt

    from chiro import scenario
    locs, provs, _ = _scenario_inputs()
    book = pd.DataFrame(scenario.slot_book(locs, provs, dt.date(2026, 10, 8), days=7))
    assert book["slot_id"].is_unique
    sat = book[pd.to_datetime(book.slot_start).dt.weekday == 5]
    assert len(sat[sat.provider == "LOC002-P0"]) == 8  # half day
    near = book[pd.to_datetime(book.slot_start).dt.date == dt.date(2026, 10, 9)]
    far = book[pd.to_datetime(book.slot_start).dt.date == dt.date(2026, 10, 15)]
    assert (~near.is_open).mean() > (~far.is_open).mean()  # nearer days are more booked


def test_scenario_leads_have_responses_only_in_the_past():
    import datetime as dt

    from chiro import scenario
    corpus = [{"message": f"msg {t} {i}", "tier": t, "complaint": "neck_pain"}
              for t in scenario.TIER_MIX for i in range(5)]
    now = dt.datetime(2026, 10, 8, 12)
    leads, actions = scenario.leads(corpus, now, ["LOC001"], days=10, per_day=20)
    assert leads and actions and all(a["status"] == "approved" for a in actions)
    by_id = {l["lead_id"]: l for l in leads}
    for a in actions:
        assert dt.datetime.fromisoformat(a["reviewed_at"]) <= now
        assert by_id[a["lead_id"]]["status"] in ("booked", "lost", "contacted")
    assert any(l["status"] == "new" for l in leads)  # recent ones still waiting


# ---------------------------------------------------------------- themes / recommender
def _patients(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rehab = i % 2 == 0
        tuesday = rng.random() < 0.5
        rows.append({
            "patient_id": f"P{i}", "age_band": "35-44", "home_location_id": "L1", "primary_complaint": "low_back_pain",
            "insurance_type": "cash", "preferred_channel": "sms", "n_visits": rng.integers(4, 20),
            "adj_share": 0.2 if rehab else 0.7, "pt_share": 0.6 if rehab else 0.05, "massage_share": 0.1,
            "reeval_share": 0.1, "avg_gap_days": 7 if rehab else 30, "top_weekday": 3 if tuesday else 6,
            "top_payment": "Self-Pay", "top_provider": "PRV1", "no_show_rate": 0.1, "top_booking_channel": "Phone",
            # Tuesday patients stay in care far more often: the recommender should find that.
            "status": "Active" if rng.random() < (0.8 if tuesday else 0.3) else "Lapsed",
        })
    return pd.DataFrame(rows)


def test_themes_and_recommendations_find_the_signal():
    m = themes.ThemeModel(_patients(), k=2)
    names = {t["name"] for t in m.theme_summary()}
    assert any("Active rehab" in n and "weekly" in n for n in names)
    friday = m.df[m.df["weekday"] == "Friday"].iloc[0]["patient_id"]
    recs = {r["aspect"]: r for r in m.recommend(friday)["recommendations"]}
    wd = recs["visit weekday"]
    assert wd["current"] == "Friday" and wd["recommended"] == "Tuesday" and wd["evidence"] == "strong"


def test_medicare_patients_get_no_payment_steering():
    df = _patients()
    df["insurance_type"] = "medicare"
    m = themes.ThemeModel(df, k=2)
    assert "payment option" not in {r["aspect"] for r in m.recommend("P1")["recommendations"]}


# ---------------------------------------------------------------- care guidance
GOOD = {"staff_summary": "Sam is 6 of 12 visits in; home stretches are the main gap. Raise it next visit.",
        "patient_intro": "Hi Sam, here's what will help you get the most from your plan.",
        "items": [{"advice": "Do your hip stretches every morning",
                   "if_skipped": "The stiffness often comes back and it can take extra visits to regain progress."}]}
PROFILE = {"patient_id": "P1", "first_name": "Sam", "days_since_last_visit": 9, "care_plan_visits": 12,
           "plan_progress": 0.5, "churn_risk": 0.4, "recent_visits": [{"status": "no_show"}]}


class LLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw["messages"])
        return NS(choices=[NS(message=NS(content=self.replies.pop(0)))])


def test_care_language_checks():
    assert care.check_care_language("You have a disc disorder that will definitely get worse")
    assert care.check_care_language("This will cure your back")
    assert not care.check_care_language("Skipping stretches tends to bring the stiffness back.")


def test_care_report_retries_then_queues_for_review():
    bad = dict(GOOD, items=[{"advice": "Keep going", "if_skipped": "You will definitely need surgery"}])
    db = DB({"care_notes": [{"note": "Hip stretches daily; skipping tends to bring stiffness back"}]})
    llm = LLM([json.dumps(bad), "Here you go:\n" + json.dumps(GOOD)])
    out = care.CareReports(db, Settings()).draft(llm, "m", PROFILE)
    assert out["ok"] and len(llm.calls) == 2 and "certainty" in llm.calls[1][-1]["content"]
    sql, p = db.writes[-1]
    assert "care_reports" in sql and "If this slips" in p["patient"] and "no shows in last 8 bookings 1" in p["staff"]


def test_care_report_needs_staff_notes_and_rejects_persistent_violations():
    assert "care note" in care.CareReports(DB(), Settings()).draft(LLM([]), "m", PROFILE)["error"]
    db = DB({"care_notes": [{"note": "x" * 20}]})
    bad = json.dumps(dict(GOOD, staff_summary="Diagnosis: lumbar disorder"))
    assert "rejected" in care.CareReports(db, Settings()).draft(LLM([bad, bad]), "m", PROFILE)["error"]
    assert not db.writes


# ---------------------------------------------------------------- lead_scores refresh
def _lead(lid, status="new", days_ago=1, msg="Neck stiff for a few days, getting worse"):
    now = pd.Timestamp("2026-10-08 12:00")
    return {"lead_id": lid, "status": status, "created_at": now - pd.Timedelta(days=days_ago), "source": "website_form",
            "complaint": "neck_pain", "insurance_type": "cash", "distance_miles": 3.0, "message": msg}


def test_refresh_keeps_first_tier_and_scores_open_leads():
    from chiro import models
    now = pd.Timestamp("2026-10-08 12:00")
    model = NS(feature_columns_=["distance_miles"], predict_proba=lambda X: np.array([[0.3, 0.7]] * len(X)))
    previous = pd.DataFrame([
        # Answered since the last run: tier must survive, row must not disappear.
        {"lead_id": "ANSWERED", "score": 0.5, "reasons": "r", "red_flags": "", "urgency": "urgent",
         "urgency_reasons": "LLM: urgent", "respond_within_hours": 1.0, "priority": 3.5, "scored_at": now},
        # Still open: score refreshes, tier stays even though rules would now say 'soon'.
        {"lead_id": "OPEN", "score": 0.1, "reasons": "r", "red_flags": "", "urgency": "urgent",
         "urgency_reasons": "LLM: urgent", "respond_within_hours": 1.0, "priority": 3.1, "scored_at": now},
        {"lead_id": "OLD", "score": 0.2, "reasons": "r", "red_flags": "", "urgency": "routine",
         "urgency_reasons": "", "respond_within_hours": 24.0, "priority": 1.2, "scored_at": now},
    ])
    leads = pd.DataFrame([_lead("ANSWERED", "contacted"), _lead("OPEN"),
                          _lead("FAST", "booked", msg="Threw my back out this morning, can't stand"),  # never seen open
                          _lead("NEW"), _lead("OLD", "lost", days_ago=400)])
    out = models.refresh_lead_scores(previous, model, leads, None, 30, now).set_index("lead_id")
    assert set(out.index) == {"ANSWERED", "OPEN", "FAST", "NEW", "OLD"}
    assert out.loc["ANSWERED", "urgency"] == "urgent" and out.loc["ANSWERED", "score"] == 0.5
    assert out.loc["OPEN", "urgency"] == "urgent" and out.loc["OPEN", "score"] == 0.7
    assert out.loc["OPEN", "priority"] == urgency.priority_key("urgent", 0.7)
    assert out.loc["FAST", "urgency"] == "urgent" and out.loc["FAST", "score"] is None  # triaged late, no score
    assert out.loc["NEW", "urgency"] == "soon" and out.loc["NEW", "score"] == 0.7
    assert out.loc["OLD", "urgency"] == "routine"
    assert all(v is None or v == v for col in out.columns for v in out[col])  # real nulls, never NaN


def test_only_untriaged_leads_in_the_window_go_to_the_llm():
    from chiro import models
    now = pd.Timestamp("2026-10-08 12:00")
    previous = pd.DataFrame([{"lead_id": "SEEN", "urgency": "soon"}, {"lead_id": "UNTRIAGED", "urgency": None}])
    leads = pd.DataFrame([_lead("SEEN"), _lead("UNTRIAGED", "contacted"), _lead("BRAND_NEW"),
                          _lead("ANCIENT", days_ago=90)])
    todo = models.leads_needing_triage(previous, leads, 30, now)
    assert sorted(todo["lead_id"]) == ["BRAND_NEW", "UNTRIAGED"]
