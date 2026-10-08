"""Clinic Copilot - Databricks App.

Chat with the Copilot agent, and approve / reject what the scheduled agents queued.
Runs as the app's service principal; reads/writes Unity Catalog through a SQL warehouse
and reasons with a Foundation Model API endpoint (both declared as app resources in the bundle).
"""
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402
from chiro import prompts  # noqa: E402
from chiro.agent import run_agent  # noqa: E402
from chiro.capacity import ACTION_TYPES  # noqa: E402
from chiro.care import CATEGORIES as CARE_CATEGORIES, CareReports  # noqa: E402
from chiro.config import Settings  # noqa: E402
from chiro.llm import get_llm_client  # noqa: E402
from chiro.sql import WarehouseSql  # noqa: E402
from chiro.tools import ClinicTools  # noqa: E402
from chiro.workflows import run_named_agent  # noqa: E402

st.set_page_config(page_title="Clinic Copilot", page_icon=":material/health_and_safety:", layout="wide")


@st.cache_resource
def deps():
    settings = Settings.from_env()
    return settings, WarehouseSql(os.environ["DATABRICKS_WAREHOUSE_ID"]), get_llm_client()


settings, db, llm = deps()
if "session" not in st.session_state:
    st.session_state.session = uuid.uuid4().hex[:8]
    st.session_state.chat = []
tools = ClinicTools(db, settings, run_id=f"app-{st.session_state.session}")
care = CareReports(db, settings)
reviewer = st.context.headers.get("X-Forwarded-Email") or st.context.headers.get("X-Forwarded-User") or "unknown"


@st.cache_data(ttl=20, show_spinner=False)
def queue_counts() -> dict[str, int]:
    t = settings.table
    rows = db.query(f"""
        SELECT 'lead' AS q, count(*) AS n FROM {t('lead_actions')} WHERE status = 'pending_review'
        UNION ALL SELECT 'retention', count(*) FROM {t('retention_actions')} WHERE status = 'pending_review'
        UNION ALL SELECT 'price', count(*) FROM {t('price_recommendations')} WHERE status = 'pending_review'
        UNION ALL SELECT 'capacity', count(*) FROM {t('capacity_actions')} WHERE status = 'pending_review'
        UNION ALL SELECT 'care', count(*) FROM {t('care_reports')} WHERE status = 'pending_review'""")
    return {r["q"]: int(r["n"]) for r in rows}


@st.cache_data(ttl=600, show_spinner=False)
def metrics(view, measures, dims, days=None) -> pd.DataFrame:
    out = tools.query_metrics(view, list(measures), list(dims), last_n_days=days, limit=500)
    return pd.DataFrame(out.get("rows", []))


def review_queue(kind: str, title_fn, body_fn, order=None, badges_fn=None):
    """Pending drafts as cards: title + badges, the draft, and Approve / Reject."""
    items = tools.pending(kind)
    if order:
        items.sort(key=order)
    if not items:
        ui.empty("Nothing waiting for review.")
        return
    key = tools.id_column(kind)
    shown = st.session_state.get(f"show-{kind}", 8)
    for item in items[:shown]:
        with st.container(border=True):
            head, actions = st.columns([5, 2], vertical_alignment="center")
            with head:
                st.markdown(f"**{ui.esc(title_fn(item))}**")
                if badges_fn:
                    badges_fn(item)
            a, r = actions.columns(2)
            if a.button("Approve", key=f"a-{item[key]}", type="primary", icon=":material/check:", width="stretch"):
                tools.review(kind, item[key], "approved", reviewer)
                queue_counts.clear()
                st.rerun()
            if r.button("Reject", key=f"r-{item[key]}", icon=":material/close:", width="stretch"):
                tools.review(kind, item[key], "rejected", reviewer)
                queue_counts.clear()
                st.rerun()
            body_fn(item)
    if len(items) > shown and st.button(f"Show {len(items) - shown} more", key=f"more-{kind}",
                                        icon=":material/expand_more:"):
        st.session_state[f"show-{kind}"] = len(items)
        st.rerun()


def due_label(minutes: int) -> str:
    m = abs(int(minutes))
    span = f"{m // 60}h {m % 60:02d}m" if m >= 60 else f"{m} min"
    return f"overdue {span}" if minutes < 0 else f"in {span}"


# ---------------------------------------------------------------- pages
def overview():
    ui.header("Overview", "Agents draft, people decide. Everything below is live from Unity Catalog.")
    try:
        k = tools.get_kpis()
    except Exception as e:
        st.error(f"Could not load KPIs - has the setup job run and access been granted? {e}")
        st.stop()
    weekly = metrics("clinic_visit_metrics", ("revenue",), ("visit_week",), 182)
    weekly = weekly.sort_values("visit_week") if not weekly.empty else weekly
    leads = metrics("clinic_lead_metrics", ("leads",), ("created_week",), 182)
    leads = leads.sort_values("created_week") if not leads.empty else leads
    visits = metrics("clinic_visit_metrics", ("active_patients",), ("visit_week",), 182)
    visits = visits.sort_values("visit_week") if not visits.empty else visits
    full = slice(1, -1)  # drop the partial first and current weeks

    def trend(df, col):
        return df[col].tolist()[full] if not df.empty else None

    top = st.columns(3)
    ui.kpi(top[0], "Revenue · last 30 days", ui.money(k.get("revenue_30d")), help="Completed visits",
           trend=trend(weekly, "revenue"))
    ui.kpi(top[1], "Open leads", k.get("open_leads"), trend=trend(leads, "leads"),
           help=f"Awaiting a first response, last 30 days (trend: new leads per week). "
                f"{k.get('stale_open_leads')} older open leads are in nurture, not triage.")
    ui.kpi(top[2], "Active patients", f"{k.get('active_patients') or 0:,}", trend=trend(visits, "active_patients"),
           help="Completed a visit in the last 45 days (trend: patients seen per week)")
    bottom = st.columns(3)
    ui.kpi(bottom[0], "Lead conversion · 180 days", ui.pct(k.get("lead_conversion_rate_180d")))
    ui.kpi(bottom[1], "Patients at high dropout risk", f"{k.get('high_risk_patients') or 0:,}",
           help="Churn risk 50% or more")
    ui.kpi(bottom[2], "Revenue at risk per year", ui.money(k.get("annual_revenue_at_risk")),
           help="Sum of churn risk x annual value")

    st.markdown("#### Needs your attention")
    counts, helps = queue_counts(), care.help_requests(50)
    sla = {r["urgency"]: r for r in tools.get_sla_report(14)["by_tier"]}
    overdue = sum((sla.get(t) or {}).get("overdue_now") or 0 for t in ("emergency", "urgent"))
    cards = [("Lead drafts", counts.get("lead", 0), PAGES["leads"], f"{overdue} overdue" if overdue else None),
             ("Retention drafts", counts.get("retention", 0), PAGES["retention"], None),
             ("Price changes", counts.get("price", 0), PAGES["pricing"], None),
             ("Capacity actions", counts.get("capacity", 0), PAGES["capacity"], None),
             ("Care guidance", counts.get("care", 0), PAGES["care"],
              f"{len(helps)} need{'s' if len(helps) == 1 else ''} help" if helps else None)]
    for col, (label, n, page, alert) in zip(st.columns(len(cards)), cards):
        with col.container(border=True, height=178):
            st.markdown(f"<span style='font-size:1.6rem;font-weight:600'>{n}</span>", unsafe_allow_html=True)
            st.caption(label)
            if alert:
                ui.badge(alert, "orange", ":material/warning:")
            st.page_link(page, label="Review", icon=":material/arrow_forward:")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Weekly revenue")
        if not weekly.empty:
            ui.line(weekly.iloc[full], "visit_week", "revenue", y_title="revenue", y_format="$,.0f")
        st.markdown("#### Share of each cohort still visiting 3 months later")
        cohorts = pd.DataFrame(tools.get_cohort_retention(3, 18))
        if not cohorts.empty:
            ui.line(cohorts.sort_values("cohort_month"), "cohort_month", "retention_rate", y_title="still visiting",
                    y_format=".0%", zero=True)
    with right:
        st.markdown("#### Lead conversion by source (12 months)")
        src = metrics("clinic_lead_metrics", ("leads", "conversion_rate"), ("source",), 365)
        if not src.empty:
            src["source"] = src["source"].str.replace("_", " ")
            ui.hbar(src, "source", "conversion_rate", val_title="converted", val_format=".0%")
        st.markdown("#### What new leads want (AI-classified, 90 days)")
        intent = metrics("clinic_lead_metrics", ("leads",), ("ai_intent",), 90)
        if not intent.empty:
            intent = intent.dropna()
            intent["ai_intent"] = intent["ai_intent"].str.replace("_", " ")
            ui.hbar(intent, "ai_intent", "leads", val_title="leads")
    st.caption("Metrics come from the Unity Catalog metric views - the same governed definitions the agents, the "
               "AI/BI dashboard and Genie use.")


def page_chat():
    ui.header("Ask Copilot", "Questions about leads, patients, pricing, capacity or KPIs - answered with live data.")
    if not st.session_state.chat:
        st.markdown("Try one of these:")
        examples = ["Which lead sources convert best this quarter?", "Which location has the most unused capacity?",
                    "How many urgent leads missed their response target this week?"]
        for col, q in zip(st.columns(len(examples)), examples):
            if col.button(q, width="stretch"):
                st.session_state.pending_question = q
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])
    question = st.chat_input("Ask about leads, at-risk patients, pricing, or KPIs...") or \
        st.session_state.pop("pending_question", None)
    if question:
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"), st.spinner("Thinking with clinic data..."):
            result = run_agent(llm, settings.llm_endpoint, prompts.render(prompts.COPILOT, settings.clinic_name),
                               question, tools.copilot_registry(), history=st.session_state.chat[-10:])
            st.markdown(result.final_text)
            if result.tool_calls:
                st.caption("Tools used: " + ", ".join(c["tool"] for c in result.tool_calls))
        st.session_state.chat += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": result.final_text}]


def page_leads():
    ui.header("Leads", "Triage sets how fast to respond (never the price). Approving a draft marks it ready to send.")
    sla = tools.get_sla_report(14)
    st.markdown("#### Response times · last 14 days")
    if sla["by_tier"]:
        cols = st.columns(len(sla["by_tier"]))
        for c, r in zip(cols, sla["by_tier"]):
            rate = r["within_sla_rate"]
            target = sla["respond_within_hours"].get(r["urgency"])
            with c:
                ui.kpi(c, r["urgency"].capitalize(), "-" if rate is None else f"{rate:.0%} on time",
                       help=f"Target {target:g}h · {r['leads']} leads · median {r['median_hours_to_response']}h · "
                            f"p90 {r['p90_hours_to_response']}h")
                if r["overdue_now"]:
                    ui.badge(f"{r['overdue_now']} overdue now", "red", ":material/timer_off:")

    st.markdown("#### Next up")
    queue = tools.list_scored_leads(limit=15)
    if queue:
        table = pd.DataFrame([{
            "#": q["queue_position"], "lead": q["lead_id"],
            "urgency": f"{q['urgency']}",
            "due": due_label(q["minutes_to_deadline"]),
            "p_convert": q["p_convert"], "value": q["patient_value"], "why": q["urgency_reasons"]} for q in queue])
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "urgency": st.column_config.TextColumn(width="small"),
            "p_convert": st.column_config.ProgressColumn("P(convert)", format="percent", min_value=0, max_value=1,
                                                         width="small"),
            "value": st.column_config.NumberColumn("Patient value", format="dollar", width="small"),
            "why": st.column_config.TextColumn("Why", width="large")})
        if st.button("Draft responses for the top 5", icon=":material/auto_awesome:"):
            with st.spinner("Lead agent is working the queue..."):
                st.markdown(run_named_agent("lead", db, settings, llm, max_items=5).final_text)
            queue_counts.clear()
    else:
        ui.empty("No new leads waiting for a first response.")

    st.markdown("#### Drafts awaiting approval")
    pending_ids = [i["lead_id"] for i in tools.pending("lead")]
    triage = {r["lead_id"]: r for r in db.query(
        f"""SELECT s.lead_id, s.urgency, s.urgency_reasons, s.respond_within_hours,
                   timestampadd(MINUTE, CAST(s.respond_within_hours * 60 AS INT), r.created_at) AS respond_by
            FROM {settings.table('lead_scores')} s JOIN {settings.table('leads_raw')} r USING (lead_id)
            WHERE array_contains(split(:ids, ','), s.lead_id)""", {"ids": ",".join(pending_ids)})} if pending_ids else {}
    tier_rank = {"emergency": 0, "urgent": 1, "soon": 2, "routine": 3}

    def order(i):
        t = triage.get(i["lead_id"]) or {}
        return tier_rank.get(t.get("urgency"), 4), str(t.get("respond_by") or "9999")

    def badges(i):
        t = triage.get(i["lead_id"]) or {}
        ui.badges(ui.urgency_tag(t.get("urgency")),
                  ui.tag(i["channel"], "blue", {"phone": ":material/call:", "sms": ":material/sms:",
                                                "email": ":material/mail:"}.get(i["channel"])),
                  ui.tag(i["action_type"].replace("_", " "), "gray"))

    def body(i):
        if i["action_type"] == "refer_out":
            st.error("Red-flag symptoms: call this person now and advise prompt medical evaluation.",
                     icon=":material/emergency:")
        t = triage.get(i["lead_id"])
        if t and t.get("urgency"):
            st.caption(f"Respond by {str(t['respond_by'])[:16].replace('T', ' ')} UTC · {t['urgency_reasons']}")
        st.markdown(f"> {ui.esc(i['message'])}")
        if i.get("proposed_slot_id"):
            st.caption(f"Offers slot {i['proposed_slot_id']}")
        st.caption(f"Agent reasoning: {ui.esc(i['reasoning'])}")

    review_queue("lead", lambda i: i["lead_id"], body, order=order, badges_fn=badges)


def page_retention():
    ui.header("Retention", "Re-engagement drafts for patients at risk of dropping out of care, including invitations "
                           "from approved fill campaigns.")

    def badges(i):
        extra = (ui.tag(i["offer_code"], "violet", ":material/redeem:") if i.get("offer_code") else
                 ui.tag("fill campaign", "violet", ":material/event_available:") if i.get("campaign_action_id") else "")
        ui.badges(ui.tag(f"risk {i['churn_risk']:.0%}", "orange" if i["churn_risk"] >= 0.5 else "gray"),
                  ui.tag(i["channel"], "blue"), extra)

    def body(i):
        st.markdown(f"> {ui.esc(i['message'])}")
        if i.get("proposed_slot_id"):
            st.caption(f"Offers slot {i['proposed_slot_id']}")
        st.caption(f"Agent reasoning: {ui.esc(i['reasoning'])}")

    review_queue("retention", lambda i: i["patient_id"], body, badges_fn=badges)


def page_pricing():
    ui.header("Pricing", "Cash-pay prices only. Approved changes update the price list immediately.")
    overview_df = pd.DataFrame(tools.get_pricing_overview())
    if not overview_df.empty:
        cols = [c for c in ["service_id", "name", "current_price", "unit_cost", "weekly_cash_volume", "elasticity",
                            "elasticity_se", "competitor_low", "competitor_high"] if c in overview_df]
        st.dataframe(overview_df[cols], hide_index=True, width="stretch", column_config={
            "current_price": st.column_config.NumberColumn("Price", format="dollar"),
            "unit_cost": st.column_config.NumberColumn("Unit cost", format="dollar"),
            "competitor_low": st.column_config.NumberColumn("Competitors from", format="dollar"),
            "competitor_high": st.column_config.NumberColumn("to", format="dollar"),
            "service_id": st.column_config.TextColumn("Service", width="small"),
            "name": st.column_config.TextColumn("Name"),
            "weekly_cash_volume": st.column_config.NumberColumn("Cash visits / wk", format="%.0f"),
            "elasticity": st.column_config.NumberColumn("Elasticity", format="%.2f"),
            "elasticity_se": st.column_config.NumberColumn("± se", format="%.2f")})

    def badges(i):
        delta = i["projected_weekly_margin_delta"]
        ui.badges(ui.tag(f"margin {delta:+,.0f}/wk", "green" if delta > 0 else "red"),
                  ui.tag(f"volume {i['projected_weekly_volume_delta']:+.1f}/wk", "gray"))

    review_queue("price", lambda i: f"{i['service_id']}: ${i['current_price']:.0f} → ${i['proposed_price']:.0f}",
                 lambda i: st.markdown(ui.esc(i["rationale"])), badges_fn=badges)


def page_capacity():
    ui.header("Capacity", "Effective capacity = the smaller of room capacity and staffed provider slots. "
                          "Utilization is benchmarked against the network's own top quartile.")

    @st.cache_data(ttl=1800, show_spinner="Analyzing capacity...")
    def utilization():
        return tools.get_utilization()

    u = utilization()
    udf = pd.DataFrame(u["locations"])
    if udf.empty:
        ui.empty("No appointment history yet.", ":material/info:")
        return
    c = st.columns(4)
    ui.kpi(c[0], "Benchmark utilization", ui.pct(u["benchmark_utilization"]), help="Network top quartile")
    ui.kpi(c[1], "Weekly revenue gap", ui.money(udf["weekly_revenue_gap"].fillna(0).sum()),
           help="Revenue a week if every location reached the benchmark")
    ui.kpi(c[2], "Locations with a problem", int((udf["problems"].map(len) > 0).sum()), help="See the diagnosis column")
    ui.kpi(c[3], "Lapsed patients to win back", f"{int(udf['lapsed_patients'].sum()):,}")

    udf["diagnosis"] = udf["problems"].map(lambda p: "; ".join(p) or "-")
    st.dataframe(udf[["location_id", "city", "utilization", "staffed_slot_fill", "no_show_rate", "weekly_trend",
                      "weekly_revenue_gap", "diagnosis"]], hide_index=True, width="stretch", column_config={
        "location_id": st.column_config.TextColumn("Location", width="small"),
        "utilization": st.column_config.ProgressColumn("Utilization", format="percent", min_value=0, max_value=1),
        "staffed_slot_fill": st.column_config.ProgressColumn("Staffed slots filled", format="percent",
                                                             min_value=0, max_value=1),
        "no_show_rate": st.column_config.NumberColumn("No-shows", format="percent"),
        "weekly_trend": st.column_config.NumberColumn("Trend (visits/wk²)", format="%+.1f"),
        "weekly_revenue_gap": st.column_config.NumberColumn("Gap / week", format="dollar"),
        "diagnosis": st.column_config.TextColumn("Diagnosis", width="large")})

    names = udf.set_index("location_id")["location_name"]
    loc = st.selectbox("Location", udf["location_id"], format_func=lambda x: f"{x} · {names[x]}")
    detail = tools.get_location_detail(loc)
    if "error" not in detail:
        fc = pd.DataFrame(detail["forecast_next_14_days"])
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown(f"#### Next 14 days · about {detail['projected_idle_slots_14d']:.0f} slots projected unused")
            if not fc.empty:
                ui.forecast(fc)
            st.caption("Expected = booked now + bookings this location usually still receives that close to the "
                       "day. Gray ticks mark effective capacity; Sundays are closed.")
        with right:
            st.markdown("#### Completed visits by weekday")
            wd = pd.DataFrame(detail["by_weekday"])
            wd["day"] = wd["weekday"].str[:3]
            ui.vbar(wd, "day", "completed_per_day", val_title="per day", val_format=".0f",
                    order=list(wd.sort_values("dow")["day"]))
            st.markdown("#### No-show plan")
            plan = pd.DataFrame(detail["no_show_plan"])
            if not plan.empty:
                plan["plan"] = [f"overbook +{int(r.recommended_overbook)} (overflow risk {r.overflow_risk:.0%})"
                                if r.full else f"standby list of {int(r.recommended_standby_list)}"
                                for r in plan.itertuples()]
                st.dataframe(plan[["weekday", "expected_lost_slots", "plan"]], hide_index=True, width="stretch",
                             column_config={"expected_lost_slots": st.column_config.NumberColumn(
                                 "Lost slots/day", format="%.1f")})
        idle = fc[fc["projected_idle_slots"] > 0].sort_values("projected_idle_slots", ascending=False) \
            if not fc.empty else fc
        if not idle.empty:
            st.markdown("#### Fill list")
            day = st.selectbox("Invite lapsed patients into", idle["day"], format_func=lambda d: f"{d} · "
                               f"{idle.set_index('day').loc[d, 'projected_idle_slots']:.0f} idle slots")
            cands = tools.get_fill_candidates(loc, day, 25)
            st.caption(f"{len(cands['candidates'])} recoverable patients · at an assumed "
                       f"{cands['assumed_response_rate']:.0%} response rate ≈ {cands['expected_bookings']} bookings")
            st.dataframe(pd.DataFrame(cands["candidates"]), hide_index=True, width="stretch", column_config={
                "weekday_match": st.column_config.CheckboxColumn("Usual weekday"),
                "fit_score": st.column_config.ProgressColumn("Fit", min_value=0, max_value=4, format="%.2f")})
    if st.button("Run capacity review now", icon=":material/auto_awesome:"):
        with st.spinner("Capacity agent is reviewing locations..."):
            st.markdown(run_named_agent("capacity", db, settings, llm, max_items=3).final_text)
        queue_counts.clear()

    st.markdown("#### Actions awaiting approval")

    def badges(i):
        ui.badges(ui.tag(f"+{i['expected_weekly_visits']:.0f} visits/wk", "green"),
                  ui.tag(f"{ui.money(i['expected_weekly_revenue'])}/wk", "gray"))

    def body(i):
        if i.get("targets"):
            st.caption(f"Fill campaign for {i['target_date']}: {len(i['targets'].split(','))} patients. On approval "
                       "the Retention agent drafts each invitation with a slot that day.")
        st.markdown(ui.esc(i["details"]))
        st.caption(f"Agent reasoning: {ui.esc(i['reasoning'])}")

    review_queue("capacity", lambda i: f"{i['location_id']} · {ACTION_TYPES.get(i['action_type'], i['action_type'])}",
                 body, badges_fn=badges)

    impact = tools.get_capacity_impact()
    if impact:
        st.markdown("#### Approved actions · impact so far")
        st.dataframe(pd.DataFrame(impact), hide_index=True, width="stretch", column_config={
            "baseline_utilization": st.column_config.NumberColumn("Baseline", format="percent"),
            "utilization_since": st.column_config.NumberColumn("Since approval", format="percent"),
            "change": st.column_config.NumberColumn(format="%+.1%")})
        st.caption("Give an action 2-4 weeks before judging it.")


def page_care():
    ui.header("Care plans", "Practical advice and what tends to happen if it's ignored - not a risk assessment, not a "
                            "diagnosis. Every item cites its source; patients see it only after you approve it.")
    due, helps = care.due(200), care.help_requests(20)
    c = st.columns(3)
    ui.kpi(c[0], "Due a guidance update", len(due), help="Care notes changed since their last report")
    ui.kpi(c[1], "Drafts awaiting approval", queue_counts().get("care", 0))
    ui.kpi(c[2], "Patients asking for help", len(helps))
    if helps:
        with st.container(border=True):
            st.markdown("**:material/front_hand: Patients who said they need help**")
            st.dataframe(pd.DataFrame(helps)[["patient_id", "first_name", "advice", "comment", "at"]],
                         hide_index=True, width="stretch",
                         column_config={"advice": st.column_config.TextColumn(width="large"),
                                        "at": st.column_config.DatetimeColumn("When", format="MMM D, h:mm a")})
    if due and st.button(f"Draft guidance for {min(len(due), 5)} due patients", icon=":material/auto_awesome:"):
        with st.spinner("Drafting from care notes and attendance..."):
            st.dataframe(pd.DataFrame(care.draft_due(llm, settings.llm_endpoint, tools, limit=5)), hide_index=True)
        queue_counts.clear()

    st.markdown("#### Patient")
    pid = st.text_input("Patient ID", placeholder="e.g. PT0021447", label_visibility="collapsed").strip()
    if pid:
        profile = tools.get_patient_profile(pid)
        if "error" in profile:
            st.error(profile["error"])
        else:
            patient_panel(pid, profile)

    st.markdown("#### Drafts awaiting approval")

    def badges(i):
        q = json.loads(i["quality"]) if i.get("quality") else {}
        ui.badges(ui.tag(f"v{i.get('version') or 1}", "gray"),
                  ui.tag(f"reading grade {q.get('reading_grade', '-')}", "blue", ":material/menu_book:"),
                  ui.tag(f"grounding {q.get('mean_grounding', '-')}", "green", ":material/link:"))

    def body(i):
        left, right = st.columns(2, gap="large")
        with left:
            st.caption("WHAT THE PATIENT WILL SEE")
            st.markdown(ui.esc(i["patient_report"]))
        with right:
            st.caption("STAFF VERSION · ITEMS AND THEIR SOURCES")
            st.markdown(ui.esc(i["staff_report"]))

    review_queue("care", lambda i: f"{i['patient_id']} · care guidance", body, badges_fn=badges)


def patient_panel(pid: str, profile: dict):
    recs = tools.get_care_recommendations(pid)
    with st.container(border=True):
        st.markdown(f"### {profile['first_name']} · {pid}")
        risk = profile.get("churn_risk") or 0
        ui.badges(ui.tag(profile["insurance_type"], "gray"),
                  ui.tag(f"plan of {profile.get('care_plan_visits')} visits", "blue"),
                  ui.tag(f"dropout risk {risk:.0%}", "orange" if risk >= 0.5 else "gray"))
        if "error" not in recs:
            st.caption(f"Theme: {recs['theme']} — {recs['theme_description']}")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**Follow-through signals**")
            signals = care.signals(pid, profile.get("care_plan_visits"))
            if not signals:
                st.caption("No recent completed visits.")
            for sig in signals:
                color = {"high": "orange", "medium": "yellow", "low": "green"}[sig["importance"]]
                ui.badge(sig["name"].replace("_", " "), color)
                st.caption(sig["fact"])
        with right:
            notes = care.notes(pid)
            st.markdown(f"**Care notes** ({len(notes)})")
            for n in notes:
                st.markdown(f"**{ui.esc(n['advice'])}**  \nIf ignored: {ui.esc(n['if_ignored']) or '—'}")
                st.caption(f"{n['category'].replace('_', ' ')} · {n['importance']} · {n['author']}, "
                           f"{str(n['created_at'])[:10]}")
    with st.expander("Add a care note", icon=":material/edit_note:"):
        with st.form("note", clear_on_submit=True, border=False):
            cat = st.selectbox("Category", list(CARE_CATEGORIES), format_func=lambda k: k.replace("_", " "))
            advice = st.text_input("Advice", placeholder="e.g. Do the two hip stretches every morning, 30 seconds each")
            ignored = st.text_input("If ignored", placeholder="e.g. The stiffness tends to come back by the afternoon")
            imp = st.radio("Importance", ["high", "medium", "low"], index=1, horizontal=True)
            if st.form_submit_button("Save note", type="primary"):
                out = care.add_note(pid, reviewer, advice, ignored, cat, imp)
                if "error" in out:
                    st.error(out["error"])
                else:
                    st.rerun()
    a, b = st.columns([1, 3])
    if a.button("Draft guidance now", type="primary", icon=":material/auto_awesome:"):
        with st.spinner("Drafting from notes and attendance..."):
            out = care.draft(llm, settings.llm_endpoint, profile, None if "error" in recs else recs)
        if "error" in out:
            st.error(out["error"])
        else:
            queue_counts.clear()
            st.success(f"Version {out['version']} queued for review below.")
    history = care.history_of_reports(pid)
    if history:
        st.markdown("**Guidance history**")
        st.dataframe(pd.DataFrame(history).drop(columns=["quality"]), hide_index=True, width="stretch")
        answers = care.responses(pid)
        if answers:
            st.caption("Patient answers: " + ", ".join(
                v["response"].replace("_", " ") + (f" ('{v['comment']}')" if v["comment"] else "")
                for v in answers.values()))
    with st.expander("Link a patient-portal account", icon=":material/link:"):
        email = st.text_input("Portal account email")
        if st.button("Link account") and email:
            db.execute(f"UPDATE {settings.table('patient_accounts')} SET patient_id = :pid WHERE email = :email",
                       {"pid": pid, "email": email.strip().lower()})
            st.success("Linked - approved guidance will show in their portal.")


def page_briefing():
    ui.header("Briefing", "The Chief of Staff agent's morning summary, and every agent run.")
    rows = db.query(f"SELECT briefing_date, content FROM {settings.table('daily_briefings')} "
                    "ORDER BY created_at DESC LIMIT 1")
    if rows:
        with st.container(border=True):
            st.caption(f"BRIEFING · {rows[0]['briefing_date']}")
            st.markdown(ui.esc(rows[0]["content"]))
    else:
        ui.empty("No briefing yet - run the daily agents job.", ":material/info:")
    st.markdown("#### Recent agent runs")
    st.dataframe(pd.DataFrame(db.query(
        f"SELECT agent, status, started_at, steps, tool_calls, failed_tool_calls FROM {settings.table('agent_runs')} "
        "ORDER BY started_at DESC LIMIT 25")), hide_index=True, width="stretch", column_config={
        "started_at": st.column_config.DatetimeColumn("Started", format="MMM D, h:mm a")})


# ---------------------------------------------------------------- navigation
counts = queue_counts()


def titled(name: str, kind: str) -> str:
    return f"{name} · {counts[kind]}" if counts.get(kind) else name


PAGES = {
    "overview": st.Page(overview, title="Overview", icon=":material/dashboard:", default=True),
    "chat": st.Page(page_chat, title="Ask Copilot", icon=":material/forum:", url_path="ask"),
    "leads": st.Page(page_leads, title=titled("Leads", "lead"), icon=":material/person_add:", url_path="leads"),
    "retention": st.Page(page_retention, title=titled("Retention", "retention"), icon=":material/autorenew:",
                         url_path="retention"),
    "pricing": st.Page(page_pricing, title=titled("Pricing", "price"), icon=":material/sell:", url_path="pricing"),
    "capacity": st.Page(page_capacity, title=titled("Capacity", "capacity"), icon=":material/event_seat:",
                        url_path="capacity"),
    "care": st.Page(page_care, title=titled("Care plans", "care"), icon=":material/favorite:", url_path="care"),
    "briefing": st.Page(page_briefing, title="Briefing", icon=":material/newspaper:", url_path="briefing"),
}
nav = st.navigation({
    "": [PAGES["overview"], PAGES["chat"]],
    "Review": [PAGES[k] for k in ("leads", "retention", "pricing", "capacity", "care")],
    "Reports": [PAGES["briefing"]],
})
ui.logo(settings.clinic_name)
nav.run()
with st.sidebar:
    st.caption(f"Signed in as {reviewer}  \nData: `{settings.catalog}.{settings.schema}`")
