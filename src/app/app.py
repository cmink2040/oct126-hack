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

from chiro import prompts  # noqa: E402
from chiro.capacity import ACTION_TYPES  # noqa: E402
from chiro.care import CATEGORIES as CARE_CATEGORIES, CareReports  # noqa: E402
from chiro.agent import run_agent  # noqa: E402
from chiro.config import Settings  # noqa: E402
from chiro.llm import get_llm_client  # noqa: E402
from chiro.sql import WarehouseSql  # noqa: E402
from chiro.tools import ClinicTools  # noqa: E402

st.set_page_config(page_title="Clinic Copilot", layout="wide")


@st.cache_resource
def deps():
    settings = Settings.from_env()
    return settings, WarehouseSql(os.environ["DATABRICKS_WAREHOUSE_ID"]), get_llm_client()


settings, db, llm = deps()
if "session" not in st.session_state:
    st.session_state.session = uuid.uuid4().hex[:8]
    st.session_state.chat = []
tools = ClinicTools(db, settings, run_id=f"app-{st.session_state.session}")
reviewer = st.context.headers.get("X-Forwarded-Email") or st.context.headers.get("X-Forwarded-User") or "unknown"

st.title(f"🩺 {settings.clinic_name} · Clinic Copilot")
st.caption(f"Agents draft, people decide. Signed in as {reviewer}. Data: `{settings.catalog}.{settings.schema}` "
           f"(linked from `{settings.source_catalog}.{settings.source_schema}`).")

try:
    k = tools.get_kpis()
    cols = st.columns(6)
    cols[0].metric("Open leads (30d)", k.get("open_leads"), help=f"{k.get('stale_open_leads')} older open leads "
                   "are out of the triage queue (nurture instead).")
    cols[1].metric("Lead conversion (180d)", f"{(k.get('lead_conversion_rate_180d') or 0):.0%}")
    cols[2].metric("Active patients", k.get("active_patients"))
    cols[3].metric("High-risk patients", k.get("high_risk_patients"))
    cols[4].metric("Revenue (30d)", f"${(k.get('revenue_30d') or 0):,.0f}")
    cols[5].metric("Revenue at risk / yr", f"${(k.get('annual_revenue_at_risk') or 0):,.0f}")
except Exception as e:
    st.error(f"Could not load KPIs - has the setup job run and access been granted? {e}")
    st.stop()

tab_chat, tab_analytics, tab_leads, tab_ret, tab_price, tab_cap, tab_care, tab_brief = st.tabs(
    ["💬 Copilot", "📊 Analytics", "🧲 Lead queue", "🔁 Retention queue", "💲 Price approvals", "🏥 Capacity",
     "📝 Care plans", "📰 Briefing"])
URGENCY_ICON = {"emergency": "🚨", "urgent": "🔴", "soon": "🟠", "routine": "⚪"}

with tab_chat:
    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])
    if question := st.chat_input("Ask about leads, at-risk patients, pricing, or KPIs..."):
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"), st.spinner("Thinking with clinic data..."):
            result = run_agent(llm, settings.llm_endpoint, prompts.render(prompts.COPILOT, settings.clinic_name),
                               question, tools.copilot_registry(), history=st.session_state.chat[-10:])
            st.markdown(result.final_text)
            if result.tool_calls:
                st.caption("Tools used: " + ", ".join(c["tool"] for c in result.tool_calls))
        st.session_state.chat += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": result.final_text}]


with tab_analytics:
    st.caption("Served by Unity Catalog metric views - the same governed definitions the agents, "
               "the AI/BI dashboard and Genie use.")

    @st.cache_data(ttl=600, show_spinner=False)
    def metrics(view, measures, dims, days=None):
        out = tools.query_metrics(view, list(measures), list(dims), last_n_days=days, limit=500)
        return pd.DataFrame(out.get("rows", []))

    left, right = st.columns(2)
    with left:
        st.subheader("Monthly revenue & margin")
        df = metrics("clinic_visit_metrics", ("revenue", "contribution_margin"), ("visit_month",), 730)
        if not df.empty:
            st.line_chart(df.set_index("visit_month"))
        st.subheader("3-month retention by cohort")
        st.bar_chart(pd.DataFrame(tools.get_cohort_retention(3, 18)), x="cohort_month", y="retention_rate")
    with right:
        st.subheader("Lead conversion by source")
        df = metrics("clinic_lead_metrics", ("leads", "conversion_rate"), ("source",), 365)
        if not df.empty:
            st.bar_chart(df, x="source", y="conversion_rate")
        st.subheader("Lead intent (AI Functions, 90 days)")
        df = metrics("clinic_lead_metrics", ("leads",), ("ai_intent",), 90)
        if not df.empty:
            st.bar_chart(df, x="ai_intent", y="leads")


def review_queue(kind: str, title_fn, body_fn, order=None):
    items = tools.pending(kind)
    if order:
        items.sort(key=order)
    if not items:
        st.info("Nothing waiting for review.")
        return
    st.write(f"**{len(items)}** item(s) waiting.")
    key = "rec_id" if kind == "price" else "action_id"
    for item in items:
        with st.expander(title_fn(item), expanded=item.get("priority") == "high"):
            body_fn(item)
            a, r, _ = st.columns([1, 1, 6])
            if a.button("Approve", key=f"a-{item[key]}", type="primary"):
                tools.review(kind, item[key], "approved", reviewer)
                st.rerun()
            if r.button("Reject", key=f"r-{item[key]}"):
                tools.review(kind, item[key], "rejected", reviewer)
                st.rerun()


with tab_leads:
    sla = tools.get_sla_report(14)
    st.subheader("Response times (last 14 days)")
    st.caption("Responded = staff approved the first outreach. Targets: " + ", ".join(
        f"{t} {h:g}h" for t, h in sla["respond_within_hours"].items()))
    if sla["by_tier"]:
        cols = st.columns(len(sla["by_tier"]))
        for c, r in zip(cols, sla["by_tier"]):
            rate = r["within_sla_rate"]
            c.metric(f"{URGENCY_ICON.get(r['urgency'], '')} {r['urgency']}",
                     "-" if rate is None else f"{rate:.0%} on time",
                     help=f"{r['leads']} leads, median {r['median_hours_to_response']}h, p90 {r['p90_hours_to_response']}h")
            if r["overdue_now"]:
                c.caption(f"⏰ {r['overdue_now']} overdue now ({r['overdue_waiting_on_approval']} waiting on approval)")

    st.subheader("Next up")
    queue = tools.list_scored_leads(limit=15)
    if queue:
        st.dataframe(pd.DataFrame([{
            "#": q["queue_position"], "lead": q["lead_id"], "urgency": f"{URGENCY_ICON.get(q['urgency'], '')} {q['urgency']}",
            "deadline": ("overdue " if q["minutes_to_deadline"] < 0 else "in ") + f"{abs(q['minutes_to_deadline'])} min",
            "p(convert)": q["p_convert"], "value": q["patient_value"], "why": q["urgency_reasons"]} for q in queue]),
            hide_index=True, use_container_width=True)
        if st.button("Draft responses for the top of the queue"):
            from chiro.workflows import run_named_agent
            with st.spinner("Lead agent is working the queue..."):
                st.markdown(run_named_agent("lead", db, settings, llm, max_items=5).final_text)
    else:
        st.info("No new leads waiting for a first response.")

    st.subheader("Drafts awaiting approval")
    st.caption("Approving marks the draft ready to send (connect your SMS/email provider or front desk here).")
    pending_ids = [i["lead_id"] for i in tools.pending("lead")]
    triage = {r["lead_id"]: r for r in db.query(
        f"""SELECT s.lead_id, s.urgency, s.urgency_reasons, s.respond_within_hours,
                   timestampadd(MINUTE, CAST(s.respond_within_hours * 60 AS INT), r.created_at) AS respond_by
            FROM {settings.table('lead_scores')} s JOIN {settings.table('leads_raw')} r USING (lead_id)
            WHERE array_contains(split(:ids, ','), s.lead_id)""", {"ids": ",".join(pending_ids)})} if pending_ids else {}
    tier_rank = {"emergency": 0, "urgent": 1, "soon": 2, "routine": 3}

    def lead_order(i):
        t = triage.get(i["lead_id"]) or {}
        return tier_rank.get(t.get("urgency"), 4), str(t.get("respond_by") or "9999")

    def lead_body(i):
        if i["action_type"] == "refer_out":
            st.error("Red-flag symptoms: call this person now and advise prompt medical evaluation.")
        t = triage.get(i["lead_id"])
        if t and t.get("urgency"):
            st.markdown(f"{URGENCY_ICON.get(t['urgency'], '')} **{t['urgency']}** · respond by "
                        f"{str(t['respond_by'])[:16].replace('T', ' ')} UTC · {t['urgency_reasons']}")
        st.markdown(f"**{i['channel'].upper()}** · slot `{i.get('proposed_slot_id') or '-'}`")
        st.text(i["message"])
        st.caption(f"Agent reasoning: {i['reasoning']}")

    review_queue("lead", lambda i: f"{URGENCY_ICON.get((triage.get(i['lead_id']) or {}).get('urgency'), '')} "
                 f"{i['lead_id']} · {i['action_type']} · {i['priority']}", lead_body, order=lead_order)

with tab_ret:
    def ret_body(i):
        st.markdown(f"**{i['channel'].upper()}** · offer `{i.get('offer_code') or 'none'}` · "
                    f"churn risk {i['churn_risk']:.0%}")
        st.text(i["message"])
        st.caption(f"Agent reasoning: {i['reasoning']}")

    review_queue("retention", lambda i: f"{i['patient_id']} · risk {i['churn_risk']:.0%}", ret_body)

with tab_price:
    st.caption("Approved prices update `services` and `price_history` immediately.")
    st.dataframe(pd.DataFrame(tools.get_pricing_overview()), hide_index=True, use_container_width=True)

    def price_body(i):
        st.markdown(f"**${i['current_price']:.2f} → ${i['proposed_price']:.2f}** · "
                    f"weekly margin {i['projected_weekly_margin_delta']:+,.0f} · "
                    f"weekly volume {i['projected_weekly_volume_delta']:+.1f}")
        st.write(i["rationale"])

    review_queue("price", lambda i: f"{i['service_id']}: ${i['current_price']:.0f} → ${i['proposed_price']:.0f}",
                 price_body)

with tab_cap:
    st.caption("Effective capacity = min(room capacity, staffed provider slots). Utilization is benchmarked against "
               "the network's top quartile. Actions are drafted by the Capacity agent and need approval.")

    @st.cache_data(ttl=1800, show_spinner="Analyzing capacity...")
    def utilization():
        return tools.get_utilization()

    u = utilization()
    if u.get("benchmark_utilization") is not None:
        a, b = st.columns(2)
        a.metric("Benchmark utilization (network top quartile)", f"{u['benchmark_utilization']:.0%}")
        b.metric("Weekly revenue gap vs benchmark", f"${sum(r['weekly_revenue_gap'] or 0 for r in u['locations']):,.0f}")
    udf = pd.DataFrame(u["locations"])
    if not udf.empty:
        udf["problems"] = udf["problems"].map(lambda p: "; ".join(p) or "-")
        st.dataframe(udf[["location_id", "city", "utilization", "staffed_slot_fill", "room_utilization",
                          "no_show_rate", "weekly_trend", "weekly_revenue_gap", "problems"]],
                     hide_index=True, use_container_width=True)
        loc = st.selectbox("Location detail", udf["location_id"], format_func=lambda x: f"{x} · "
                           f"{udf.set_index('location_id').loc[x, 'location_name']}")
        detail = tools.get_location_detail(loc)
        if "error" not in detail:
            fc = pd.DataFrame(detail["forecast_next_14_days"])
            st.markdown(f"**Next 14 days:** about **{detail['projected_idle_slots_14d']:.0f}** slots projected to go "
                        f"unused (booked now + bookings still expected from the lead-time curve).")
            if not fc.empty:
                st.bar_chart(fc.set_index("day")[["booked_now", "expected_final_bookings", "effective_capacity"]],
                             stack=False)
            left, right = st.columns(2)
            left.markdown("**By weekday (trailing 12 weeks)**")
            left.dataframe(pd.DataFrame(detail["by_weekday"]).drop(columns=["dow"]), hide_index=True,
                           use_container_width=True)
            right.markdown("**No-show plan**")
            right.dataframe(pd.DataFrame(detail["no_show_plan"]), hide_index=True, use_container_width=True)
            if not fc.empty:
                idle_days = fc[fc["projected_idle_slots"] > 0].sort_values("projected_idle_slots", ascending=False)
                if not idle_days.empty:
                    day = st.selectbox("Fill list for", idle_days["day"], format_func=lambda d: f"{d} · "
                                       f"{idle_days.set_index('day').loc[d, 'projected_idle_slots']:.0f} idle slots")
                    cands = tools.get_fill_candidates(loc, day, 25)
                    st.caption(f"{len(cands['candidates'])} recoverable patients; at an assumed "
                               f"{cands['assumed_response_rate']:.0%} response rate ≈ {cands['expected_bookings']} bookings.")
                    st.dataframe(pd.DataFrame(cands["candidates"]), hide_index=True, use_container_width=True)
    if st.button("Run capacity review now"):
        from chiro.workflows import run_named_agent
        with st.spinner("Capacity agent is reviewing locations..."):
            res = run_named_agent("capacity", db, settings, llm, max_items=3)
        st.markdown(res.final_text)

    def cap_body(i):
        st.markdown(f"**{ACTION_TYPES.get(i['action_type'], i['action_type'])}** · "
                    f"+{i['expected_weekly_visits']:.0f} visits/wk · ${i['expected_weekly_revenue']:,.0f}/wk")
        if i.get("targets"):
            st.caption(f"Fill campaign for {i['target_date']}: {len(i['targets'].split(','))} patients - on approval "
                       "the Retention agent drafts each invitation with a slot that day.")
        st.write(i["details"])
        st.caption(f"Agent reasoning: {i['reasoning']}")

    review_queue("capacity", lambda i: f"{i['location_id']} · {i['action_type']}", cap_body)

    impact = tools.get_capacity_impact()
    if impact:
        st.subheader("Approved actions: impact so far")
        st.dataframe(pd.DataFrame(impact), hide_index=True, use_container_width=True)
        st.caption("Measured as utilization since approval vs the location's utilization when it was approved; "
                   "give an action 2-4 weeks before judging it.")

with tab_care:
    st.caption("Care guidance = practical advice and what tends to happen if it's ignored - not a risk assessment, "
               "not a diagnosis. Drafted from your care notes and the patient's attendance, every item cites its "
               "source, and patients see it in their portal only after you approve it.")
    care = CareReports(db, settings)
    due, helps = care.due(50), care.help_requests(20)
    a, b, c = st.columns(3)
    a.metric("Patients due a guidance update", len(due), help="Care notes changed since their last report")
    b.metric("Drafts awaiting approval", len(tools.pending("care")))
    c.metric("Patients asking for help", len(helps))
    if helps:
        with st.expander(f"🙋 {len(helps)} patient(s) said they need help with an item", expanded=True):
            st.dataframe(pd.DataFrame(helps)[["patient_id", "first_name", "advice", "comment", "at"]], hide_index=True,
                         use_container_width=True)
    if due and st.button(f"Draft guidance for {min(len(due), 5)} due patient(s)"):
        with st.spinner("Drafting from care notes and attendance..."):
            results = care.draft_due(llm, settings.llm_endpoint, tools, limit=5)
        st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)

    pid = st.text_input("Patient ID", placeholder="e.g. PT0009863").strip()
    if pid:
        profile = tools.get_patient_profile(pid)
        if "error" in profile:
            st.error(profile["error"])
        else:
            recs = tools.get_care_recommendations(pid)
            st.markdown(f"**{profile['first_name']}** · {profile['insurance_type']} · plan of "
                        f"{profile.get('care_plan_visits')} visits · dropout risk {(profile.get('churn_risk') or 0):.0%}")
            if "error" not in recs:
                st.caption(f"Theme: {recs['theme']} — {recs['theme_description']}")
            signals = care.signals(pid, profile.get("care_plan_visits"))
            if signals:
                st.markdown("**Follow-through signals** (from attendance)")
                for sig in signals:
                    st.markdown(f"- **{sig['name'].replace('_', ' ')}** ({sig['importance']}): {sig['fact']}")
            notes = care.notes(pid)
            st.markdown(f"**Care notes** ({len(notes)})")
            for n in notes:
                st.markdown(f"- _{n['category'].replace('_', ' ')} · {n['importance']}_ — **{n['advice']}**  \n"
                            f"  If ignored: {n['if_ignored'] or '—'}  \n  <small>{n['author']}, "
                            f"{str(n['created_at'])[:10]}</small>", unsafe_allow_html=True)
            with st.form("note", clear_on_submit=True):
                st.markdown("**Add a care note** — one piece of advice and what tends to happen if it's ignored")
                cat = st.selectbox("Category", list(CARE_CATEGORIES), format_func=lambda k: k.replace("_", " "))
                advice = st.text_input("Advice", placeholder="e.g. Do the two hip stretches every morning, 30 seconds each")
                ignored = st.text_input("If ignored", placeholder="e.g. The stiffness tends to come back by the "
                                        "afternoon, and progress between visits slows")
                imp = st.radio("Importance", ["high", "medium", "low"], index=1, horizontal=True)
                if st.form_submit_button("Save note"):
                    out = care.add_note(pid, reviewer, advice, ignored, cat, imp)
                    if "error" in out:
                        st.error(out["error"])
                    else:
                        st.rerun()
            if st.button("Draft guidance now", type="primary"):
                with st.spinner("Drafting from notes and attendance..."):
                    out = care.draft(llm, settings.llm_endpoint, profile, None if "error" in recs else recs)
                if "error" in out:
                    st.error(out["error"])
                else:
                    st.success(f"Version {out['version']} queued for review below (reading grade "
                               f"{out['quality']['reading_grade']}).")
            history = care.history_of_reports(pid)
            if history:
                st.markdown("**Guidance history**")
                st.dataframe(pd.DataFrame(history).drop(columns=["quality"]), hide_index=True, use_container_width=True)
                answers = care.responses(pid)
                if answers:
                    st.caption("Patient answers: " + ", ".join(f"{v['response']}" + (f" ('{v['comment']}')"
                                                                                       if v["comment"] else "")
                                                               for v in answers.values()))
            with st.expander("Link a patient-portal account to this patient"):
                email = st.text_input("Portal account email")
                if st.button("Link account") and email:
                    db.execute(f"UPDATE {settings.table('patient_accounts')} SET patient_id = :pid "
                               "WHERE email = :email", {"pid": pid, "email": email.strip().lower()})
                    st.success("Linked - approved guidance will show in their portal.")

    def care_body(i):
        q = json.loads(i["quality"]) if i.get("quality") else {}
        if q:
            st.caption(f"Version {i.get('version')} · reading grade {q.get('reading_grade')} · grounding "
                       f"{q.get('mean_grounding')} (min {q.get('min_grounding')}) · drafted in {q.get('attempts')} "
                       "attempt(s). Approving replaces the patient's current guidance.")
        left, right = st.columns(2)
        left.markdown("**Staff version (items with their sources)**")
        left.markdown(i["staff_report"])
        right.markdown("**What the patient will see**")
        right.markdown(i["patient_report"])

    review_queue("care", lambda i: f"{i['patient_id']} · care guidance v{i.get('version') or 1}", care_body)

with tab_brief:
    rows = db.query(f"SELECT briefing_date, content FROM {settings.table('daily_briefings')} "
                    "ORDER BY created_at DESC LIMIT 1")
    if rows:
        st.subheader(f"Briefing for {rows[0]['briefing_date']}")
        st.markdown(rows[0]["content"])
    else:
        st.info("No briefing yet - run the daily agents job.")
    st.subheader("Recent agent runs")
    st.dataframe(pd.DataFrame(db.query(
        f"SELECT agent, status, started_at, steps, tool_calls, failed_tool_calls FROM {settings.table('agent_runs')} "
        "ORDER BY started_at DESC LIMIT 20")), hide_index=True, use_container_width=True)
