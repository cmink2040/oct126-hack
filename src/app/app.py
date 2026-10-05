"""Clinic Copilot - Databricks App.

Chat with the Copilot agent, and approve / reject what the scheduled agents queued.
Runs as the app's service principal; reads/writes Unity Catalog through a SQL warehouse
and reasons with a Foundation Model API endpoint (both declared as app resources in the bundle).
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from chiro import prompts  # noqa: E402
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
    cols[0].metric("Open leads", k.get("open_leads"))
    cols[1].metric("Lead conversion (180d)", f"{(k.get('lead_conversion_rate_180d') or 0):.0%}")
    cols[2].metric("Active patients", k.get("active_patients"))
    cols[3].metric("High-risk patients", k.get("high_risk_patients"))
    cols[4].metric("Revenue (30d)", f"${(k.get('revenue_30d') or 0):,.0f}")
    cols[5].metric("Revenue at risk / yr", f"${(k.get('annual_revenue_at_risk') or 0):,.0f}")
except Exception as e:
    st.error(f"Could not load KPIs - has the setup job run and access been granted? {e}")
    st.stop()

tab_chat, tab_analytics, tab_leads, tab_ret, tab_price, tab_brief = st.tabs(
    ["💬 Copilot", "📊 Analytics", "🧲 Lead queue", "🔁 Retention queue", "💲 Price approvals", "📰 Briefing"])

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


def review_queue(kind: str, title_fn, body_fn):
    items = tools.pending(kind)
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
    st.caption("Approving marks the draft ready to send (connect your SMS/email provider or front desk here).")

    def lead_body(i):
        if i["action_type"] == "refer_out":
            st.error("Red-flag symptoms: call this person now and advise prompt medical evaluation.")
        st.markdown(f"**{i['channel'].upper()}** · slot `{i.get('proposed_slot_id') or '-'}`")
        st.text(i["message"])
        st.caption(f"Agent reasoning: {i['reasoning']}")

    review_queue("lead", lambda i: f"{'🚨 ' if i['action_type'] == 'refer_out' else ''}{i['lead_id']} · "
                 f"{i['action_type']} · {i['priority']}", lead_body)

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
