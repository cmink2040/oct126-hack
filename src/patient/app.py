"""Patient portal - create an account and send the clinic an inquiry.

Each inquiry becomes a lead in `leads_raw`, is scored immediately, and the Lead agent drafts a
response in the background; staff approve it in Clinic Copilot before anything is sent.
Databricks Apps require a workspace login, so a public deployment of this portal needs hosting
outside Databricks; locally it runs as your Databricks CLI profile.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from chiro.care import RESPONSES, CareReports  # noqa: E402
from chiro.config import Settings  # noqa: E402
from chiro.intake import COMPLAINTS, INSURANCE, Intake, IntakeError, process_inquiry  # noqa: E402
from chiro.llm import get_llm_client  # noqa: E402
from chiro.sql import WarehouseSql  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="Patient Portal", page_icon=":material/health_and_safety:", layout="centered")

# Plain-language status of an inquiry, and the badge color it wears.
STATUS = {None: ("Received", "gray"), "pending_review": ("Our team is reviewing it", "blue"),
          "approved": ("We'll be in touch shortly", "green"), "rejected": ("Our team is reviewing it", "blue")}


@st.cache_resource
def deps():
    settings = Settings.from_env()
    db = WarehouseSql(os.environ["DATABRICKS_WAREHOUSE_ID"])
    return settings, db, get_llm_client(), Intake(db, settings)


@st.cache_resource(show_spinner=False)
def lead_model():
    return deps()[3].train_lead_model()


@st.cache_data(ttl=3600)
def locations() -> dict[str, str]:
    settings, db, _, _ = deps()
    rows = db.query(f"SELECT location_id, location_name, city FROM {settings.source_table('locations')} "
                    "ORDER BY location_name")
    return {r["location_id"]: f"{r['location_name']} ({r['city']})" for r in rows}


def process_in_background(lead_id: str) -> None:
    settings, db, llm, _ = deps()

    def work():
        try:
            process_inquiry(lead_id, db, settings, llm)
        except Exception as e:  # the daily lead agent will still pick the lead up
            print(f"Lead agent failed for {lead_id}: {e}")

    threading.Thread(target=work, daemon=True).start()


settings, db, llm, intake = deps()
ui.logo(settings.clinic_name)
account = st.session_state.get("account")

if account is None:
    st.markdown(f"## Welcome to {settings.clinic_name}")
    st.caption("Ask a question, request a first visit, and see the care guidance your team shares with you.")
    tab_login, tab_new = st.tabs(["Log in", "Create account"])
    with tab_login, st.form("login", border=False):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary", width="stretch"):
            found = intake.log_in(email, password)
            if found:
                st.session_state.account = found
                st.rerun()
            st.error("Email or password is incorrect.")
    with tab_new, st.form("signup", border=False):
        a, b = st.columns(2)
        first_name = a.text_input("First name")
        phone = b.text_input("Phone (optional)")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password", help="At least 8 characters.")
        contact = st.checkbox("The clinic may contact me about my inquiry", value=True)
        c1, c2 = st.columns(2)
        sms = c1.checkbox("By text message")
        mail = c2.checkbox("By email", value=True)
        if st.form_submit_button("Create account", type="primary", width="stretch"):
            try:
                st.session_state.account = intake.create_account(email, password, first_name, phone, contact, sms, mail)
                st.rerun()
            except IntakeError as e:
                st.error(str(e))
    st.stop()

head, out = st.columns([5, 1], vertical_alignment="center")
head.markdown(f"## Hi {account['first_name']}")
if out.button("Log out", type="tertiary", icon=":material/logout:"):
    del st.session_state["account"]
    st.rerun()
if not account["consent_to_contact"]:
    st.warning("You haven't allowed the clinic to contact you, so we can't reply to inquiries.",
               icon=":material/notifications_off:")

if account.get("patient_id"):
    care = CareReports(db, settings)
    report = care.latest_approved(account["patient_id"])
    if report:
        care.mark_viewed(report["report_id"])
        answers = care.responses(account["patient_id"])
        st.markdown("### Your care plan")
        st.caption(f"What to keep doing, and what tends to happen if it slips. From your care team, "
                   f"{str(report['reviewed_at'])[:10]}. This is guidance on following your plan, not a medical "
                   "assessment - ask us at your next visit if anything is unclear.")
        if not report["items"]:  # guidance approved before item-level answers existed
            st.markdown(ui.esc(report["patient_report"]))
        else:
            st.markdown(ui.esc(report["items"]["intro"]))
            for n, item in enumerate(report["items"]["items"], 1):
                with st.container(border=True):
                    st.markdown(f"**{n}. {ui.esc(item['advice'])}**")
                    st.markdown(f":gray[If this slips:] {ui.esc(item['if_ignored'])}")
                    if item.get("why_now"):
                        st.caption(item["why_now"])
                    answer = answers.get(item["key"])
                    choice = st.pills("How is this going?", list(RESPONSES), format_func=RESPONSES.get,
                                      key=f"{item['key']}-pill", label_visibility="collapsed",
                                      default=answer["response"] if answer else None)
                    if answer and choice == answer["response"]:
                        st.caption("Thanks - your care team can see your answer"
                                   + (f": \"{answer['comment']}\"" if answer["comment"] else "."))
                    elif choice:
                        note = st.text_input("Anything we should know? (optional)", key=f"{item['key']}-note")
                        if st.button("Send", key=f"{item['key']}-send", type="primary", icon=":material/send:"):
                            care.respond(report["report_id"], account["patient_id"], item["key"], choice, note)
                            st.rerun()

st.markdown("### Ask us something")
with st.form("inquiry", clear_on_submit=True):
    a, b = st.columns(2)
    complaint = a.selectbox("What would you like help with?", list(COMPLAINTS), format_func=COMPLAINTS.get)
    insurance = b.selectbox("How do you plan to pay?", list(INSURANCE), format_func=INSURANCE.get)
    locs = locations()
    c, d = st.columns([3, 1])
    location = c.selectbox("Preferred location", list(locs), format_func=locs.get)
    distance = d.number_input("Miles away", 0.0, 200.0, 5.0, step=0.5)
    message = st.text_area("Tell us what's going on", placeholder="How long it's been bothering you, "
                           "what makes it better or worse, and anything you'd like to ask.")
    if st.form_submit_button("Send to the team", type="primary", icon=":material/send:"):
        try:
            with st.spinner("Sending..."):
                result = intake.submit_inquiry(account, complaint, insurance, distance, location, message,
                                               lead_model(), llm.with_options(timeout=30, max_retries=0),
                                               settings.llm_endpoint)
            if account["consent_to_contact"]:
                process_in_background(result["lead_id"])
            if result["red_flags"]:
                st.error("Some of what you describe can need prompt medical attention. If this is an emergency, "
                         "call 911 or go to the nearest ER. Otherwise please see urgent care or your doctor soon. "
                         "Our team will also call you.", icon=":material/emergency:")
            elif result["urgency"] == "urgent":
                st.success("Thanks, we've got your message. It sounds like this is really affecting you, so a member "
                           "of our team will reach out within the hour during clinic hours.")
            else:
                st.success("Thanks, we've got your message. A member of our team will reach out soon.")
        except IntakeError as e:
            st.error(str(e))

rows = intake.my_inquiries(account["account_id"])
if rows:
    st.markdown("### Your messages")
    for r in rows:
        with st.container(border=True):
            label, color = STATUS.get(r["review_status"], STATUS[None])
            top, right = st.columns([3, 2], vertical_alignment="center")
            top.caption(f"{str(r['submitted_at'])[:10]} · {COMPLAINTS.get(r['complaint'], r['complaint'])}")
            with right:
                ui.badge(label, color)
            st.markdown(ui.esc(r["message"]))
