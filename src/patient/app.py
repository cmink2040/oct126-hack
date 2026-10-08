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

from chiro.care import CareReports  # noqa: E402
from chiro.config import Settings  # noqa: E402
from chiro.intake import COMPLAINTS, INSURANCE, Intake, IntakeError, process_inquiry  # noqa: E402
from chiro.llm import get_llm_client  # noqa: E402
from chiro.sql import WarehouseSql  # noqa: E402

st.set_page_config(page_title="Patient Portal", layout="centered")

STATUS = {None: "Received", "pending_review": "Our team is reviewing it",
          "approved": "Our team will be in touch shortly", "rejected": "Our team is reviewing it"}


@st.cache_resource
def deps():
    settings = Settings.from_env()
    db = WarehouseSql(os.environ["DATABRICKS_WAREHOUSE_ID"])
    return settings, db, get_llm_client(), Intake(db, settings)


@st.cache_resource(show_spinner="Getting things ready...")
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
account = st.session_state.get("account")
st.title(settings.clinic_name)

if account is None:
    st.write("Create an account or log in to ask a question or request a first visit.")
    tab_new, tab_login = st.tabs(["Create account", "Log in"])
    with tab_new, st.form("signup"):
        first_name = st.text_input("First name")
        email = st.text_input("Email")
        phone = st.text_input("Phone (optional)")
        password = st.text_input("Password", type="password", help="At least 8 characters.")
        contact = st.checkbox("The clinic may contact me about my inquiry", value=True)
        sms = st.checkbox("By text message")
        mail = st.checkbox("By email", value=True)
        if st.form_submit_button("Create account", type="primary"):
            try:
                st.session_state.account = intake.create_account(email, password, first_name, phone, contact, sms, mail)
                st.rerun()
            except IntakeError as e:
                st.error(str(e))
    with tab_login, st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            found = intake.log_in(email, password)
            if found:
                st.session_state.account = found
                st.rerun()
            st.error("Email or password is incorrect.")
    st.stop()

head, out = st.columns([5, 1])
head.subheader(f"Hi {account['first_name']}")
if out.button("Log out"):
    del st.session_state["account"]
    st.rerun()

if not account["consent_to_contact"]:
    st.warning("You haven't allowed the clinic to contact you, so we can't reply to inquiries.")

model = lead_model()
with st.form("inquiry", clear_on_submit=True):
    st.markdown("**How can we help?**")
    complaint = st.selectbox("What would you like help with?", list(COMPLAINTS), format_func=COMPLAINTS.get)
    insurance = st.selectbox("How do you plan to pay?", list(INSURANCE), format_func=INSURANCE.get)
    locs = locations()
    location = st.selectbox("Preferred location", list(locs), format_func=locs.get)
    distance = st.number_input("About how far do you live from it (miles)?", 0.0, 200.0, 5.0, step=0.5)
    message = st.text_area("Tell us what's going on", placeholder="How long it's been bothering you, "
                           "what makes it better or worse, and anything you'd like to ask.")
    if st.form_submit_button("Send", type="primary"):
        try:
            with st.spinner("Sending..."):
                result = intake.submit_inquiry(account, complaint, insurance, distance, location, message, model,
                                               llm.with_options(timeout=30, max_retries=0), settings.llm_endpoint)
            if account["consent_to_contact"]:
                process_in_background(result["lead_id"])
            if result["red_flags"]:
                st.error("Some of what you describe can need prompt medical attention. If this is an emergency, "
                         "call 911 or go to the nearest ER. Otherwise please see urgent care or your doctor soon. "
                         "Our team will also call you.")
            elif result["urgency"] == "urgent":
                st.success("Thanks, we've got your message. It sounds like this is really affecting you, so a member "
                           "of our team will reach out within the hour during clinic hours.")
            else:
                st.success("Thanks, we've got your message. A member of our team will reach out soon.")
        except IntakeError as e:
            st.error(str(e))

if account.get("patient_id"):
    report = CareReports(db, settings).latest_approved(account["patient_id"])
    if report:
        with st.container(border=True):
            st.markdown("**Your care plan: what to keep doing**")
            st.markdown(report["patient_report"])
            st.caption(f"From your care team, {str(report['reviewed_at'])[:10]}. This is guidance on following "
                       "your plan, not a medical assessment - ask us at your next visit if anything is unclear.")

st.markdown("**Your inquiries**")
rows = intake.my_inquiries(account["account_id"])
if not rows:
    st.caption("Nothing yet.")
for r in rows:
    with st.container(border=True):
        st.caption(f"{str(r['submitted_at'])[:16].replace('T', ' ')} · {COMPLAINTS.get(r['complaint'], r['complaint'])}"
                   f" · {STATUS.get(r['review_status'], 'Received')}")
        st.write(r["message"])
