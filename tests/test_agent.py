"""Agent loop + tool guardrails, with a scripted fake LLM and a stub SQL backend."""
import json
from types import SimpleNamespace as NS

from chiro.agent import Tool, ToolRegistry, params, run_agent
from chiro.config import Settings
from chiro.tools import ClinicTools


class FakeLLM:
    """Replays a script of responses: each item is a list of (tool, args) calls, or a final string."""

    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.seen.append(kw["messages"])
        step = self.script.pop(0)
        if isinstance(step, str):
            msg = NS(content=step, tool_calls=None)
        else:
            msg = NS(content=None, tool_calls=[
                NS(id=f"c{i}", function=NS(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(step)])
        return NS(choices=[NS(message=msg)])


def test_loop_executes_tools_and_feeds_results_back():
    reg = ToolRegistry([Tool("add", "Add", params(["a", "b"], a=("number", ""), b=("number", "")),
                             lambda a, b: {"sum": a + b})])
    llm = FakeLLM([[("add", {"a": 2, "b": 3})], "The sum is 5."])
    res = run_agent(llm, "m", "sys", "add 2+3", reg)
    assert res.status == "completed" and res.final_text == "The sum is 5."
    tool_msgs = [m for m in res.messages if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"]) == {"sum": 5}


def test_tool_errors_are_returned_to_the_model_not_raised():
    reg = ToolRegistry([Tool("boom", "x", params(), lambda: 1 / 0)])
    llm = FakeLLM([[("boom", {}), ("nope", {})], "recovered"])
    res = run_agent(llm, "m", "sys", "go", reg)
    assert res.status == "completed" and [c["ok"] for c in res.tool_calls] == [False, False]


def test_step_limit():
    reg = ToolRegistry([Tool("noop", "x", params(), lambda: {})])
    res = run_agent(FakeLLM([[("noop", {})]] * 3), "m", "sys", "go", reg, max_steps=3)
    assert res.status == "max_steps"


class StubDB:
    """Routes queries by table name; records writes."""

    def __init__(self, lead=None, patient=None):
        self.lead, self.patient, self.writes = lead, patient, []

    def query(self, sql, params=None):
        if "lead_actions" in sql and "status = 'pending_review'" in sql:
            return []
        if "`leads`" in sql:
            return [dict(self.lead)] if self.lead else []
        if "retention_offers" in sql:
            return [{"offer_code": "REBOOK_CALL", "eligible_payers": "cash,ppo,medicare"},
                    {"offer_code": "MEMBER20", "eligible_payers": "cash"}]
        if "SELECT insurance_type" in sql:
            return [{"insurance_type": self.patient["insurance_type"]}]
        if "`visits`" in sql:
            return []
        if "`patients`" in sql:
            return [dict(self.patient)]
        return []

    def execute(self, sql, params=None):
        self.writes.append((sql, params))


LEAD = dict(lead_id="L1", message="Hi, I've had lower back stiffness for two weeks.", consent_to_contact=True,
            consent_sms=False, consent_email=True, status="new", score=0.4, reasons="")
GOOD_MSG = "Hi Sam, thanks for reaching out about your back. We have Tue 9:00 or 9:30 open - want one?"


def tools_for(**kw):
    return ClinicTools(StubDB(**kw), Settings(), run_id="test")


def test_lead_consent_enforced():
    out = tools_for(lead=LEAD).queue_lead_action("L1", "outreach", "sms", GOOD_MSG, "high score")
    assert "no consent for sms" in out["error"]
    out = tools_for(lead=LEAD).queue_lead_action("L1", "outreach", "email", GOOD_MSG, "high score")
    assert out["ok"]


def test_red_flag_lead_must_be_referred_out_by_phone():
    lead = {**LEAD, "message": "Back pain and trouble controlling my bladder since yesterday"}
    t = tools_for(lead=lead)
    assert "red-flag" in t.queue_lead_action("L1", "outreach", "email", GOOD_MSG, "x")["error"]
    assert "phone" in t.queue_lead_action("L1", "refer_out", "email", GOOD_MSG, "x")["error"]
    ok = t.queue_lead_action("L1", "refer_out", "phone",
                             "Please call 911 or get to urgent care now; these symptoms need prompt evaluation.", "x")
    assert ok["ok"]
    sql, p = t.db.writes[-1]
    assert p["priority"] == "high" and p["slot"] == ""


def test_medicare_patients_cannot_get_monetary_offers():
    patient = dict(patient_id="P1", first_name="Pat", insurance_type="medicare", consent_sms=True,
                   consent_email=True, churn_risk=0.8)
    t = tools_for(patient=patient)
    out = t.queue_retention_action("P1", "email", GOOD_MSG, "MEMBER20", "finished plan")
    assert "not eligible" in out["error"]
    assert t.queue_retention_action("P1", "email", GOOD_MSG, "REBOOK_CALL", "finished plan")["ok"]


def test_review_is_not_exposed_to_agents():
    t = tools_for(lead=LEAD)
    assert "review" not in t.copilot_registry().names
    assert len(t.copilot_registry().names) == len(set(t.copilot_registry().names)) >= 12


def test_semantic_query_rejects_unknown_fields_and_parameterises_values():
    import pytest

    from chiro import semantic

    s = Settings()
    with pytest.raises(semantic.MetricQueryError):
        semantic.build_metric_query(s, "clinic_visit_metrics", ["revenue; DROP TABLE x"])
    with pytest.raises(semantic.MetricQueryError):
        semantic.build_metric_query(s, "clinic_visit_metrics", ["revenue"],
                                    filters=[{"dimension": "1=1 OR payer", "value": "x"}])
    sql, p = semantic.build_metric_query(s, "clinic_visit_metrics", ["revenue"], ["payer"],
                                         filters=[{"dimension": "payer", "op": "=", "value": "cash' OR '1'='1"}])
    assert "MEASURE(`revenue`)" in sql and "cash" not in sql and p["f0"].startswith("cash'")


def test_metric_view_yaml_matches_allowlist():
    from chiro import semantic

    for mv in semantic.METRIC_VIEWS.values():
        y = mv.yaml(Settings())
        assert all(f"name: {m}" in y for m in mv.measures) and f"source: workspace.chiro.{mv.source}" in y


def test_agents_get_analytics_tools():
    t = tools_for(lead=LEAD)
    for reg in (t.pricing_registry(), t.briefing_registry(), t.copilot_registry()):
        assert {"query_metrics", "forecast_demand", "get_cohort_retention"} <= set(reg.names)
