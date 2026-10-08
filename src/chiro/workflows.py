"""Entry points used by the Databricks job notebooks."""
from __future__ import annotations

import datetime as dt
import uuid

from chiro import prompts
from chiro.agent import AgentResult, run_agent
from chiro.config import Settings
from chiro.sql import SqlRunner
from chiro.tools import ClinicTools

AGENTS = {
    # name: (prompt template, registry factory, task instruction)
    "lead": (prompts.LEAD_AGENT, ClinicTools.lead_registry,
             "Process today's new leads now."),
    "retention": (prompts.RETENTION_AGENT, ClinicTools.retention_registry,
                  "Work today's at-risk patient list now."),
    "pricing": (prompts.PRICING_AGENT, ClinicTools.pricing_registry,
                "Run this week's cash-price review now."),
    "capacity": (prompts.CAPACITY_AGENT, ClinicTools.capacity_registry,
                 "Run this week's capacity review now."),
    "briefing": (prompts.BRIEFING_AGENT, ClinicTools.briefing_registry,
                 "Write today's briefing."),
}


def new_run_id(agent: str) -> str:
    return f"{agent}-{dt.datetime.utcnow():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def run_named_agent(agent: str, db: SqlRunner, settings: Settings, client, max_items: int = 10,
                    max_steps: int = 40, instruction: str | None = None) -> AgentResult:
    template, registry_factory, default_instruction = AGENTS[agent]
    tools = ClinicTools(db, settings, new_run_id(agent))
    started = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        result = run_agent(
            client, settings.llm_endpoint,
            system=prompts.render(template, settings.clinic_name, max_items),
            user=instruction or default_instruction, registry=registry_factory(tools), max_steps=max_steps,
        )
    except Exception as e:
        result = AgentResult(f"{type(e).__name__}: {e}", "error", 0)
    tools.log_run(agent, result, started)
    if agent == "briefing" and result.status == "completed":
        db.execute(
            f"INSERT INTO {settings.table('daily_briefings')} VALUES "
            "(current_date(), :run, :content, current_timestamp())",
            {"run": tools.run_id, "content": result.final_text})
    if result.status == "error":
        raise RuntimeError(f"{agent} agent failed: {result.final_text}")
    return result
