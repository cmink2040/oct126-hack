"""Clinic Copilot packaged for the Mosaic AI Agent Framework (MLflow "models from code").

Logged and registered to Unity Catalog by notebooks/40_register_copilot_agent.py so it can be
evaluated with MLflow, reviewed in the Review App, or served with `databricks.agents.deploy`
on workspaces that allow custom model serving. The Databricks App uses the same tools and
prompt directly, so behaviour is identical in both places.
"""
from __future__ import annotations

import uuid

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from chiro import prompts
from chiro.agent import run_agent
from chiro.config import Settings
from chiro.llm import get_llm_client
from chiro.sql import WarehouseSql
from chiro.tools import ClinicTools


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


class ClinicCopilotAgent(ResponsesAgent):
    def __init__(self):
        cfg = mlflow.models.ModelConfig(development_config={
            "catalog": "workspace", "schema": "chiro", "llm_endpoint": Settings.llm_endpoint,
            "warehouse_id": "", "clinic_name": Settings.clinic_name,
        })
        self.settings = Settings(catalog=cfg.get("catalog"), schema=cfg.get("schema"),
                                 llm_endpoint=cfg.get("llm_endpoint"), clinic_name=cfg.get("clinic_name"))
        self.warehouse_id = cfg.get("warehouse_id")
        self._client = None
        self._db = None

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        if self._client is None:
            self._client, self._db = get_llm_client(), WarehouseSql(self.warehouse_id)
        turns = []
        for item in request.input:
            d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            if d.get("role") in ("user", "assistant"):
                turns.append({"role": d["role"], "content": _text(d.get("content"))})
        if not turns or turns[-1]["role"] != "user":
            raise ValueError("request must end with a user message")
        tools = ClinicTools(self._db, self.settings, run_id=f"copilot-{uuid.uuid4().hex[:8]}")
        result = run_agent(self._client, self.settings.llm_endpoint,
                           system=prompts.render(prompts.COPILOT, self.settings.clinic_name),
                           user=turns[-1]["content"], history=turns[:-1], registry=tools.copilot_registry())
        return ResponsesAgentResponse(output=[self.create_text_output_item(result.final_text, str(uuid.uuid4()))])


mlflow.models.set_model(ClinicCopilotAgent())
