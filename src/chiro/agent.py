"""Minimal tool-calling agent loop on Databricks Foundation Model APIs (OpenAI-compatible).

Every run and every tool call is captured as an MLflow trace when mlflow is available.
"""
from __future__ import annotations

import datetime as dt
import decimal
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from chiro.tracing import span, trace

MAX_TOOL_OUTPUT_CHARS = 12_000


def _json_default(o: Any):
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
        return float(o)
    return str(o)


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=_json_default)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def params(required: list[str] | None = None, **props: tuple[str, str] | dict) -> dict:
    """params(lead_id=("string", "The lead id"), limit=("integer", "Max rows")) -> JSON schema."""
    properties = {}
    for name, p in props.items():
        properties[name] = p if isinstance(p, dict) else {"type": p[0], "description": p[1]}
    return {"type": "object", "properties": properties, "required": required or []}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def __iter__(self):
        return iter(self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def call(self, name: str, arguments: str | dict | None) -> str:
        with span(f"tool:{name}", "TOOL", {"arguments": arguments}) as s:
            tool = self._tools.get(name)
            if tool is None:
                out = {"error": f"unknown tool {name!r}; available: {self.names}"}
            else:
                try:
                    args = json.loads(arguments) if isinstance(arguments, str) and arguments else (arguments or {})
                    out = tool.fn(**args)
                except TypeError as e:
                    out = {"error": f"bad arguments for {name}: {e}"}
                except Exception as e:  # surface tool failures to the model so it can recover
                    out = {"error": f"{type(e).__name__}: {e}"}
            text = to_json(out)
            if len(text) > MAX_TOOL_OUTPUT_CHARS:
                text = text[:MAX_TOOL_OUTPUT_CHARS] + '..."[truncated]'
            if s is not None:
                s.set_outputs(text)
            return text


@dataclass
class AgentResult:
    final_text: str
    status: str  # completed | max_steps | error
    steps: int
    tool_calls: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)


@trace(name="run_agent", span_type="AGENT")
def run_agent(client, model: str, system: str, user: str, registry: ToolRegistry,
              history: list[dict] | None = None, max_steps: int = 25, temperature: float = 0.2) -> AgentResult:
    messages: list[dict] = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]
    calls: list[dict] = []
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=registry.specs() or None,
            temperature=temperature, max_tokens=2048,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                for tc in tool_calls
            ]
        messages.append(assistant)
        if not tool_calls:
            return AgentResult(msg.content or "", "completed", step, calls, messages)
        for tc in tool_calls:
            out = registry.call(tc.function.name, tc.function.arguments)
            calls.append({"tool": tc.function.name, "arguments": tc.function.arguments, "ok": '"error"' not in out[:20]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return AgentResult("Stopped after reaching the step limit.", "max_steps", max_steps, calls, messages)
