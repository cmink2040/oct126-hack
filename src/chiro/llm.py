"""LLM client for Databricks Foundation Model APIs (pay-per-token endpoints, available on Free Edition).

Set OPENAI_BASE_URL to use any other OpenAI-compatible server instead (e.g. a self-hosted vLLM for
local dev); LLM_ENDPOINT then names that server's model, and LLM_MAX_TOKENS raises the output cap.
"""
from __future__ import annotations

import os


def get_llm_client(workspace_client=None):
    """OpenAI-compatible client authenticated as the current Databricks identity
    (the job's run-as user, or the App's service principal), unless OPENAI_BASE_URL is set."""
    if base_url := os.getenv("OPENAI_BASE_URL"):
        from openai import OpenAI

        return OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY") or "unused")

    from databricks.sdk import WorkspaceClient

    return (workspace_client or WorkspaceClient()).serving_endpoints.get_open_ai_client()


def max_output_tokens(default: int = 2048) -> int:
    """Output-token cap per call. Reasoning models (e.g. Qwen3 on vLLM) spend output tokens thinking before
    they answer, so give them more with LLM_MAX_TOKENS (e.g. 8192)."""
    return int(os.getenv("LLM_MAX_TOKENS") or default)


def request_options(thinking: bool = True) -> dict:
    """Per-request extras. LLM_NO_THINKING=1 turns off Qwen3-style reasoning on vLLM for calls that don't
    need it (much faster); Databricks endpoints get nothing extra."""
    if not thinking and os.getenv("LLM_NO_THINKING") and os.getenv("OPENAI_BASE_URL"):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}
