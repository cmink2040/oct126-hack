"""LLM client for Databricks Foundation Model APIs (pay-per-token endpoints, available on Free Edition)."""
from __future__ import annotations


def get_llm_client(workspace_client=None):
    """OpenAI-compatible client authenticated as the current Databricks identity
    (the job's run-as user, or the App's service principal)."""
    from databricks.sdk import WorkspaceClient

    return (workspace_client or WorkspaceClient()).serving_endpoints.get_open_ai_client()
