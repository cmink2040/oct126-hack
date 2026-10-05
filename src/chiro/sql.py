"""One SQL interface, two Databricks backends.

Agent tools are written once against `SqlRunner` and run on:
  * `SparkSql`     - serverless job compute (Spark Connect), used by the scheduled agents
  * `WarehouseSql` - a Databricks SQL warehouse via the Statement Execution API, used by the App

Both use named parameter markers (`:name`). Never pass None as a parameter;
use '' together with NULLIF(:x, '') in SQL instead.
"""
from __future__ import annotations

import time
from typing import Any, Protocol


class SqlRunner(Protocol):
    def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None: ...


def _check_params(params: dict[str, Any] | None) -> dict[str, Any]:
    params = params or {}
    for k, v in params.items():
        if v is None:
            raise ValueError(f"SQL parameter {k!r} is None; use '' with NULLIF instead")
    return params


class SparkSql:
    def __init__(self, spark):
        self.spark = spark

    def query(self, sql, params=None):
        rows = self.spark.sql(sql, args=_check_params(params)).collect()
        return [r.asDict(recursive=True) for r in rows]

    def execute(self, sql, params=None):
        self.spark.sql(sql, args=_check_params(params)).collect()


_NUMERIC = {"BYTE", "SHORT", "INT", "LONG"}
_FLOATING = {"FLOAT", "DOUBLE", "DECIMAL"}


def _param_type(v: Any) -> str:
    if isinstance(v, bool):
        return "BOOLEAN"
    if isinstance(v, int):
        return "BIGINT"
    if isinstance(v, float):
        return "DOUBLE"
    return "STRING"


def _convert(value: str | None, type_name: str) -> Any:
    if value is None:
        return None
    if type_name in _NUMERIC:
        return int(value)
    if type_name in _FLOATING:
        return float(value)
    if type_name == "BOOLEAN":
        return value.lower() == "true"
    return value


class WarehouseSql:
    def __init__(self, warehouse_id: str, workspace_client=None, timeout_s: int = 120):
        from databricks.sdk import WorkspaceClient

        self.w = workspace_client or WorkspaceClient()
        self.warehouse_id = warehouse_id
        self.timeout_s = timeout_s

    def _run(self, sql: str, params):
        from databricks.sdk.service.sql import StatementParameterListItem, StatementState

        items = [
            StatementParameterListItem(
                name=k,
                value=str(v).lower() if isinstance(v, bool) else str(v),
                type=_param_type(v),
            )
            for k, v in _check_params(params).items()
        ]
        resp = self.w.statement_execution.execute_statement(
            statement=sql, warehouse_id=self.warehouse_id, parameters=items, wait_timeout="30s"
        )
        deadline = time.time() + self.timeout_s
        while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
            if time.time() > deadline:
                self.w.statement_execution.cancel_execution(resp.statement_id)
                raise TimeoutError("SQL statement timed out")
            time.sleep(0.5)
            resp = self.w.statement_execution.get_statement(resp.statement_id)
        if resp.status.state != StatementState.SUCCEEDED:
            err = resp.status.error.message if resp.status.error else resp.status.state
            raise RuntimeError(f"SQL failed: {err}")
        return resp

    def query(self, sql, params=None):
        resp = self._run(sql, params)
        if not resp.manifest or not resp.manifest.schema:
            return []
        cols = [(c.name, c.type_name.value if c.type_name else "STRING") for c in resp.manifest.schema.columns]
        data = list(resp.result.data_array or []) if resp.result else []
        chunk = resp.result.next_chunk_index if resp.result else None
        while chunk is not None:
            part = self.w.statement_execution.get_statement_result_chunk_n(resp.statement_id, chunk)
            data.extend(part.data_array or [])
            chunk = part.next_chunk_index
        return [{name: _convert(v, t) for (name, t), v in zip(cols, row)} for row in data]

    def execute(self, sql, params=None):
        self._run(sql, params)
