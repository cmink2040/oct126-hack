"""MLflow tracing helpers that degrade to no-ops when mlflow is not installed (e.g. local tests)."""
from __future__ import annotations

import contextlib

try:
    import mlflow

    _HAS_MLFLOW = hasattr(mlflow, "trace")
except Exception:  # pragma: no cover - depends on environment
    mlflow = None
    _HAS_MLFLOW = False


def trace(name: str, span_type: str = "UNKNOWN"):
    def wrap(fn):
        return mlflow.trace(name=name, span_type=span_type)(fn) if _HAS_MLFLOW else fn

    return wrap


@contextlib.contextmanager
def span(name: str, span_type: str = "UNKNOWN", inputs: dict | None = None):
    if not _HAS_MLFLOW:
        yield None
        return
    with mlflow.start_span(name=name, span_type=span_type) as s:
        if inputs:
            s.set_inputs(inputs)
        yield s


def setup_experiment(path: str | None) -> None:
    """Send traces for job runs to the bundle-managed MLflow experiment."""
    if _HAS_MLFLOW and path:
        try:
            mlflow.set_experiment(path)
        except Exception as e:  # tracing must never break the agent
            print(f"MLflow experiment setup skipped: {e}")
