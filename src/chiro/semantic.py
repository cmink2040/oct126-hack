"""Semantic layer: Unity Catalog metric views + a safe MEASURE() query builder for agents.

The metric views are defined once here. That single definition produces:
  * the `CREATE VIEW ... WITH METRICS LANGUAGE YAML` DDL (notebooks/04_semantic_layer.py),
  * the allowlist the `query_metrics` agent tool validates against, and
  * the tool's JSON schema (enums), so the LLM can only ask for real measures/dimensions.

Agents never write SQL: they pick measures, dimensions and structured filters, and
Databricks resolves the business logic (ratios, distinct counts, filters) inside the
metric view - the numbers match the AI/BI dashboard and Genie exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from chiro.config import Settings


@dataclass(frozen=True)
class MetricView:
    name: str
    source: str  # gold table (pipeline materialized view)
    comment: str
    time_dimension: str
    dimensions: dict[str, str]  # name -> SQL expression over source
    measures: dict[str, str]    # name -> aggregate SQL expression
    descriptions: dict[str, str] = field(default_factory=dict)

    def yaml(self, settings: Settings) -> str:
        lines = ["version: 0.1", f"source: {settings.catalog}.{settings.schema}.{self.source}", "dimensions:"]
        for n, e in self.dimensions.items():
            lines += [f"  - name: {n}", f"    expr: {_quote(e)}"]
        lines.append("measures:")
        for n, e in self.measures.items():
            lines += [f"  - name: {n}", f"    expr: {_quote(e)}"]
        return "\n".join(lines) + "\n"

    def ddl(self, settings: Settings) -> str:
        return (f"CREATE OR REPLACE VIEW {settings.table(self.name)}\n"
                f"WITH METRICS\nLANGUAGE YAML\nAS $$\n{self.yaml(settings)}$$")


def _quote(expr: str) -> str:
    return '"' + expr.replace("\\", "\\\\").replace('"', '\\"') + '"'


VISIT_METRICS = MetricView(
    name="clinic_visit_metrics",
    source="gold_visit_facts",
    comment="Visit, revenue, margin and retention KPIs for the clinic.",
    time_dimension="visit_date",
    dimensions={
        "visit_date": "visit_date",
        "visit_week": "DATE_TRUNC('WEEK', visit_date)",
        "visit_month": "DATE_TRUNC('MONTH', visit_date)",
        "day_of_week": "DATE_FORMAT(visit_date, 'E')",
        "service_name": "service_name",
        "service_category": "category",
        "payer": "payer",
        "provider": "provider",
        "acquisition_channel": "acquisition_channel",
        "is_member": "is_member",
        "cohort_month": "cohort_month",
        "months_since_first_visit": "months_since_first_visit",
    },
    measures={
        "completed_visits": "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)",
        "no_show_rate": "AVG(CASE WHEN status = 'no_show' THEN 1.0 ELSE 0.0 END)",
        "revenue": "SUM(price_paid)",
        "contribution_margin": "SUM(margin)",
        "margin_pct": "SUM(margin) / NULLIF(SUM(price_paid), 0)",
        "revenue_per_visit": "SUM(price_paid) / NULLIF(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0)",
        "active_patients": "COUNT(DISTINCT CASE WHEN status = 'completed' THEN patient_id END)",
        "visits_per_patient": "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / "
                              "NULLIF(COUNT(DISTINCT CASE WHEN status = 'completed' THEN patient_id END), 0)",
        "new_patients": "COUNT(DISTINCT CASE WHEN service_id = 'NEW_EXAM' AND status = 'completed' THEN patient_id END)",
        "cash_revenue_share": "SUM(CASE WHEN payer = 'cash' THEN price_paid ELSE 0 END) / NULLIF(SUM(price_paid), 0)",
    },
)

LEAD_METRICS = MetricView(
    name="clinic_lead_metrics",
    source="gold_lead_facts",
    comment="Lead funnel KPIs, incl. AI-classified intent.",
    time_dimension="created_date",
    dimensions={
        "created_date": "created_date",
        "created_week": "DATE_TRUNC('WEEK', created_date)",
        "created_month": "DATE_TRUNC('MONTH', created_date)",
        "source": "source",
        "complaint": "complaint",
        "insurance_type": "insurance_type",
        "ai_intent": "COALESCE(ai_intent, 'not_classified')",
        "distance_band": "distance_band",
        "response_band": "response_band",
    },
    measures={
        "leads": "COUNT(1)",
        "closed_leads": "SUM(CASE WHEN converted IS NOT NULL THEN 1 ELSE 0 END)",
        "booked_leads": "SUM(CASE WHEN converted THEN 1 ELSE 0 END)",
        "conversion_rate": "SUM(CASE WHEN converted THEN 1 ELSE 0 END) / "
                           "NULLIF(SUM(CASE WHEN converted IS NOT NULL THEN 1 ELSE 0 END), 0)",
        "median_response_hours": "PERCENTILE_APPROX(first_response_hours, 0.5)",
        "open_leads": "SUM(CASE WHEN status = 'new' THEN 1 ELSE 0 END)",
    },
)

METRIC_VIEWS = {v.name: v for v in (VISIT_METRICS, LEAD_METRICS)}
_OPS = {"=", "!=", ">", ">=", "<", "<=", "in"}


class MetricQueryError(ValueError):
    pass


def build_metric_query(settings: Settings, metric_view: str, measures: list[str], dimensions: list[str] | None = None,
                       filters: list[dict] | None = None, last_n_days: int | None = None,
                       order_by: str | None = None, descending: bool = True, limit: int = 100) -> tuple[str, dict]:
    """Compile a structured request into a parameterised MEASURE() query. Raises MetricQueryError."""
    mv = METRIC_VIEWS.get(metric_view)
    if mv is None:
        raise MetricQueryError(f"unknown metric_view {metric_view!r}; choose from {sorted(METRIC_VIEWS)}")
    dimensions = list(dimensions or [])
    if not measures:
        raise MetricQueryError("at least one measure is required")
    for m in measures:
        if m not in mv.measures:
            raise MetricQueryError(f"unknown measure {m!r} for {metric_view}; choose from {sorted(mv.measures)}")
    for d in dimensions:
        if d not in mv.dimensions:
            raise MetricQueryError(f"unknown dimension {d!r} for {metric_view}; choose from {sorted(mv.dimensions)}")

    where, params = [], {}
    for i, f in enumerate(filters or []):
        dim, op, value = f.get("dimension"), str(f.get("op", "=")).lower(), f.get("value")
        if dim not in mv.dimensions:
            raise MetricQueryError(f"filter on unknown dimension {dim!r}")
        if op not in _OPS:
            raise MetricQueryError(f"filter op must be one of {sorted(_OPS)}")
        if op == "in":
            values = value if isinstance(value, list) else [value]
            names = [f"f{i}_{j}" for j in range(len(values))]
            params.update({n: v for n, v in zip(names, values)})
            where.append(f"`{dim}` IN ({', '.join(':' + n for n in names)})")
        else:
            params[f"f{i}"] = value
            where.append(f"`{dim}` {op} :f{i}")
    if last_n_days:
        params["n_days"] = int(last_n_days)
        where.append(f"`{mv.time_dimension}` >= date_sub(current_date(), :n_days)")

    select = [f"`{d}`" for d in dimensions] + [f"MEASURE(`{m}`) AS `{m}`" for m in measures]
    sql = f"SELECT {', '.join(select)}\nFROM {settings.table(mv.name)}"
    if where:
        sql += "\nWHERE " + " AND ".join(where)
    if dimensions:
        sql += "\nGROUP BY ALL"
    if order_by:
        if order_by not in measures and order_by not in dimensions:
            raise MetricQueryError("order_by must be one of the requested measures or dimensions")
        sql += f"\nORDER BY `{order_by}` {'DESC' if descending else 'ASC'}"
    elif dimensions:
        sql += "\nORDER BY ALL"
    sql += f"\nLIMIT {max(1, min(int(limit), 500))}"
    return sql, params


def catalog_description() -> str:
    """Compact description of the semantic layer for prompts."""
    parts = []
    for mv in METRIC_VIEWS.values():
        parts.append(f"- {mv.name} ({mv.comment}) time={mv.time_dimension}\n"
                     f"  measures: {', '.join(mv.measures)}\n  dimensions: {', '.join(mv.dimensions)}")
    return "\n".join(parts)
