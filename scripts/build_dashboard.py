"""Generate src/dashboards/clinic_performance.lvdash.json (AI/BI dashboard on the UC metric views and the
operational tables).

Run: python scripts/build_dashboard.py   (re-run after editing; the bundle deploys the JSON)
     python scripts/build_dashboard.py --check   (also runs every dataset query against $CHIRO_SCHEMA)

Layout is a 12-column grid. Queries use bare table names; the bundle supplies catalog and schema.
Colors are the validated reference palette shared with the Streamlit apps (src/ui/__init__.py).
"""
import json
import os
import pathlib
import sys

DATASETS = {
    # ---------------------------------------------------------------- overview
    "kpi_30d": ("KPIs - last 30 days", """
SELECT MEASURE(revenue) AS revenue, MEASURE(contribution_margin) AS contribution_margin,
       MEASURE(active_patients) AS active_patients, MEASURE(no_show_rate) AS no_show_rate,
       MEASURE(revenue_per_visit) AS revenue_per_visit
FROM clinic_visit_metrics WHERE visit_date >= date_sub(current_date(), 30)"""),
    "weekly": ("Weekly revenue (complete weeks)", """
SELECT visit_week, MEASURE(revenue) AS revenue, MEASURE(completed_visits) AS completed_visits
FROM clinic_visit_metrics
WHERE visit_date >= date_sub(current_date(), 182) AND visit_week < date_trunc('WEEK', current_date())
  AND visit_week > date_sub(current_date(), 182)
GROUP BY ALL"""),
    "payer": ("Monthly revenue by payer", """
SELECT visit_month, payer, MEASURE(revenue) AS revenue
FROM clinic_visit_metrics
WHERE visit_date >= add_months(date_trunc('MONTH', current_date()), -12) AND visit_month < date_trunc('MONTH', current_date())
GROUP BY ALL"""),
    "service_mix": ("Revenue by service, last 90 days", """
SELECT service_name, MEASURE(revenue) AS revenue, MEASURE(completed_visits) AS visits
FROM clinic_visit_metrics WHERE visit_date >= date_sub(current_date(), 90) GROUP BY ALL"""),
    "cohort": ("Retention curve", """
SELECT months_since_first_visit, ROUND(AVG(retention_rate), 3) AS avg_retention
FROM gold_cohort_retention WHERE months_since_first_visit BETWEEN 1 AND 12 GROUP BY ALL"""),
    # ---------------------------------------------------------------- leads
    "lead_kpi": ("Lead KPIs", """
SELECT
  (SELECT count(*) FROM leads_raw WHERE status = 'new' AND first_response_hours IS NULL
     AND created_at > current_timestamp() - INTERVAL 30 DAYS) AS open_leads,
  (SELECT count(*) FROM leads_raw WHERE created_at > current_timestamp() - INTERVAL 30 DAYS) AS leads_30d,
  (SELECT MEASURE(conversion_rate) FROM clinic_lead_metrics
     WHERE created_date >= date_sub(current_date(), 180)) AS conversion_rate"""),
    "lead_source": ("Lead conversion by source, 12 months", """
SELECT replace(source, '_', ' ') AS source, MEASURE(leads) AS leads, MEASURE(conversion_rate) AS conversion_rate
FROM clinic_lead_metrics WHERE created_date >= date_sub(current_date(), 365) GROUP BY ALL"""),
    "lead_intent": ("AI-classified lead intent, 90 days", """
SELECT replace(ai_intent, '_', ' ') AS ai_intent, MEASURE(leads) AS leads
FROM clinic_lead_metrics WHERE created_date >= date_sub(current_date(), 90) AND ai_intent IS NOT NULL GROUP BY ALL"""),
    "response": ("Conversion by first-response time", """
SELECT response_band, MEASURE(conversion_rate) AS conversion_rate, MEASURE(closed_leads) AS closed_leads
FROM clinic_lead_metrics WHERE response_band <> 'not yet' GROUP BY ALL"""),
    "sla": ("Response times by urgency, 14 days", """
WITH l AS (
  SELECT r.lead_id, r.created_at, r.status, coalesce(s.urgency, 'routine') AS urgency,
         timestampadd(MINUTE, CAST(coalesce(s.respond_within_hours, 24) * 60 AS INT), r.created_at) AS respond_by,
         CASE WHEN r.first_response_hours IS NOT NULL
              THEN timestampadd(MINUTE, CAST(r.first_response_hours * 60 AS INT), r.created_at) END AS imported
  FROM leads_raw r LEFT JOIN lead_scores s USING (lead_id)
  WHERE r.created_at > current_timestamp() - INTERVAL 14 DAYS),
a AS (SELECT lead_id, min(CASE WHEN status = 'approved' THEN reviewed_at END) AS approved FROM lead_actions GROUP BY 1),
j AS (SELECT l.*, coalesce(a.approved, l.imported) AS responded_at FROM l LEFT JOIN a USING (lead_id))
SELECT urgency, count(*) AS leads,
       avg(CASE WHEN responded_at <= respond_by THEN 1.0 WHEN responded_at IS NOT NULL THEN 0.0 END) AS on_time,
       percentile(timestampdiff(MINUTE, created_at, responded_at) / 60.0, 0.5) AS median_hours,
       count_if(responded_at IS NULL AND status = 'new' AND current_timestamp() > respond_by) AS overdue_now
FROM j GROUP BY urgency"""),
    # ---------------------------------------------------------------- capacity & care
    "location_load": ("Visits per open day by location, 12 weeks", """
SELECT location_id,
       count_if(status = 'Completed') / count(DISTINCT appointment_date) AS completed_per_day,
       count_if(status IN ('No-Show', 'Cancelled')) / count(*) AS lost_slot_rate
FROM appointments
WHERE appointment_date > date_sub(current_date(), 84) AND appointment_date <= current_date()
GROUP BY ALL"""),
    "weekday": ("Completed visits by weekday, 12 weeks", """
SELECT date_format(appointment_date, 'E') AS weekday, dayofweek(appointment_date) AS dow,
       count_if(status = 'Completed') / count(DISTINCT appointment_date) AS completed_per_day
FROM appointments
WHERE appointment_date > date_sub(current_date(), 84) AND appointment_date <= current_date()
GROUP BY ALL"""),
    "queues": ("Agent review queues", """
SELECT 'Leads' AS queue, status, COUNT(*) AS items FROM lead_actions GROUP BY ALL
UNION ALL SELECT 'Retention', status, COUNT(*) FROM retention_actions GROUP BY ALL
UNION ALL SELECT 'Pricing', status, COUNT(*) FROM price_recommendations GROUP BY ALL
UNION ALL SELECT 'Capacity', status, COUNT(*) FROM capacity_actions GROUP BY ALL
UNION ALL SELECT 'Care guidance', status, COUNT(*) FROM care_reports GROUP BY ALL"""),
    "care_kpi": ("Care guidance KPIs", """
SELECT
  (SELECT count(DISTINCT patient_id) FROM care_reports WHERE status = 'approved') AS patients_with_guidance,
  (SELECT avg(CASE WHEN viewed_at IS NOT NULL THEN 1.0 ELSE 0.0 END) FROM care_reports WHERE status = 'approved')
    AS viewed_rate,
  (SELECT count(*) FROM (SELECT patient_id, item_key, max_by(response, created_at) AS r
                         FROM care_report_responses GROUP BY ALL) WHERE r = 'need_help') AS help_requests"""),
}

PALETTE = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
           "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
STATUS = {"pending_review": "#eda100", "approved": "#1baf7a", "rejected": "#a8a69e", "superseded": "#d6d4cc"}

USD = {"type": "number-currency", "currencyCode": "USD", "abbreviation": "compact",
       "decimalPlaces": {"type": "max", "places": 1}}
PCT = {"type": "number-percent", "decimalPlaces": {"type": "max", "places": 0}}
NUM = {"type": "number", "abbreviation": "compact", "decimalPlaces": {"type": "max", "places": 1}}


def pos(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


def text(name, lines, position):
    return {"widget": {"name": name, "multilineTextboxSpec": {"lines": lines}}, "position": position}


def q(dataset, *fields):
    return [{"name": "main_query", "query": {"datasetName": dataset, "disaggregated": True,
                                             "fields": [{"name": f, "expression": f"`{f}`"} for f in fields]}}]


def counter(name, dataset, field, title, position, fmt=None, description=None):
    value = {"fieldName": field, "displayName": title}
    if fmt:
        value["format"] = fmt
    frame = {"showTitle": True, "title": title}
    if description:
        frame |= {"showDescription": True, "description": description}
    return {"widget": {"name": name, "queries": q(dataset, field), "spec": {
        "version": 2, "widgetType": "counter", "encodings": {"value": value}, "frame": frame}}, "position": position}


def chart(kind, name, dataset, x, y, title, position, *, x_type="categorical", y_title=None, y_fmt=None,
          color=None, color_map=None, horizontal=False, sort=None, layout=None, description=None, labels=False):
    cat_axis = {"fieldName": x, "scale": {"type": x_type}, "displayName": x.replace("_", " ")}
    if sort:
        cat_axis["scale"]["sort"] = sort
    val_axis = {"fieldName": y, "scale": {"type": "quantitative"}, "displayName": y_title or y.replace("_", " ")}
    if y_fmt:
        val_axis["format"] = y_fmt
    enc = {"x": val_axis, "y": cat_axis} if horizontal else {"x": cat_axis, "y": val_axis}
    fields = [x, y]
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color.replace("_", " ")}
        if color_map:
            enc["color"]["scale"]["mappings"] = [{"value": k, "color": v} for k, v in color_map.items()]
        fields.append(color)
    if labels:
        enc["label"] = {"show": True}
    spec = {"version": 3, "widgetType": kind, "encodings": enc, "frame": {"showTitle": True, "title": title}}
    if description:
        spec["frame"] |= {"showDescription": True, "description": description}
    if layout:
        spec["mark"] = {"layout": layout}
    return {"widget": {"name": name, "queries": q(dataset, *fields), "spec": spec}, "position": position}


def table(name, dataset, columns, title, position):
    fields = [c[0] for c in columns]
    cols = []
    for field, label, fmt in columns:
        col = {"fieldName": field, "displayName": label}
        if fmt:
            col["format"] = fmt
        cols.append(col)
    return {"widget": {"name": name, "queries": q(dataset, *fields), "spec": {
        "version": 2, "widgetType": "table", "encodings": {"columns": cols},
        "frame": {"showTitle": True, "title": title}}}, "position": position}


overview = [
    text("h_overview", ["## Clinic overview\n", "\n",
                        "Money and visits from the governed Unity Catalog metric views - the same definitions the "
                        "agents and Clinic Copilot use."], pos(0, 0, 12, 2)),
    counter("c_rev", "kpi_30d", "revenue", "Revenue · 30 days", pos(0, 2, 3, 3), USD),
    counter("c_margin", "kpi_30d", "contribution_margin", "Contribution margin · 30 days", pos(3, 2, 3, 3), USD),
    counter("c_active", "kpi_30d", "active_patients", "Active patients · 30 days", pos(6, 2, 2, 3), NUM),
    counter("c_rpv", "kpi_30d", "revenue_per_visit", "Revenue per visit", pos(8, 2, 2, 3),
            {**USD, "abbreviation": None, "decimalPlaces": {"type": "exact", "places": 0}}),
    counter("c_noshow", "kpi_30d", "no_show_rate", "No-show rate · 30 days", pos(10, 2, 2, 3),
            {**PCT, "decimalPlaces": {"type": "max", "places": 1}}),
    chart("line", "l_weekly", "weekly", "visit_week", "revenue", "Weekly revenue", pos(0, 5, 7, 6),
          x_type="temporal", y_title="Revenue", y_fmt=USD, description="Complete weeks, last 26"),
    chart("bar", "b_service", "service_mix", "service_name", "revenue", "Revenue by service", pos(7, 5, 5, 6),
          y_title="Revenue", y_fmt=USD, horizontal=True, sort={"by": "y-reversed"}, labels=True,
          description="Last 90 days"),
    chart("bar", "b_payer", "payer", "visit_month", "revenue", "Monthly revenue by payer", pos(0, 11, 7, 6),
          x_type="temporal", y_title="Revenue", y_fmt=USD, color="payer",
          color_map={"cash": PALETTE["light"][0], "ppo": PALETTE["light"][1], "medicare": PALETTE["light"][2]}),
    chart("line", "l_cohort", "cohort", "months_since_first_visit", "avg_retention",
          "Share of patients still visiting, by months since first visit", pos(7, 11, 5, 6),
          x_type="quantitative", y_title="Still visiting", y_fmt=PCT, description="Average across monthly cohorts"),
]
leads = [
    text("h_leads", ["## Leads & response\n", "\n",
                     "Where leads come from, what they want, and how fast the team responds. Urgency sets the "
                     "response target (emergency 15 min, urgent 1 h, soon 4 h, routine 24 h), never the price."],
         pos(0, 0, 12, 2)),
    counter("c_open", "lead_kpi", "open_leads", "Open leads awaiting a first response", pos(0, 2, 4, 3),
            description="Last 30 days"),
    counter("c_leads30", "lead_kpi", "leads_30d", "New leads · 30 days", pos(4, 2, 4, 3), NUM),
    counter("c_conv", "lead_kpi", "conversion_rate", "Lead conversion · 180 days", pos(8, 2, 4, 3), PCT),
    table("t_sla", "sla", [("urgency", "Urgency", None), ("leads", "Leads", None), ("on_time", "On time", PCT),
                           ("median_hours", "Median hours to response", NUM), ("overdue_now", "Overdue now", None)],
          "Response times by urgency · 14 days", pos(0, 5, 6, 5)),
    chart("bar", "b_response", "response", "response_band", "conversion_rate", "Conversion by first-response time",
          pos(6, 5, 6, 5), y_title="Converted", y_fmt=PCT, labels=True,
          sort={"by": "custom-order", "orderedValues": ["< 1h", "1-4h", "4-24h", "24h+"]}),
    chart("bar", "b_source", "lead_source", "source", "conversion_rate", "Lead conversion by source",
          pos(0, 10, 6, 6), y_title="Converted", y_fmt=PCT, horizontal=True, sort={"by": "y-reversed"}, labels=True,
          description="Last 12 months"),
    chart("bar", "b_intent", "lead_intent", "ai_intent", "leads", "What new leads want", pos(6, 10, 6, 6),
          y_title="Leads", horizontal=True, sort={"by": "y-reversed"}, labels=True,
          description="Classified by Databricks AI Functions, last 90 days"),
]
capacity = [
    text("h_capacity", ["## Capacity & care\n", "\n",
                        "Where visits happen, where slots are lost, what the agents have queued for approval, and how "
                        "patients are engaging with their care guidance."], pos(0, 0, 12, 2)),
    counter("c_guidance", "care_kpi", "patients_with_guidance", "Patients with approved care guidance",
            pos(0, 2, 4, 3)),
    counter("c_viewed", "care_kpi", "viewed_rate", "Guidance opened in the portal", pos(4, 2, 4, 3), PCT),
    counter("c_help", "care_kpi", "help_requests", "Items patients need help with", pos(8, 2, 4, 3)),
    chart("bar", "b_load", "location_load", "location_id", "completed_per_day", "Completed visits per open day",
          pos(0, 5, 6, 8), y_title="Visits / day", y_fmt=NUM, horizontal=True, sort={"by": "y-reversed"},
          description="By location, last 12 weeks"),
    chart("bar", "b_lost", "location_load", "location_id", "lost_slot_rate", "Bookings lost to no-shows and cancellations",
          pos(6, 5, 6, 8), y_title="Lost", y_fmt=PCT, horizontal=True, sort={"by": "y-reversed"},
          description="By location, last 12 weeks"),
    chart("bar", "b_weekday", "weekday", "weekday", "completed_per_day", "Completed visits by weekday",
          pos(0, 13, 5, 6), y_title="Visits / day", y_fmt=NUM, description="All locations, last 12 weeks",
          sort={"by": "custom-order", "orderedValues": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}),
    chart("bar", "b_queue", "queues", "queue", "items", "Agent drafts by review status", pos(5, 13, 7, 6),
          y_title="Items", color="status", color_map=STATUS, horizontal=True,
          description="Everything agents draft waits for a person to approve it"),
]

dashboard = {
    "datasets": [{"name": k, "displayName": d, "queryLines": [ln + "\n" for ln in sql.strip().splitlines()]}
                 for k, (d, sql) in DATASETS.items()],
    "pages": [
        {"name": "overview", "displayName": "Overview", "pageType": "PAGE_TYPE_CANVAS", "layout": overview},
        {"name": "leads", "displayName": "Leads & response", "pageType": "PAGE_TYPE_CANVAS", "layout": leads},
        {"name": "capacity", "displayName": "Capacity & care", "pageType": "PAGE_TYPE_CANVAS", "layout": capacity},
    ],
    "uiSettings": {"theme": {
        "canvasBackgroundColor": {"light": "#f3f2ee", "dark": "#141413"},
        "widgetBackgroundColor": {"light": "#fcfcfb", "dark": "#1f1f1d"},
        "widgetBorderColor": {"light": "#e4e2db", "dark": "#34332f"},
        "fontColor": {"light": "#1c1c1a", "dark": "#f1f0ea"},
        "selectionColor": {"light": "#0f766e", "dark": "#3cc3b2"},
        "visualizationColors": PALETTE["light"],
        "widgetHeaderAlignment": "LEFT"}},
}
# Drop format keys explicitly set to None (e.g. no abbreviation).
for page in dashboard["pages"]:
    for item in page["layout"]:
        for enc in (item["widget"].get("spec", {}).get("encodings", {}) or {}).values():
            for e in (enc if isinstance(enc, list) else [enc]):
                if isinstance(e, dict) and isinstance(e.get("format"), dict):
                    e["format"] = {k: v for k, v in e["format"].items() if v is not None}


def check() -> None:
    """Run every dataset query with bare table names resolved against $CHIRO_CATALOG.$CHIRO_SCHEMA, the way the
    dashboard resolves them."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState

    from chiro.config import Settings
    s, w = Settings.from_env(), WorkspaceClient()
    failed = 0
    for name, (_, sql) in DATASETS.items():
        r = w.statement_execution.execute_statement(statement=sql, warehouse_id=os.environ["DATABRICKS_WAREHOUSE_ID"],
                                                    catalog=s.catalog, schema=s.schema, wait_timeout="50s")
        if r.status.state == StatementState.SUCCEEDED:
            rows = (r.result.data_array or []) if r.result else []
            print(f"OK   {name:<14} {len(rows):>3} rows  {rows[:2]}"[:170])
        else:
            failed += 1
            print(f"FAIL {name:<14} {(r.status.error.message if r.status.error else r.status.state)[:150]}")
    if failed:
        sys.exit(f"{failed} dataset queries failed")


out = pathlib.Path(__file__).resolve().parents[1] / "src" / "dashboards" / "clinic_performance.lvdash.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out}")
if "--check" in sys.argv:
    check()
