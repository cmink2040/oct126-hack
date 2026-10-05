"""Generate src/dashboards/clinic_performance.lvdash.json (AI/BI dashboard on the UC metric views).

Run: python scripts/build_dashboard.py   (re-run after editing; the bundle deploys the JSON)
"""
import json
import pathlib

DATASETS = {
    "kpi_30d": ("KPIs - last 30 days", """
SELECT MEASURE(revenue) AS revenue, MEASURE(contribution_margin) AS contribution_margin,
       MEASURE(active_patients) AS active_patients, MEASURE(no_show_rate) AS no_show_rate
FROM clinic_visit_metrics WHERE visit_date >= date_sub(current_date(), 30)"""),
    "lead_kpi": ("Lead KPIs - last 180 days", """
SELECT MEASURE(leads) AS leads, MEASURE(conversion_rate) AS conversion_rate
FROM clinic_lead_metrics WHERE created_date >= date_sub(current_date(), 180)"""),
    "monthly": ("Monthly revenue and margin", """
SELECT visit_month, MEASURE(revenue) AS revenue, MEASURE(contribution_margin) AS contribution_margin,
       MEASURE(active_patients) AS active_patients
FROM clinic_visit_metrics WHERE visit_date >= add_months(current_date(), -24) GROUP BY ALL"""),
    "service_mix": ("Revenue by service", """
SELECT visit_month, service_name, MEASURE(revenue) AS revenue
FROM clinic_visit_metrics WHERE visit_date >= add_months(current_date(), -12) GROUP BY ALL"""),
    "lead_source": ("Lead conversion by source", """
SELECT source, MEASURE(leads) AS leads, MEASURE(conversion_rate) AS conversion_rate
FROM clinic_lead_metrics WHERE created_date >= date_sub(current_date(), 365) GROUP BY ALL"""),
    "lead_intent": ("AI-classified lead intent", """
SELECT ai_intent, MEASURE(leads) AS leads
FROM clinic_lead_metrics WHERE created_date >= date_sub(current_date(), 90) GROUP BY ALL"""),
    "response": ("Conversion by response time", """
SELECT response_band, MEASURE(conversion_rate) AS conversion_rate, MEASURE(closed_leads) AS closed_leads
FROM clinic_lead_metrics WHERE response_band <> 'not yet' GROUP BY ALL"""),
    "cohort": ("Cohort retention curve", """
SELECT months_since_first_visit, ROUND(AVG(retention_rate), 3) AS avg_retention
FROM gold_cohort_retention WHERE months_since_first_visit BETWEEN 0 AND 12 GROUP BY ALL"""),
    "payer": ("Revenue by payer", """
SELECT visit_month, payer, MEASURE(revenue) AS revenue
FROM clinic_visit_metrics WHERE visit_date >= add_months(current_date(), -12) GROUP BY ALL"""),
    "queues": ("Agent review queues", """
SELECT 'lead' AS queue, status, COUNT(*) AS items FROM lead_actions GROUP BY ALL
UNION ALL SELECT 'retention', status, COUNT(*) FROM retention_actions GROUP BY ALL
UNION ALL SELECT 'price', status, COUNT(*) FROM price_recommendations GROUP BY ALL"""),
    "risk": ("Churn risk distribution", """
SELECT CASE WHEN churn_risk >= 0.75 THEN '4 critical' WHEN churn_risk >= 0.5 THEN '3 high'
            WHEN churn_risk >= 0.25 THEN '2 medium' ELSE '1 low' END AS risk_band,
       COUNT(*) AS patients, ROUND(SUM(churn_risk * ltv_annual), 0) AS expected_annual_loss
FROM patient_churn_risk GROUP BY ALL"""),
}


def q(dataset, *fields):
    return [{"name": "main_query", "query": {"datasetName": dataset, "disaggregated": True,
                                             "fields": [{"name": f, "expression": f"`{f}`"} for f in fields]}}]


def counter(name, dataset, field, title, pos):
    return {"widget": {"name": name, "queries": q(dataset, field), "spec": {
        "version": 2, "widgetType": "counter",
        "encodings": {"value": {"fieldName": field, "displayName": title}},
        "frame": {"showTitle": True, "title": title}}}, "position": pos}


def chart(kind, name, dataset, x, y, title, pos, color=None, x_type="categorical"):
    enc = {"x": {"fieldName": x, "scale": {"type": x_type}, "displayName": x.replace("_", " ")},
           "y": {"fieldName": y, "scale": {"type": "quantitative"}, "displayName": y.replace("_", " ")}}
    fields = [x, y]
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": color.replace("_", " ")}
        fields.append(color)
    return {"widget": {"name": name, "queries": q(dataset, *fields), "spec": {
        "version": 3, "widgetType": kind, "encodings": enc, "frame": {"showTitle": True, "title": title}}},
        "position": pos}


def pos(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


overview = [
    counter("c_rev", "kpi_30d", "revenue", "Revenue (30d)", pos(0, 0, 1, 3)),
    counter("c_margin", "kpi_30d", "contribution_margin", "Contribution margin (30d)", pos(1, 0, 1, 3)),
    counter("c_active", "kpi_30d", "active_patients", "Active patients (30d)", pos(2, 0, 1, 3)),
    counter("c_noshow", "kpi_30d", "no_show_rate", "No-show rate (30d)", pos(3, 0, 1, 3)),
    counter("c_leads", "lead_kpi", "leads", "Leads (180d)", pos(4, 0, 1, 3)),
    counter("c_conv", "lead_kpi", "conversion_rate", "Lead conversion (180d)", pos(5, 0, 1, 3)),
    chart("line", "l_revenue", "monthly", "visit_month", "revenue", "Monthly revenue", pos(0, 3, 3, 6),
          x_type="temporal"),
    chart("bar", "b_mix", "service_mix", "visit_month", "revenue", "Revenue by service", pos(3, 3, 3, 6),
          color="service_name", x_type="temporal"),
    chart("bar", "b_payer", "payer", "visit_month", "revenue", "Revenue by payer", pos(0, 9, 3, 6),
          color="payer", x_type="temporal"),
    chart("line", "l_cohort", "cohort", "months_since_first_visit", "avg_retention",
          "Retention curve (avg of monthly cohorts)", pos(3, 9, 3, 6), x_type="quantitative"),
]
growth = [
    chart("bar", "b_source", "lead_source", "source", "conversion_rate", "Lead conversion by source", pos(0, 0, 3, 6)),
    chart("bar", "b_response", "response", "response_band", "conversion_rate",
          "Conversion by first-response time", pos(3, 0, 3, 6)),
    chart("bar", "b_intent", "lead_intent", "ai_intent", "leads", "Lead intent (Databricks AI Functions, 90d)",
          pos(0, 6, 3, 6)),
    chart("bar", "b_risk", "risk", "risk_band", "expected_annual_loss", "Revenue at risk by churn band",
          pos(3, 6, 3, 6)),
    chart("bar", "b_queue", "queues", "queue", "items", "Agent review queues", pos(0, 12, 6, 5), color="status"),
]

dashboard = {
    "datasets": [{"name": k, "displayName": d, "queryLines": [ln + "\n" for ln in sql.strip().splitlines()]}
                 for k, (d, sql) in DATASETS.items()],
    "pages": [
        {"name": "overview", "displayName": "Clinic overview", "pageType": "PAGE_TYPE_CANVAS", "layout": overview},
        {"name": "growth", "displayName": "Growth, retention & agents", "pageType": "PAGE_TYPE_CANVAS",
         "layout": growth},
    ],
}

out = pathlib.Path(__file__).resolve().parents[1] / "src" / "dashboards" / "clinic_performance.lvdash.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out}")
