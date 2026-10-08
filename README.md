# Chiro Agents: agentic AI for a chiropractic clinic on Databricks

AI agents that **convert leads**, **retain patients** and **optimize cash prices** for a chiropractic
clinic. The whole lakehouse (data, models, agents, semantic layer, dashboard, app) is built on
**Databricks Free Edition** and deployed as a single **Databricks Asset Bundle (DAB)**.

The pipeline is **linked to a real clinic dataset** (`workspace.chiro_hackathon`: leads, patients,
visits, appointments, providers, locations, referrals, marketing_campaigns). The setup job projects
that source into the landing and operational tables the agents read (`chiro/link.py`); set
`data_source=synthetic` to fall back to the built-in generator.

> Agents draft, people decide. Every outreach message and price change lands in a review queue and
> is only sent or applied after a human approves it in the Clinic Copilot app.

## Architecture

```
 landing (*_raw Delta)          Lakeflow Declarative Pipeline (serverless)              Unity Catalog semantic layer
 ┌──────────────┐   STREAM   ┌───────────────────────────────────────────┐        ┌──────────────────────────────┐
 │ leads_raw    │──────────▶ │ leads  (streaming, expectations,          │        │ clinic_visit_metrics  (YAML) │
 │ patients_raw │            │         ai_classify + ai_extract per lead)│──gold─▶│ clinic_lead_metrics   (YAML) │
 │ visits_raw   │──────────▶ │ patients / visits (MVs, liquid clustering)│        │   MEASURE(revenue) ...       │
 └──────────────┘            │ gold_visit_facts · gold_lead_facts        │        └──────────────┬───────────────┘
                             │ gold_weekly_service_demand · cohorts      │                       │
                             └───────────────────────┬───────────────────┘                       │
                                                     ▼                                           │
                      ML refresh (sklearn + MLflow, models registered in UC)                     │
                      lead_scores · patient_churn_risk · service_pricing_stats                   │
                                                     │                                           │
            ┌────────────────────────────────────────┼───────────────────────────────────────────┤
            ▼                    ▼                   ▼                     ▼                     ▼
      Lead agent         Retention agent      Pricing agent        Briefing agent        AI/BI dashboard
   (score, red flags,   (churn x LTV, payer-  (elasticity, ai_     (KPIs + WoW trends     (metric views)
    consent, slots)      aware offers)         forecast, guards)    from metric views)
            └──────────── tool calls via Foundation Model API, traced in MLflow ─────────┘
                                                     │ pending_review rows (Delta, CDF audit)
                                                     ▼
                         Databricks App "Clinic Copilot": chat agent + analytics + approvals
```

## Why this needs Databricks (and wouldn't survive a naive Postgres port)

| Capability | Databricks feature used | Where |
|---|---|---|
| Incremental medallion ETL with data-quality rules | **Lakeflow Declarative Pipelines**: streaming tables, materialized views, `EXPECT ... ON VIOLATION DROP ROW` | `src/pipeline/*.sql` |
| LLM enrichment inside SQL, billed once per new lead | **AI Functions** `ai_classify`, `ai_extract` in a streaming table | `src/pipeline/01_silver.sql` |
| Governed business metrics shared by agents, dashboard and Genie | **Unity Catalog metric views** (`WITH METRICS LANGUAGE YAML`, `MEASURE()`) | `src/chiro/semantic.py` |
| Demand forecasting in SQL | **`ai_forecast`** table-valued function | `ClinicTools.forecast_demand` |
| Agent reasoning | **Foundation Model APIs** (pay-per-token, tool calling) | `src/chiro/agent.py` |
| Observability | **MLflow Tracing** (every agent step and tool call) | `src/chiro/tracing.py` |
| Models and agent registry | **MLflow + Unity Catalog model registry**; Copilot as an MLflow `ResponsesAgent` with GenAI evaluation | `notebooks/03`, `notebooks/40` |
| Audit trail of AI drafts and human decisions | **Delta Change Data Feed** (`table_changes()`), liquid clustering | `src/chiro/schema.py`, `get_decision_audit` |
| BI | **AI/BI (Lakeview) dashboard** on the metric views | `src/dashboards/` |
| App hosting and identity | **Databricks Apps** (service principal, declared warehouse and endpoint resources) | `src/app/app.py` |
| Governance | Unity Catalog grants, tags, comments | `notebooks/04`, `notebooks/30` |
| Deployment | **Databricks Asset Bundles**: pipeline, jobs, app, dashboard, experiment | `databricks.yml`, `resources/` |

## The agents

All agents share one tool-calling loop (`chiro/agent.py`) over the Foundation Model API. Each one
gets only the tools it needs (`ClinicTools.*_registry`).

| Agent | Goal | Key tools | Hard guardrails (enforced in code) |
|---|---|---|---|
| **Lead Concierge** | Book new-patient exams fast | `list_scored_leads` (ML score + AI intent), `get_lead`, `get_open_slots`, `queue_lead_action` | Red-flag symptoms (e.g. bladder dysfunction, chest pain) force a phone `refer_out` with no sales pitch; channel consent; no outcome claims or pressure tactics; SMS opt-out |
| **Retention** | Re-engage at-risk patients, ranked by churn risk × LTV | `list_at_risk_patients`, `get_patient_profile`, `get_retention_offers`, `queue_retention_action` | Medicare patients only get non-monetary offers (beneficiary-inducement rules); consent; 30-day contact cooldown |
| **Pricing Analyst** | Raise contribution margin on cash-pay services | `query_metrics`, `forecast_demand`, `get_cohort_retention`, `optimize_price`, `simulate_price_change`, `propose_price_change` | Cash prices only (never insurance fee schedules); max ±10% per cycle (+5% on core visits); owner's floor and ceiling; ≤110% of competitor high; ≥115% of unit cost |
| **Chief of Staff** | Morning briefing | `get_kpis`, `recent_agent_activity`, `query_metrics` | Numbers come only from tools |
| **Clinic Copilot** (App) | Ad-hoc analytics and drafting for staff | all of the above, plus `get_decision_audit` | Can't approve anything; `review()` is human-only |

The predictive models (`chiro/models.py`) are deliberately simple and transparent:

- **Lead scoring:** gradient boosting on lead attributes.
- **Churn risk:** gradient boosting on multi-snapshot recency, frequency, no-shows and plan progress, with a 60-day label horizon.
- **Price elasticity:** log-log OLS on weekly cash demand, shrunk toward a prior by its standard error. The standard error goes to the pricing agent so it can hedge when the evidence is weak.

## Deploy (Databricks Free Edition)

> Nothing has been deployed yet. These are the steps for when you're ready.

1. Sign up at <https://www.databricks.com/learn/free-edition> and install the CLI (v0.250+; tested against v1.19 schema).
2. Authenticate: `databricks auth login --host https://<your-workspace>.cloud.databricks.com`
3. Check a tool-calling model is available: **Serving** → pick an endpoint (default
   `databricks-meta-llama-3-3-70b-instruct`; a `databricks-claude-*` or `databricks-gpt-oss-*` endpoint also works).
   To use a different one, deploy with `--var llm_endpoint=<name>`.
4. Deploy and bootstrap:
   ```bash
   databricks bundle validate
   databricks bundle deploy                 # dev target, resources prefixed [dev <you>]
   databricks bundle run chiro_setup        # schema → link clinic data → pipeline → models + metric views → app grants
   databricks bundle run clinic_copilot     # start the App
   databricks bundle run chiro_daily_agents # one agent cycle (the schedule ships PAUSED)
   ```
5. Optional: `databricks bundle run chiro_register_copilot` logs, evaluates (MLflow GenAI judges) and
   registers the Copilot as a `ResponsesAgent` in UC. Serving it with `agents.deploy` is off by default
   because Free Edition limits model serving, and the App already runs the same agent.
6. Optional: create a **Genie space** on `clinic_visit_metrics` and `clinic_lead_metrics` for
   natural-language BI. It shares the same definitions as the agents.

The source dataset defaults to `workspace.chiro_hackathon`; override with
`--var source_catalog=... --var source_schema=...` (or `data_source=synthetic` for the generator).

Free Edition notes: everything runs on serverless compute plus the *Serverless Starter Warehouse*
(found with a bundle `lookup`), with one App and one pipeline. AI Functions only enrich the most
recent 90 days of leads on the first refresh, to keep token usage small. The full bootstrap (linking
~45k leads / 60k patients / 350k appointments, the pipeline and model training) can consume the
Free Edition daily compute allowance in one go — it then resets the next day.

## Data and compliance

- The lakehouse is built from the clinic dataset in `workspace.chiro_hackathon`. `chiro/link.py`
  maps it into the landing/operational tables (`leads_raw`, `patients_raw`, `visits_raw`, `services`,
  `price_history`, `appointment_slots`) the rest of the repo expects:
  `appointments` is the visit spine (status → completed/no_show/cancelled, `revenue` → `price_paid`,
  `payment_type` → `payer`), `service_type` drives the derived service catalogue, and the service
  economics (unit cost, competitor band, policy floor/ceiling) are deterministic assumptions because
  the source doesn't carry them. A short synthetic lead message/consent/insurance is added per lead so
  the AI Functions and guardrails still have inputs. Only recent leads (90 days) are AI-enriched.
- `data_source=synthetic` (job parameter) uses the self-contained generator in `chiro/datagen.py`.
  **Don't load real PHI** into Free Edition, because it isn't a HIPAA-eligible environment. For
  production, use a HIPAA-compliant workspace (Compliance Security Profile) with a signed BAA.
- Patients are identified by first name only (derived pseudonym here). Messages stay generic about
  conditions over SMS.
- Every agent draft and human decision is recorded with Change Data Feed on the action tables, and
  every agent run goes to `agent_runs` plus MLflow traces.

## Repo layout

```
databricks.yml                 bundle: variables (catalog, schema, llm_endpoint, warehouse lookup), dev/prod targets
resources/                     pipeline, jobs (setup, daily agents, register copilot), app, dashboard, experiment
src/pipeline/                  Lakeflow SQL: silver (expectations, AI Functions) and gold facts
src/chiro/                     python package shared by jobs, App and the registered agent
  agent.py tools.py prompts.py   tool-calling loop, governed tools, prompts
  semantic.py                    metric view definitions and the MEASURE() query builder
  models.py pricing.py           ML models, elasticity, guardrailed price optimiser
  guardrails.py                  red flags, advertising claims, SMS rules
  sql.py                         one SQL interface for Spark (jobs) and SQL warehouses (App)
  link.py                        maps the real clinic dataset into the landing/operational tables
  datagen.py schema.py           synthetic data, UC DDL
  copilot_agent.py               MLflow ResponsesAgent wrapper
src/notebooks/                 thin job entry points (00 setup … 40 register agent)
src/app/app.py                 Streamlit Databricks App
src/dashboards/                AI/BI dashboard (generated by scripts/build_dashboard.py)
tests/                         local tests: guardrails, pricing, models, agent loop, semantic layer
```

## Product features

| Feature | What it does | Where |
|---|---|---|
| Urgency triage | Every lead gets a tier (emergency / urgent / soon / routine) and a respond-by deadline. An LLM classifies (primary); negation-aware rules are a safety floor and the fallback; the final tier is the more severe of the two. Urgency sets response speed only, never price. | `chiro/urgency.py`, `scripts/eval_triage.py` |
| Lead prioritization + SLA | Queue order: tier, then deadline pressure (overdue first), then expected value = P(convert) x patient value by payer, with P(convert) shrunk toward the base rate by the lead model's measured AUC. 30-day active window. Response-time report per tier (on-time rate, median/p90, overdue, waiting on approval). | `chiro/priority.py`, Lead queue |
| Capacity agent | Effective capacity = min(rooms, staffed slots). Diagnoses each location (staffing-constrained, high no-shows, declining, weekday imbalance, low demand), forecasts idle slots 14 days out from the forward book and booking lead-time curve, sizes overbooking / standby lists from no-show rates, builds ranked fill lists of lapsed patients for specific days. Approved fill campaigns hand off to the Retention agent, which drafts each invitation with a slot; impact is tracked against the baseline at approval. | `chiro/capacity.py`, Capacity tab |
| Care guidance reports | Staff write care notes (advice + what happens if it's skipped, not diagnoses); a staff report and a plain-language patient report are drafted, language-checked, approved, then shown in the patient portal. | `chiro/care.py`, Care plans tab |
| Treatment themes + care-settings recommender | Clusters patients by service mix and visit rhythm; recommends non-medical settings (weekday, cadence, payment option, booking channel, add-ons, provider) from similar patients who stayed in care, with evidence strength. | `chiro/themes.py`, Care plans tab |
| Patient portal | Sign up, send an inquiry (scored and drafted by the Lead agent in seconds), see approved care guidance. | `src/patient/app.py` |

### Triage accuracy

`python scripts/eval_triage.py --set holdout --llm` scores triage on hand-labelled messages. `golden` was used to
develop the rules; `holdout` was written separately. On the holdout set (30 cases, Qwen3 27B as the LLM):

| | accuracy | emergency recall |
|---|---|---|
| rules alone (before the red-flag lexicon was widened) | 47% | 14% |
| LLM alone | 90% | 100% |
| ensemble (more severe of the two) | 90% | 100% |

With reasoning switched off (`LLM_NO_THINKING=1`, vLLM only) the LLM scores 87% with 100% emergency recall at
roughly a tenth of the latency, so use it for batch triage. Rules overfit their development set, which is why the
LLM is primary. The holdout set later informed the red-flag
lexicon, so it no longer measures the rules cleanly; write a fresh set before quoting rules-only numbers again.

### Demo scenario

The linked dataset is uniform (every location, weekday and lead behaves alike), so the triage, SLA and capacity
features have nothing to find in it. `scripts/load_scenario.py` writes a labelled scenario into the operational
schema only (never the source): the last 12 weeks of appointments with weekday shape, staffing limits, booking lead
times and one planted problem per profiled location (`chiro/scenario.py: PROFILES`), a consistent forward slot
book, ~3 weeks of leads from an LLM-written message corpus (`GEN-` ids), and simulated staff responses
(`run_id = 'scenario'`).

## Local development

```bash
pip install -e ".[dev]"
pytest                               # 24 tests, no Databricks needed (fake LLM and stub SQL)
python scripts/build_dashboard.py    # regenerate the dashboard JSON
```

Run the apps locally against your workspace (any OpenAI-compatible server works for the LLM; reasoning
models need a larger output cap):

```bash
export DATABRICKS_CONFIG_PROFILE=<profile> DATABRICKS_WAREHOUSE_ID=<warehouse id> CHIRO_SCHEMA=chiro_dev
export OPENAI_BASE_URL=<server>/v1 LLM_ENDPOINT=<model> LLM_MAX_TOKENS=8192   # omit to use Databricks FM APIs
streamlit run src/app/app.py                          # Clinic Copilot (staff)
streamlit run src/patient/app.py --server.port 8502   # Patient portal
python scripts/load_scenario.py --llm --pipeline      # demo scenario into $CHIRO_SCHEMA (dev only)
```
