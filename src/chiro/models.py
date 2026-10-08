"""Predictive models the agents consume as tools: lead scoring, churn risk, price elasticity.

Pure pandas / scikit-learn so they can be unit tested locally; the Databricks job
loads Delta tables, calls these, logs to MLflow, and writes the gold tables back.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from chiro import urgency
from chiro.guardrails import detect_red_flags

_URGENT = re.compile(r"asap|barely|worse|this week|can't|killing|threw", re.I)

# ---------------------------------------------------------------- leads

LEAD_CATEGORICAL = ["source", "complaint", "insurance_type"]
LEAD_SCORE_COLUMNS = ["lead_id", "score", "reasons", "red_flags", "urgency", "urgency_reasons",
                      "respond_within_hours", "priority", "scored_at"]


def lead_features(leads: pd.DataFrame) -> pd.DataFrame:
    X = pd.get_dummies(leads[LEAD_CATEGORICAL].astype(str), dtype=float)
    X["distance_miles"] = leads["distance_miles"].astype(float)
    X["urgent_language"] = leads["message"].str.contains(_URGENT).astype(float)
    X["message_len"] = leads["message"].str.len().astype(float)
    return X


def _align(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return X.reindex(columns=columns, fill_value=0.0)


def train_lead_model(leads: pd.DataFrame, seed: int = 7):
    hist = leads[leads["converted"].notna()].reset_index(drop=True)
    X, y = lead_features(hist), hist["converted"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    model.feature_columns_ = list(X.columns)
    return model, {"lead_auc": float(auc), "lead_train_rows": int(len(X_tr)), "lead_base_rate": float(y.mean())}


def _lead_reasons(row) -> str:
    reasons = []
    if row.source in ("referral", "phone_inquiry"):
        reasons.append(f"high-intent source ({row.source})")
    if _URGENT.search(row.message or ""):
        reasons.append("urgent language")
    if row.distance_miles <= 5:
        reasons.append("lives nearby")
    elif row.distance_miles > 15:
        reasons.append("far away")
    if row.complaint in ("sciatica", "low_back_pain", "auto_injury"):
        reasons.append(f"core complaint ({row.complaint})")
    if row.insurance_type == "ppo":
        reasons.append("PPO insured")
    return ", ".join(reasons) or "no strong signals"


def triage_frame(leads: pd.DataFrame, llm_triage: dict[str, dict | None] | None = None) -> pd.DataFrame:
    """Urgency tier, reasons, response target and red flags for any leads (open or not)."""
    leads = leads.reset_index(drop=True)
    llm_triage = llm_triage or {}
    intents = leads["ai_intent"] if "ai_intent" in leads else [None] * len(leads)
    triage = [urgency.triage(m, c, i, llm_triage.get(lid), lid in llm_triage)
              for lid, m, c, i in zip(leads["lead_id"], leads["message"], leads["complaint"], intents)]
    return pd.DataFrame({
        "lead_id": leads["lead_id"],
        "red_flags": [", ".join(detect_red_flags(m)) for m in leads["message"]],
        "urgency": [t for t, _ in triage],
        "urgency_reasons": [", ".join(r) for _, r in triage],
        "respond_within_hours": [urgency.RESPOND_WITHIN_HOURS[t] for t, _ in triage],
    })


def score_leads(model, leads: pd.DataFrame, llm_triage: dict[str, dict | None] | None = None) -> pd.DataFrame:
    """Score open leads. `llm_triage` holds LLM second opinions for the leads it was run on ({lead_id: result
    or None if the call failed}); leads not in it are triaged by rules alone."""
    open_leads = leads[leads["status"] == "new"].reset_index(drop=True)
    if open_leads.empty:
        return pd.DataFrame(columns=LEAD_SCORE_COLUMNS)
    X = _align(lead_features(open_leads), model.feature_columns_)
    out = open_leads[["lead_id"]].copy()
    out["score"] = model.predict_proba(X)[:, 1].round(4)
    out["reasons"] = [_lead_reasons(r) for r in open_leads.itertuples()]
    out = out.merge(triage_frame(open_leads, llm_triage), on="lead_id")
    out["priority"] = [round(urgency.priority_key(t, s), 4) for t, s in zip(out["urgency"], out["score"])]
    out["scored_at"] = pd.Timestamp.now()
    return out[LEAD_SCORE_COLUMNS]


TRIAGE_COLUMNS = ["urgency", "urgency_reasons", "respond_within_hours"]


def _in_window(leads: pd.DataFrame, active_days: int, now: pd.Timestamp) -> pd.Series:
    created = pd.to_datetime(leads["created_at"], utc=True).dt.tz_localize(None)
    return created >= now - pd.Timedelta(days=active_days)


def leads_needing_triage(previous: pd.DataFrame | None, leads: pd.DataFrame, active_days: int,
                         now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Leads in the active window (any status) that have never been given an urgency tier."""
    now = now or pd.Timestamp.now()
    triaged = set() if previous is None or previous.empty else set(previous.loc[previous["urgency"].notna(), "lead_id"])
    return leads[_in_window(leads, active_days, now) & ~leads["lead_id"].isin(triaged)]


def refresh_lead_scores(previous: pd.DataFrame | None, model, leads: pd.DataFrame,
                        llm_triage: dict[str, dict | None] | None = None, active_days: int = 30,
                        now: pd.Timestamp | None = None) -> pd.DataFrame:
    """The next lead_scores table.

    Conversion scores are recomputed for open leads every run. The urgency tier is assigned once, the first
    time a lead is seen, and then kept: a lead's response deadline must not move after it arrives, and the
    SLA report judges answered leads by the tier they had when they came in. Leads that were answered before
    any refresh saw them still get a tier (with no conversion score). Earlier rows are never dropped."""
    now = now or pd.Timestamp.now()
    scored = score_leads(model, leads, llm_triage).set_index("lead_id")
    late = leads_needing_triage(previous, leads, active_days, now)
    late = late[~late["lead_id"].isin(scored.index)]
    if not late.empty:
        extra = triage_frame(late, llm_triage).set_index("lead_id")
        extra["score"], extra["reasons"], extra["scored_at"] = None, "", now
        scored = pd.concat([scored, extra])
    prev = (pd.DataFrame(columns=LEAD_SCORE_COLUMNS) if previous is None else previous)
    if "scored_at" in prev and len(prev):
        prev = prev.sort_values("scored_at", na_position="first")
    prev = prev.drop_duplicates("lead_id", keep="last").set_index("lead_id")
    # Keep the first tier a lead was given.
    kept = prev.loc[prev.index.intersection(scored.index), TRIAGE_COLUMNS]
    kept = kept[kept["urgency"].notna()]
    scored.loc[kept.index, TRIAGE_COLUMNS] = kept
    out = pd.concat([prev.drop(index=scored.index, errors="ignore"), scored]).reset_index(names="lead_id")
    has_score = out["score"].notna() & out["urgency"].notna()
    out.loc[has_score, "priority"] = [round(urgency.priority_key(t, s), 4)
                                      for t, s in zip(out.loc[has_score, "urgency"], out.loc[has_score, "score"])]
    out = out[LEAD_SCORE_COLUMNS]
    # Real nulls, not NaN: Spark stores NaN as a value that compares greater than every number.
    return out.astype(object).where(out.notna(), None)


# ---------------------------------------------------------------- churn

ACTIVE_WINDOW_DAYS = 45
CHURN_HORIZON_DAYS = 60
CHURN_FEATURES = [
    "days_since_last_visit", "visits_90d", "no_show_rate_90d", "plan_progress", "is_member",
    "distance_miles", "tenure_days", "is_cash", "avg_price_paid_90d",
]


def churn_features(patients: pd.DataFrame, visits: pd.DataFrame, asof: dt.date) -> pd.DataFrame:
    v = visits[pd.to_datetime(visits["visit_date"]) <= pd.Timestamp(asof)].copy()
    v["visit_date"] = pd.to_datetime(v["visit_date"])
    asof_ts = pd.Timestamp(asof)
    done = v[v["status"] == "completed"]
    recent = v[v["visit_date"] > asof_ts - pd.Timedelta(days=90)]
    recent_done = recent[recent["status"] == "completed"]

    last = done.groupby("patient_id")["visit_date"].max()
    total = done[done["service_id"].isin(["ADJ", "ADJ_ST"])].groupby("patient_id").size()
    f = patients.set_index("patient_id")[
        ["is_member", "distance_miles", "care_plan_visits", "first_visit_date", "insurance_type"]].copy()
    f = f[pd.to_datetime(f["first_visit_date"]) <= asof_ts]
    f["last_visit_date"] = last
    f = f[f["last_visit_date"].notna()]
    f["days_since_last_visit"] = (asof_ts - f["last_visit_date"]).dt.days
    f["visits_90d"] = recent_done.groupby("patient_id").size().reindex(f.index).fillna(0)
    sched = recent.groupby("patient_id").size().reindex(f.index).fillna(0)
    no_show = recent[recent["status"] == "no_show"].groupby("patient_id").size().reindex(f.index).fillna(0)
    f["no_show_rate_90d"] = np.where(sched > 0, no_show / sched.replace(0, 1), 0.0)
    f["plan_progress"] = (total.reindex(f.index).fillna(0) / f["care_plan_visits"]).clip(upper=2.0)
    f["tenure_days"] = (asof_ts - pd.to_datetime(f["first_visit_date"])).dt.days
    f["is_cash"] = (f["insurance_type"] == "cash").astype(float)
    f["is_member"] = f["is_member"].astype(float)
    f["avg_price_paid_90d"] = recent_done.groupby("patient_id")["price_paid"].mean().reindex(f.index).fillna(0)
    f["ltv_annual"] = (done[done["visit_date"] > asof_ts - pd.Timedelta(days=180)]
                       .groupby("patient_id")["price_paid"].sum().reindex(f.index).fillna(0) * 2).round(0)
    return f[f["days_since_last_visit"] <= ACTIVE_WINDOW_DAYS]


def _churn_labels(f: pd.DataFrame, visits: pd.DataFrame, asof: dt.date) -> pd.Series:
    v = visits[visits["status"] == "completed"]
    d = pd.to_datetime(v["visit_date"])
    window = v[(d > pd.Timestamp(asof)) & (d <= pd.Timestamp(asof) + pd.Timedelta(days=CHURN_HORIZON_DAYS))]
    returned = set(window["patient_id"])
    return pd.Series([0 if pid in returned else 1 for pid in f.index], index=f.index)


def train_churn_model(patients, visits, today: dt.date, seed: int = 7):
    frames = []
    for back in (60, 90, 120, 150, 180):  # several snapshots, all with a fully observed 60-day horizon
        asof = today - dt.timedelta(days=back)
        f = churn_features(patients, visits, asof)
        f["label"] = _churn_labels(f, visits, asof)
        frames.append(f)
    data = pd.concat(frames)
    X, y = data[CHURN_FEATURES].astype(float), data["label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return model, {"churn_auc": float(auc), "churn_train_rows": int(len(X_tr)), "churn_base_rate": float(y.mean())}


def _risk_reasons(r) -> str:
    out = []
    if r.days_since_last_visit > 30:
        out.append(f"{int(r.days_since_last_visit)} days since last visit")
    if r.no_show_rate_90d >= 0.15:
        out.append("recent no-shows")
    if r.plan_progress >= 1.0:
        out.append("care plan finished - no maintenance plan")
    elif r.plan_progress < 0.5:
        out.append("early in care plan")
    if not r.is_member and r.is_cash:
        out.append("cash-pay, not a member")
    if r.distance_miles > 12:
        out.append("long commute")
    return ", ".join(out) or "low engagement trend"


def score_churn(model, patients, visits, today: dt.date) -> pd.DataFrame:
    f = churn_features(patients, visits, today)
    out = f[["days_since_last_visit", "visits_90d", "plan_progress", "ltv_annual"]].copy()
    out["churn_risk"] = model.predict_proba(f[CHURN_FEATURES].astype(float))[:, 1].round(4)
    out["risk_reasons"] = [_risk_reasons(r) for r in f.itertuples()]
    out["plan_progress"] = out["plan_progress"].round(2)
    out["scored_at"] = pd.Timestamp.now()
    return out.reset_index().rename(columns={"index": "patient_id"})


# ---------------------------------------------------------------- elasticity

PRIOR_ELASTICITY = -1.0
PRIOR_SD = 0.5


def estimate_elasticity(services: pd.DataFrame, price_history: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Log-log OLS per service on weekly cash demand, shrunk toward a prior by its standard error.

    Mix/attach services are modelled as a share of all cash core visits, which
    controls for clinic-wide demand swings. Core visits use log volume + trend.
    """
    v = visits[(visits["status"] == "completed")].copy()
    v["week"] = pd.to_datetime(v["visit_date"]).dt.to_period("W").dt.start_time
    ph = price_history.copy()
    ph["effective_date"] = pd.to_datetime(ph["effective_date"])
    weeks = pd.Series(sorted(v["week"].unique()))
    is_core = v["service_id"].isin(["ADJ", "ADJ_ST"])
    core_cash = v[(v["payer"] == "cash") & is_core].groupby("week").size()
    core_all = v[is_core].groupby("week").size()  # add-ons are cash-paid by every payer type

    rows = []
    for s in services.itertuples():
        hist = ph[ph["service_id"] == s.service_id].sort_values("effective_date")
        idx = np.searchsorted(hist["effective_date"].to_numpy(), weeks.to_numpy(), side="right") - 1
        price = pd.Series(hist["price"].to_numpy()[np.clip(idx, 0, None)], index=weeks)
        qty = v[(v["payer"] == "cash") & (v["service_id"] == s.service_id)].groupby("week").size()
        base = core_all if s.category == "addon" else core_cash
        # Skip the first half-year while the simulated patient panel ramps up, and the partial last week.
        df = pd.DataFrame({"p": price, "q": qty.reindex(weeks).to_numpy(),
                           "core": base.reindex(weeks).to_numpy()}, index=weeks).iloc[26:-1].dropna()
        df = df[(df["q"] > 0) & (df["core"] > 0)]
        if s.category == "core" and s.service_id != "ADJ_ST":
            y = np.log(df["q"])
            X = np.column_stack([np.ones(len(df)), np.log(df["p"]), np.arange(len(df)) / 52.0])
        else:
            y = np.log(df["q"] / df["core"])
            X = np.column_stack([np.ones(len(df)), np.log(df["p"])])
        est, se = PRIOR_ELASTICITY, np.inf
        if len(df) > 20 and np.ptp(np.log(df["p"])) > 0:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            sigma2 = resid @ resid / max(len(y) - X.shape[1], 1)
            se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1]))
            est = float(beta[1])
        w_data, w_prior = (1 / se**2 if np.isfinite(se) else 0.0), 1 / PRIOR_SD**2
        shrunk = (w_data * est + w_prior * PRIOR_ELASTICITY) / (w_data + w_prior)
        rows.append({"service_id": s.service_id, "elasticity_raw": round(est, 3),
                     "elasticity_se": round(se, 3) if np.isfinite(se) else None,
                     "elasticity": round(float(np.clip(shrunk, -3.0, -0.2)), 3), "weeks_used": int(len(df))})
    return pd.DataFrame(rows)


def pricing_stats(services, visits, elasticity: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    v = visits[visits["status"] == "completed"].copy()
    d = pd.to_datetime(v["visit_date"])
    last12 = v[d > pd.Timestamp(today) - pd.Timedelta(weeks=12)]
    cash = last12[last12["payer"] == "cash"]
    out = services.merge(elasticity[["service_id", "elasticity", "elasticity_raw", "elasticity_se"]], on="service_id")
    out["weekly_cash_volume"] = out["service_id"].map(cash.groupby("service_id").size() / 12).fillna(0).round(2)
    out["weekly_insured_volume"] = out["service_id"].map(
        last12[last12["payer"] != "cash"].groupby("service_id").size() / 12).fillna(0).round(2)
    out["cash_revenue_12w"] = out["service_id"].map(cash.groupby("service_id")["price_paid"].sum()).fillna(0).round(0)
    out["competitor_mid"] = ((out["competitor_low"] + out["competitor_high"]) / 2).round(2)
    out["updated_at"] = pd.Timestamp.now()
    cols = ["service_id", "name", "category", "current_price", "unit_cost", "weekly_cash_volume",
            "weekly_insured_volume", "cash_revenue_12w", "elasticity", "elasticity_raw", "elasticity_se",
            "min_price", "max_price", "competitor_low", "competitor_mid", "competitor_high", "is_cash_pay", "updated_at"]
    return out[cols]
