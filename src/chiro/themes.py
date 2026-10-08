"""Treatment themes and a recommender for the non-medical side of each patient's care.

Themes: patients are clustered on how their care actually runs (service mix, visit rhythm, plan
length) and each cluster gets a plain-language name ("Active rehab, weekly rhythm").

Recommendations: for one patient, look at similar patients (same theme and complaint, nearest age
band) and compare how often those who chose each option are still active. Options are only the
non-medical knobs staff can tune - visit weekday, cadence, payment option, booking channel, massage
add-on, provider - never which treatment to give. Each suggestion carries its evidence (support
and lift) so staff can judge it; weak evidence is reported as such rather than hidden.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from chiro.config import Settings

MIX = {"adj_share": "adjustment-focused", "pt_share": "active rehab", "massage_share": "soft-tissue & relaxation",
       "reeval_share": "progress check-ins"}
CLUSTER_FEATURES = [*MIX, "log_gap_days", "log_visits"]
WEEKDAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
MIN_SUPPORT = 20  # peers who chose an option before we trust its success rate
PRIOR_WEIGHT = 20  # shrink small groups toward the peer base rate


def features_sql(s: Settings) -> str:
    return f"""
    WITH mix AS (
      SELECT patient_id, count(*) AS n_visits,
             avg(CASE WHEN service_type IN ('Spinal Adjustment', 'Follow-Up Adjustment') THEN 1.0 ELSE 0 END) AS adj_share,
             avg(CASE WHEN service_type = 'Physical Therapy' THEN 1.0 ELSE 0 END) AS pt_share,
             avg(CASE WHEN service_type = 'Therapeutic Massage' THEN 1.0 ELSE 0 END) AS massage_share,
             avg(CASE WHEN service_type = 'Re-Evaluation' THEN 1.0 ELSE 0 END) AS reeval_share,
             datediff(max(visit_date), min(visit_date)) / greatest(count(*) - 1, 1) AS avg_gap_days,
             mode(dayofweek(visit_date)) AS top_weekday,
             mode(payment_type) AS top_payment,
             mode(provider_id) AS top_provider,
             max(visit_date) AS last_visit
      FROM {s.source_table('visits')} GROUP BY ALL
    ),
    appts AS (
      SELECT patient_id, avg(CASE WHEN status = 'No-Show' THEN 1.0 ELSE 0 END) AS no_show_rate,
             mode(booked_channel) AS top_booking_channel
      FROM {s.source_table('appointments')} GROUP BY ALL
    )
    SELECT p.patient_id, p.age_band, p.status, p.home_location_id, op.primary_complaint, op.insurance_type,
           op.preferred_channel, mix.*, appts.no_show_rate, appts.top_booking_channel
    FROM {s.source_table('patients')} p
    JOIN mix USING (patient_id)
    LEFT JOIN appts USING (patient_id)
    LEFT JOIN {s.table('patients')} op USING (patient_id)"""


def cadence_bucket(gap_days: float) -> str:
    return ("weekly" if gap_days <= 10 else "every 2-3 weeks" if gap_days <= 24 else "monthly" if gap_days <= 45
            else "occasional")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in [*MIX, "avg_gap_days", "n_visits", "no_show_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["log_gap_days"] = np.log1p(df["avg_gap_days"])
    df["log_visits"] = np.log1p(df["n_visits"])
    df["cadence"] = df["avg_gap_days"].map(cadence_bucket)
    df["weekday"] = pd.to_numeric(df["top_weekday"], errors="coerce").map(WEEKDAYS)
    df["massage_addon"] = np.where(df["massage_share"] >= 0.15, "with massage add-on", "without massage add-on")
    df["active"] = (df["status"] == "Active").astype(float)
    return df


class ThemeModel:
    """Clusters patients into treatment themes and recommends non-medical care settings."""

    def __init__(self, df: pd.DataFrame, k: int = 6, seed: int = 7):
        self.df = prepare(df).set_index("patient_id", drop=False)
        self.scaler = StandardScaler().fit(self.df[CLUSTER_FEATURES])
        self.km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(self.scaler.transform(self.df[CLUSTER_FEATURES]))
        self.df["theme_id"] = self.km.labels_
        self.themes = self._name_themes()
        self.df["theme"] = self.df["theme_id"].map(lambda i: self.themes[i]["name"])

    def _name_themes(self) -> dict[int, dict]:
        overall = self.df[list(MIX)].mean()
        themes = {}
        for i, g in self.df.groupby("theme_id"):
            lift = g[list(MIX)].mean() / overall.replace(0, np.nan)
            visits, gap = g["n_visits"].median(), g["avg_gap_days"].median()
            rhythm = cadence_bucket(gap)
            if visits <= 1.5:
                name, desc = "Single visit", "Patients seen once so far; the next step is a clear plan for visit two."
            else:
                mixed = lift.max() < 1.3
                focus = "mixed care" if mixed else MIX[lift.idxmax()]
                name = f"{focus.capitalize()}, {rhythm} visits"
                lean = ("No single type of visit dominates" if mixed else
                        f"Care leans on {focus} ({lift.max():.1f}x the clinic-wide share of those visits)")
                desc = (f"{lean}; {rhythm} rhythm (median {gap:.0f} days between visits) "
                        f"over about {visits:.0f} visits.")
            themes[int(i)] = {"theme_id": int(i), "name": name, "patients": int(len(g)),
                              "active_rate": round(float(g["active"].mean()), 3), "description": desc}
        # Two clusters can earn the same name; keep them distinguishable.
        seen: dict[str, int] = {}
        for t in themes.values():
            seen[t["name"]] = seen.get(t["name"], 0) + 1
            if seen[t["name"]] > 1:
                t["name"] += f" ({seen[t['name']]})"
        return themes

    def theme_summary(self) -> list[dict]:
        return sorted(self.themes.values(), key=lambda t: -t["patients"])

    def peers(self, patient_id: str) -> pd.DataFrame:
        me = self.df.loc[patient_id]
        pool = self.df[(self.df["theme_id"] == me["theme_id"]) & (self.df["patient_id"] != patient_id)]
        same = pool[pool["primary_complaint"] == me["primary_complaint"]]
        narrower = same[same["age_band"] == me["age_band"]]
        return narrower if len(narrower) >= 150 else same if len(same) >= 150 else pool

    def recommend(self, patient_id: str) -> dict:
        if patient_id not in self.df.index:
            return {"error": f"no visit history for {patient_id}"}
        me, peers = self.df.loc[patient_id], self.peers(patient_id)
        base = float(peers["active"].mean())
        aspects = {
            "visit weekday": ("weekday", None),
            "visit cadence": ("cadence", None),
            "massage add-on": ("massage_addon", None),
            "booking channel": ("top_booking_channel", None),
            "provider": ("top_provider", peers["home_location_id"] == me["home_location_id"]),
        }
        if me.get("insurance_type") != "medicare":  # Medicare billing rules: don't steer payment options
            aspects["payment option"] = ("top_payment", None)
        recs = []
        for aspect, (col, mask) in aspects.items():
            pool = peers if mask is None else peers[mask]
            stats = pool.groupby(col)["active"].agg(["sum", "count"])
            stats = stats[stats["count"] >= MIN_SUPPORT]
            if stats.empty:
                continue
            stats["rate"] = (stats["sum"] + PRIOR_WEIGHT * base) / (stats["count"] + PRIOR_WEIGHT)
            best = stats["rate"].idxmax()
            current = me[col]
            cur_rate = float(stats["rate"].get(current, base))
            lift = float(stats.loc[best, "rate"]) - cur_rate
            se = math.sqrt(max(base * (1 - base), 1e-6) / stats.loc[best, "count"])
            recs.append({
                "aspect": aspect, "current": None if pd.isna(current) else str(current), "recommended": str(best),
                "keep_current": bool(best == current),
                "peer_active_rate_recommended": round(float(stats.loc[best, "rate"]), 3),
                "peer_active_rate_current": round(cur_rate, 3),
                "lift": round(lift, 3), "support": int(stats.loc[best, "count"]),
                "evidence": "strong" if lift >= 2 * se and lift > 0.03 else "weak",
            })
        recs.sort(key=lambda r: (r["keep_current"], -r["lift"]))
        theme = self.themes[int(me["theme_id"])]
        return {"patient_id": patient_id, "theme": theme["name"], "theme_description": theme["description"],
                "peers": int(len(peers)), "peer_active_rate": round(base, 3), "recommendations": recs}


_CACHE: dict[tuple, tuple[float, ThemeModel]] = {}
CACHE_TTL_S = 6 * 3600


def get_model(db, settings: Settings) -> ThemeModel:
    """Build once per schema and process (about 60k patients; a few seconds), then reuse."""
    key = (settings.catalog, settings.schema, settings.source_catalog, settings.source_schema)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    model = ThemeModel(pd.DataFrame(db.query(features_sql(settings))))
    _CACHE[key] = (time.time(), model)
    return model
