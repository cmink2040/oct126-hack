import datetime as dt

import pytest

from chiro import datagen, models

TODAY = dt.date(2026, 10, 1)


@pytest.fixture(scope="module")
def data():
    return datagen.generate(TODAY, seed=42)


def test_tables_have_expected_shape(data):
    assert set(data) == {"services", "price_history", "patients", "visits", "leads", "appointment_slots",
                         "retention_offers"}
    leads = data["leads"]
    assert (leads["status"] == "new").sum() == 40
    assert leads.loc[leads["status"] == "new", "converted"].isna().all()


def test_lead_model_beats_chance(data):
    model, metrics = models.train_lead_model(data["leads"])
    assert metrics["lead_auc"] > 0.58
    scored = models.score_leads(model, data["leads"])
    assert len(scored) == 40 and scored["score"].between(0, 1).all()


def test_churn_model_learns_signal(data):
    model, metrics = models.train_churn_model(data["patients"], data["visits"], TODAY)
    assert metrics["churn_auc"] > 0.7
    scored = models.score_churn(model, data["patients"], data["visits"], TODAY)
    assert {"patient_id", "churn_risk", "ltv_annual", "risk_reasons"} <= set(scored.columns)


def test_elasticities_are_negative_and_bounded(data):
    e = models.estimate_elasticity(data["services"], data["price_history"], data["visits"])
    assert e["elasticity"].between(-3.0, -0.2).all()
    stats = models.pricing_stats(data["services"], data["visits"], e, TODAY)
    assert (stats["weekly_cash_volume"] > 0).all()
