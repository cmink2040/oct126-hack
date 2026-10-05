import pytest

from chiro import pricing


def sp(**kw):
    base = dict(service_id="MASSAGE60", name="Massage", category="addon", current_price=90.0, unit_cost=48.0,
                weekly_cash_volume=16.0, elasticity=-0.8, min_price=75, max_price=135,
                competitor_low=80, competitor_high=130)
    return pricing.ServicePricing(**{**base, **kw})


def test_projection_identity_at_current_price():
    p = pricing.project(sp(), 90.0)
    assert p["weekly_margin_delta"] == 0 and p["weekly_volume_projected"] == 16.0


def test_inelastic_demand_hits_upper_guardrail_not_beyond():
    best = pricing.optimize(sp(elasticity=-0.5))
    lo, hi = pricing.bounds(sp(elasticity=-0.5))
    assert best["new_price"] <= hi and best["new_price"] == pytest.approx(99.0)


def test_elastic_demand_moves_toward_textbook_optimum():
    # Constant elasticity optimum p* = c * e / (1 + e) = 48 * -3 / -2 = 72 -> clipped by the 10% step to 81.
    best = pricing.optimize(sp(elasticity=-3.0))
    assert best["new_price"] == pytest.approx(81.0)


def test_core_services_rise_more_slowly():
    lo, hi = pricing.bounds(sp(category="core", elasticity=-0.3))
    assert hi == pytest.approx(90 * 1.05)


def test_violations():
    assert pricing.violations(sp(), 120.0)
    assert pricing.violations(sp(is_cash_pay=False), 90.0)
    assert pricing.violations(sp(), 95.0) == []
