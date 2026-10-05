"""Constant-elasticity price simulation and guardrailed optimisation (cash-pay services only)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

MAX_STEP_PCT = 0.10          # max price move per review cycle
CORE_MAX_INCREASE_PCT = 0.05  # core visits drive retention; raise them more slowly
COMPETITOR_CEILING = 1.10     # never price above 110% of the local competitor high


@dataclass
class ServicePricing:
    service_id: str
    name: str
    category: str
    current_price: float
    unit_cost: float
    weekly_cash_volume: float
    elasticity: float
    min_price: float
    max_price: float
    competitor_low: float
    competitor_high: float
    is_cash_pay: bool = True

    @classmethod
    def from_row(cls, row: dict) -> "ServicePricing":
        fields = cls.__dataclass_fields__
        return cls(**{k: row[k] for k in fields if k in row})


def project(sp: ServicePricing, new_price: float) -> dict:
    """Weekly volume / revenue / contribution margin at `new_price` vs today."""
    ratio = new_price / sp.current_price
    vol = sp.weekly_cash_volume * ratio ** sp.elasticity
    base_margin = (sp.current_price - sp.unit_cost) * sp.weekly_cash_volume
    margin = (new_price - sp.unit_cost) * vol
    return {
        "service_id": sp.service_id,
        "current_price": round(sp.current_price, 2),
        "new_price": round(new_price, 2),
        "price_change_pct": round(100 * (ratio - 1), 1),
        "elasticity": round(sp.elasticity, 2),
        "weekly_volume_now": round(sp.weekly_cash_volume, 1),
        "weekly_volume_projected": round(vol, 1),
        "weekly_revenue_now": round(sp.current_price * sp.weekly_cash_volume, 0),
        "weekly_revenue_projected": round(new_price * vol, 0),
        "weekly_margin_now": round(base_margin, 0),
        "weekly_margin_projected": round(margin, 0),
        "weekly_margin_delta": round(margin - base_margin, 0),
        "margin_delta_pct": round(100 * (margin - base_margin) / base_margin, 1) if base_margin else None,
    }


def bounds(sp: ServicePricing) -> tuple[float, float]:
    up = CORE_MAX_INCREASE_PCT if sp.category == "core" else MAX_STEP_PCT
    lo = max(sp.min_price, sp.current_price * (1 - MAX_STEP_PCT), sp.unit_cost * 1.15)
    hi = min(sp.max_price, sp.current_price * (1 + up), sp.competitor_high * COMPETITOR_CEILING)
    return lo, hi


def violations(sp: ServicePricing, new_price: float) -> list[str]:
    if not sp.is_cash_pay:
        return ["service is insurance-billed; fee schedules are out of scope for the pricing agent"]
    lo, hi = bounds(sp)
    if new_price < lo - 1e-6 or new_price > hi + 1e-6:
        return [f"price {new_price:.2f} outside allowed band [{lo:.2f}, {hi:.2f}] "
                f"(policy floor/ceiling, max step, competitor ceiling, cost floor)"]
    return []


def optimize(sp: ServicePricing) -> dict:
    """Grid search whole-dollar prices inside the guardrail band for max contribution margin."""
    lo, hi = bounds(sp)
    if hi < lo:
        return {"service_id": sp.service_id, "recommendation": "hold", "reason": "no feasible price band"}
    grid = np.arange(np.ceil(lo), np.floor(hi) + 1, 1.0)
    if sp.current_price not in grid:
        grid = np.append(grid, sp.current_price)
    best = max((project(sp, float(p)) for p in grid), key=lambda r: r["weekly_margin_projected"])
    best["allowed_band"] = [round(lo, 2), round(hi, 2)]
    best["recommendation"] = "hold" if best["new_price"] == round(sp.current_price, 2) else "change"
    return best


def to_dict(sp: ServicePricing) -> dict:
    return asdict(sp)
