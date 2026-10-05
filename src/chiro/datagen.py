"""Synthetic clinic data (no real PHI - Databricks Free Edition is not a HIPAA environment).

The generator plants real signal so the models have something to learn:
  * lead conversion depends on source, complaint, urgency, distance, payer and response speed
  * patient continuation depends on loyalty (membership, distance, referral) and cash price
  * add-on / upgrade take-up responds to price with a known elasticity
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "service_id", "name", "category", "ref_price", "unit_cost", "duration_min",
    "competitor_low", "competitor_high", "min_price", "max_price", "insurance_allowed",
]
SERVICES = [
    ("NEW_EXAM", "New patient exam + X-ray", "core", 129, 45, 45, 99, 179, 99, 179, 110),
    ("ADJ", "Chiropractic adjustment", "core", 65, 22, 15, 45, 85, 50, 90, 48),
    ("ADJ_ST", "Adjustment + soft tissue", "core", 85, 30, 30, 65, 110, 65, 115, 62),
    ("MASSAGE60", "60-min therapeutic massage", "addon", 95, 48, 60, 80, 130, 75, 135, None),
    ("DECOMP", "Spinal decompression session", "addon", 110, 35, 30, 80, 150, 80, 155, None),
]
# Ground-truth elasticities (the pricing model has to rediscover these).
TRUE_ELASTICITY = {"NEW_EXAM": -0.8, "ADJ": -0.6, "ADJ_ST": -1.1, "MASSAGE60": -1.6, "DECOMP": -1.3}
MEMBER_DISCOUNT = 0.8

SOURCES = ["google_ads", "website_form", "facebook", "referral", "yelp", "phone_call"]
SOURCE_P = [0.28, 0.20, 0.15, 0.15, 0.12, 0.10]
SOURCE_EFFECT = {"referral": 1.0, "phone_call": 0.7, "website_form": 0.3, "google_ads": 0.2, "yelp": 0.1, "facebook": -0.4}

COMPLAINTS = {
    "low_back_pain": (0.3, ["my lower back has been killing me", "lower back stiffness every morning", "I threw my back out lifting a box"]),
    "neck_pain": (0.2, ["neck pain from sitting at a desk all day", "a stiff neck I can't turn to the left"]),
    "headaches": (-0.1, ["tension headaches most afternoons", "headaches that start in my neck"]),
    "sciatica": (0.4, ["pain shooting down my right leg", "sciatica flaring up again"]),
    "sports_injury": (0.2, ["a tweaked shoulder from pickleball", "hip pain since I started running again"]),
    "posture_wellness": (-0.5, ["general posture and wellness", "wanting a tune-up, nothing specific"]),
    "auto_injury": (0.3, ["neck and back pain since a fender bender last week", "whiplash after a car accident"]),
}
PAYERS = ["cash", "ppo", "medicare", "unknown"]
PAYER_P = [0.40, 0.38, 0.12, 0.10]
PAYER_EFFECT = {"cash": 0.0, "ppo": 0.3, "medicare": -0.2, "unknown": -0.3}
PAYER_QUESTION = {
    "cash": "What's your self-pay rate?", "ppo": "Do you take PPO insurance?",
    "medicare": "Do you accept Medicare?", "unknown": "",
}
DURATIONS = ["a few days", "two weeks", "about a month", "months now"]
URGENT = ["Hoping to get in ASAP.", "Can barely sit at work today.", "It's getting worse every day.", "Need help this week if possible."]
CALM = ["Just looking for info.", "Curious if you take new patients.", "Would like to come in sometime."]
RED_FLAG_TEXT = [
    "lower back pain and now numbness in my groin area and trouble controlling my bladder",
    "the worst headache of my life, it came on suddenly this morning",
    "back pain with a high fever and chills",
    "chest pain that spreads into my left arm",
    "back pain and my legs keep giving out",
]
FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn", "Drew", "Sam", "Robin",
    "Maria", "Wei", "Priya", "Diego", "Aisha", "Noah", "Emma", "Liam", "Olivia", "Mateo", "Sofia", "Kenji",
]
PROVIDERS = ["Dr. Patel", "Dr. Nguyen"]

RETENTION_OFFERS = pd.DataFrame(
    [
        ("REBOOK_CALL", "Personal call to rebook at a time that works for you", 0.0, "cash,ppo,medicare"),
        ("PRIORITY_SLOT", "Priority access to early-morning / lunch slots", 0.0, "cash,ppo,medicare"),
        ("MEMBER20", "20% off the first month of the wellness membership", 20.0, "cash"),
        ("PKG10", "10% off a prepaid 6-visit care package", 10.0, "cash"),
        ("MASSAGE15", "15% off a therapeutic massage add-on", 15.0, "cash,ppo"),
    ],
    columns=["offer_code", "description", "discount_pct", "eligible_payers"],
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def services_frame() -> pd.DataFrame:
    df = pd.DataFrame(SERVICES, columns=SERVICE_COLUMNS)
    df["is_cash_pay"] = True  # cash price list; insured visits are billed at `insurance_allowed`
    return df


def price_history(start: dt.date, days: int, rng) -> pd.DataFrame:
    rows = []
    offsets = [0, days // 4, days // 2, 3 * days // 4]
    for s in services_frame().itertuples():
        mults = rng.permutation([0.92, 0.97, 1.03, 1.08])
        for off, m in zip(offsets, mults):
            rows.append((s.service_id, start + dt.timedelta(days=off), float(round(s.ref_price * m))))
    return pd.DataFrame(rows, columns=["service_id", "effective_date", "price"])


class _PriceBook:
    def __init__(self, ph: pd.DataFrame, start: dt.date):
        self.book = {}
        for sid, g in ph.sort_values("effective_date").groupby("service_id"):
            offs = np.array([(d - start).days for d in g["effective_date"]])
            self.book[sid] = (offs, g["price"].to_numpy())
        self.ref = dict(zip(services_frame().service_id, services_frame().ref_price))

    def price(self, sid: str, day: int) -> float:
        offs, prices = self.book[sid]
        return float(prices[np.searchsorted(offs, day, side="right") - 1])

    def rel(self, sid: str, day: int) -> float:
        return self.price(sid, day) / self.ref[sid]


def _lead_message(rng, complaint: str, payer: str, urgent: bool, red_flag: bool) -> str:
    if red_flag:
        body = f"Hi, I have {rng.choice(RED_FLAG_TEXT)}."
    else:
        body = f"Hi, I've had {rng.choice(COMPLAINTS[complaint][1])} for {rng.choice(DURATIONS)}."
    tail = rng.choice(URGENT) if urgent else rng.choice(CALM)
    return " ".join(x for x in [body, tail, PAYER_QUESTION[payer]] if x)


def make_leads(rng, n: int, start_ts: dt.datetime, end_ts: dt.datetime, historical: bool, id_offset: int = 0,
               red_flag_rate: float = 0.02) -> pd.DataFrame:
    span = (end_ts - start_ts).total_seconds()
    rows = []
    for i in range(n):
        source = rng.choice(SOURCES, p=SOURCE_P)
        complaint = rng.choice(list(COMPLAINTS))
        payer = rng.choice(PAYERS, p=PAYER_P)
        urgent = rng.random() < 0.35
        red_flag = rng.random() < red_flag_rate
        distance = float(rng.gamma(2.0, 3.0))
        consent = rng.random() < 0.93
        response_hours = float(rng.exponential(8.0)) if historical else None
        converted = None
        status = "new"
        if historical:
            logit = (-0.9 + SOURCE_EFFECT[source] + COMPLAINTS[complaint][0] + PAYER_EFFECT[payer]
                     + 0.7 * urgent - 0.07 * distance - 0.25 * np.log1p(response_hours))
            converted = bool(rng.random() < _sigmoid(logit)) and not red_flag
            status = "booked" if converted else "lost"
        rows.append({
            "lead_id": f"L{id_offset + i:06d}",
            "created_at": start_ts + dt.timedelta(seconds=float(rng.random() * span)),
            "first_name": rng.choice(FIRST_NAMES),
            "source": source,
            "complaint": complaint,
            "insurance_type": payer,
            "distance_miles": round(distance, 1),
            "message": _lead_message(rng, complaint, payer, urgent, red_flag),
            "consent_to_contact": consent,
            "consent_sms": consent and rng.random() < 0.75,
            "consent_email": consent and rng.random() < 0.85,
            "first_response_hours": None if response_hours is None else round(response_hours, 2),
            "status": status,
            "converted": converted,
        })
    df = pd.DataFrame(rows)
    # Explicit nullable dtypes so open leads (all-null columns) keep the Delta schema on append.
    df["converted"] = df["converted"].astype("boolean")
    df["first_response_hours"] = df["first_response_hours"].astype("float64")
    return df


def simulate_patients(rng, start: dt.date, days: int, n_patients: int, book: _PriceBook):
    patients, visits = [], []
    allowed = dict(zip(services_frame().service_id, services_frame().insurance_allowed))
    e = TRUE_ELASTICITY
    vid = 0
    for pid in range(n_patients):
        first_day = int(rng.integers(0, days - 14))
        age = int(rng.integers(18, 85))
        payer = rng.choice(["cash", "ppo", "medicare"], p=[0.5, 0.38, 0.12])
        if payer == "medicare" and age < 65:
            payer = "ppo"
        member = payer == "cash" and rng.random() < 0.2
        distance = float(rng.gamma(2.0, 3.0))
        channel = rng.choice(SOURCES, p=SOURCE_P)
        plan = int(rng.choice([6, 8, 12]))
        loyalty = rng.normal(0, 0.6) + 0.9 * member - 0.06 * distance + 0.35 * (channel == "referral")
        no_show_p = 0.04 + 0.05 * (distance > 10) + 0.04 * (loyalty < -0.5)
        consent_sms = rng.random() < 0.8
        patients.append({
            "patient_id": f"P{pid:05d}",
            "first_name": rng.choice(FIRST_NAMES),
            "age": age,
            "insurance_type": payer,
            "is_member": bool(member),
            "distance_miles": round(distance, 1),
            "acquisition_channel": channel,
            "primary_complaint": rng.choice(list(COMPLAINTS)),
            "care_plan_visits": plan,
            "first_visit_date": start + dt.timedelta(days=first_day),
            "consent_sms": consent_sms,
            "consent_email": rng.random() < 0.9,
            "preferred_channel": "sms" if consent_sms and rng.random() < 0.7 else "email",
        })

        def add(day, sid, status, visit_payer):
            nonlocal vid
            if visit_payer == "cash":
                price = book.price(sid, day) * (MEMBER_DISCOUNT if member and sid in ("ADJ", "ADJ_ST") else 1.0)
            else:
                price = float(allowed[sid])
            visits.append({
                "visit_id": f"V{vid:07d}", "patient_id": f"P{pid:05d}",
                "visit_date": start + dt.timedelta(days=day), "service_id": sid,
                "provider": PROVIDERS[pid % 2], "payer": visit_payer, "status": status,
                "price_paid": round(price, 2) if status == "completed" else 0.0,
            })
            vid += 1

        add(first_day, "NEW_EXAM", "completed", payer)
        day, n_done = first_day, 1
        while True:
            in_plan = n_done < plan
            day += int(rng.integers(3, 8) if in_plan else rng.integers(18, 40))
            if day >= days:
                break
            p = _sigmoid((2.6 if in_plan else 1.2) + loyalty)
            if payer == "cash" and not member:
                p *= book.rel("ADJ", day) ** e["ADJ"]  # higher price -> lower continuation
            if rng.random() > min(p, 0.98):
                break
            if rng.random() < no_show_p:
                add(day, "ADJ", "no_show", payer)
                continue
            st_p = 0.35 * (book.rel("ADJ_ST", day) ** e["ADJ_ST"] if payer == "cash" else 1.0)
            add(day, "ADJ_ST" if rng.random() < st_p else "ADJ", "completed", payer)
            n_done += 1
            # Add-ons are not covered by insurance: always cash-priced, always price-sensitive.
            if rng.random() < 0.10 * book.rel("MASSAGE60", day) ** e["MASSAGE60"]:
                add(day, "MASSAGE60", "completed", "cash")
            if rng.random() < 0.07 * book.rel("DECOMP", day) ** e["DECOMP"]:
                add(day, "DECOMP", "completed", "cash")
    return pd.DataFrame(patients), pd.DataFrame(visits)


def appointment_slots(rng, now: dt.datetime, days_ahead: int = 14, open_rate: float = 0.3) -> pd.DataFrame:
    rows = []
    day = now.date()
    for _ in range(days_ahead):
        day += dt.timedelta(days=1)
        if day.weekday() >= 5:
            continue
        for provider in PROVIDERS:
            for half_hour in range(16):  # 09:00 - 16:30
                start = dt.datetime.combine(day, dt.time(9)) + dt.timedelta(minutes=30 * half_hour)
                rows.append({
                    "slot_id": f"S{start:%Y%m%d%H%M}{provider[-1]}",
                    "slot_start": start, "provider": provider,
                    "is_open": bool(rng.random() < open_rate),
                })
    return pd.DataFrame(rows)


def generate(today: dt.date | None = None, seed: int = 42, days: int = 730, n_patients: int = 2400,
             n_hist_leads: int = 3000, n_new_leads: int = 40) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    today = today or dt.date.today()
    now = dt.datetime.combine(today, dt.time(7))
    start = today - dt.timedelta(days=days)

    services = services_frame()
    ph = price_history(start, days, rng)
    current = ph.sort_values("effective_date").groupby("service_id")["price"].last()
    services["current_price"] = services["service_id"].map(current).astype(float)

    patients, visits = simulate_patients(rng, start, days, n_patients, _PriceBook(ph, start))
    hist = make_leads(rng, n_hist_leads, dt.datetime.combine(start, dt.time()), now - dt.timedelta(days=3), True)
    new = make_leads(rng, n_new_leads, now - dt.timedelta(days=3), now, False, id_offset=n_hist_leads,
                     red_flag_rate=0.05)
    return {
        "services": services,
        "price_history": ph,
        "patients": patients,
        "visits": visits,
        "leads": pd.concat([hist, new], ignore_index=True),
        "appointment_slots": appointment_slots(rng, now),
        "retention_offers": RETENTION_OFFERS.copy(),
    }


def simulate_new_leads(now: dt.datetime, n: int, seed: int, id_offset: int) -> pd.DataFrame:
    """Daily 'inbound' leads for the demo loop (stands in for a web-form / ads ingestion feed)."""
    rng = np.random.default_rng(seed)
    return make_leads(rng, n, now - dt.timedelta(days=1), now, False, id_offset=id_offset, red_flag_rate=0.05)
