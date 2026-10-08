"""Demo scenario: realistic recent operations layered onto the linked dataset (dev / demo only).

The linked dataset is synthetic and uniform - every location, weekday and lead behaves alike - so the
triage, SLA and capacity features have nothing to find. This generator writes a clearly labelled
scenario into the operational schema (never the source dataset):

  appointments       last SCENARIO_WEEKS at every location, with a realistic weekday shape (Sunday closed,
                     Saturday half day), staffing limits, booking lead times, and one planted problem per
                     profiled location (see PROFILES); earlier history stays as linked
  appointment_slots  the forward 14-day book, consistent with that demand and lead-time curve
  leads_raw          ~3 weeks of inbound leads written from a labelled message corpus (ids GEN-...)
  lead_actions       staff responses to the older ones, with realistic delays (run_id 'scenario')

Everything is deterministic for a given seed and date. Planted problems are documented so a demo can
show the analysis finding them.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass

import numpy as np

from chiro.capacity import SLOTS_PER_PROVIDER_DAY

SCENARIO_WEEKS = 12
LEAD_DAYS = 21
RUN_ID = "scenario"

# Mon..Sun multipliers on a location's typical demand (Python weekday(): Monday=0).
WEEKDAY_SHAPE = [1.10, 1.08, 1.04, 1.00, 0.88, 0.55, 0.0]


@dataclass(frozen=True)
class Profile:
    problem: str
    occupancy: float = 0.78  # typical bookings / effective capacity
    no_show_extra: float = 0.0
    weekday_shape: tuple | None = None
    weekly_trend: float = 0.0  # multiplicative change per week
    demand_vs_room: float | None = None  # demand set from room capacity instead (staffing-constrained)


PROFILES = {
    "LOC002": Profile("staffing-constrained: one provider for a 48-patient space", demand_vs_room=0.75),
    "LOC013": Profile("low demand: large site, few bookings", occupancy=0.42),
    "LOC019": Profile("high no-shows, worst on bookings made far ahead", no_show_extra=0.14),
    "LOC011": Profile("weekday imbalance: early week overbooked, late week empty",
                      weekday_shape=(1.45, 1.40, 1.05, 0.70, 0.45, 0.30, 0.0)),
    "LOC005": Profile("declining: losing about 5% of bookings every week", occupancy=0.85, weekly_trend=-0.05),
}

APPOINTMENT_TYPES = (["Spinal Adjustment", "Follow-Up Adjustment", "Re-Evaluation", "Physical Therapy",
                      "Therapeutic Massage", "Initial Consultation"], [0.30, 0.25, 0.08, 0.15, 0.12, 0.10])
CHANNELS = (["Online", "Phone", "Mobile App", "In-Person"], [0.35, 0.30, 0.20, 0.15])


def staffed_slots(providers: int, day: dt.date) -> int:
    wd = day.weekday()
    return 0 if wd == 6 else providers * (SLOTS_PER_PROVIDER_DAY // 2 if wd == 5 else SLOTS_PER_PROVIDER_DAY)


def lead_time(rng: np.random.Generator) -> int:
    """Days between booking and visit: a third within 2 days, most within 10, a tail to 30."""
    u = rng.random()
    if u < 0.33:
        return int(rng.integers(0, 3))
    if u < 0.78:
        return int(rng.integers(3, 11))
    return int(rng.integers(11, 31))


def profile_for(location_id: str, rng_seed: int) -> Profile:
    if location_id in PROFILES:
        return PROFILES[location_id]
    occ = 0.68 + 0.20 * np.random.default_rng(rng_seed + int(location_id[-3:])).random()
    return Profile("none", occupancy=round(float(occ), 3))


def expected_demand(loc: dict, prof: Profile, day: dt.date, start: dt.date) -> float:
    shape = prof.weekday_shape or WEEKDAY_SHAPE
    ref = (loc["room_capacity"] * prof.demand_vs_room if prof.demand_vs_room
           else min(loc["room_capacity"], loc["providers"] * SLOTS_PER_PROVIDER_DAY) * prof.occupancy)
    weeks = (day - start).days / 7.0
    return ref * shape[day.weekday()] * (1 + prof.weekly_trend) ** weeks


def appointments(locations: list[dict], providers: dict[str, list[str]], patients: dict[str, list[str]],
                 start: dt.date, end: dt.date, seed: int = 7) -> list[dict]:
    """Appointments for [start, end] at every location (see module docstring)."""
    rng = np.random.default_rng(seed)
    rows, n = [], 0
    for loc in locations:
        lid, prof = loc["location_id"], profile_for(loc["location_id"], seed)
        provs, pats = providers.get(lid) or [], patients.get(lid) or []
        if not provs or not pats:
            continue
        day = start
        while day <= end:
            cap = min(loc["room_capacity"], staffed_slots(len(provs), day))
            booked = min(int(rng.poisson(expected_demand(loc, prof, day, start))), cap) if cap else 0
            for i in range(booked):
                lt = lead_time(rng)
                no_show = 0.05 + 0.003 * lt + prof.no_show_extra * (0.5 + lt / 30)
                cancel = 0.04 + 0.002 * lt
                u = rng.random()
                status = ("No-Show" if u < no_show else "Cancelled" if u < no_show + cancel
                          else "Rescheduled" if u < no_show + cancel + 0.02 else "Completed")
                n += 1
                rows.append({
                    "appointment_id": f"GA{seed:02d}{n:07d}", "patient_id": pats[int(rng.integers(len(pats)))],
                    "provider_id": provs[i % len(provs)], "location_id": lid, "appointment_date": day.isoformat(),
                    "appointment_type": str(rng.choice(APPOINTMENT_TYPES[0], p=APPOINTMENT_TYPES[1])),
                    "booked_channel": str(rng.choice(CHANNELS[0], p=CHANNELS[1])), "status": status,
                    "lead_time_days": lt})
            day += dt.timedelta(days=1)
    return rows


def slot_book(locations: list[dict], providers: dict[str, list[str]], today: dt.date, days: int = 14,
              history_start: dt.date | None = None, seed: int = 7) -> list[dict]:
    """Forward slots: a day's final bookings follow the same demand model; a booking already exists today
    when its lead time is at least the days remaining. Mornings fill first."""
    rng = np.random.default_rng(seed + 1)
    start = history_start or today - dt.timedelta(weeks=SCENARIO_WEEKS)
    rows = []
    for loc in locations:
        lid, prof, provs = loc["location_id"], profile_for(loc["location_id"], seed), providers.get(loc["location_id"]) or []
        for d in range(1, days + 1):
            day = today + dt.timedelta(days=d)
            per_prov = staffed_slots(1, day)
            if not provs or not per_prov:
                continue
            cap = min(loc["room_capacity"], per_prov * len(provs))
            final = min(int(rng.poisson(expected_demand(loc, prof, day, start))), cap)
            booked_now = sum(1 for _ in range(final) if lead_time(rng) >= d)
            slots = [(p, k) for k in range(per_prov) for p in provs]  # slot-major: spreads across providers
            weights = np.array([1.0 / (1 + 0.08 * k) for _, k in slots])
            taken = set(rng.choice(len(slots), size=min(booked_now, len(slots)), replace=False,
                                   p=weights / weights.sum()).tolist()) if booked_now else set()
            for idx, (prov, k) in enumerate(slots):
                start_ts = dt.datetime.combine(day, dt.time(8)) + dt.timedelta(minutes=30 * k)
                rows.append({"slot_id": f"S{day:%Y%m%d}{prov}{k:02d}", "slot_start": start_ts.isoformat(),
                             "provider": prov, "is_open": idx not in taken})
    return rows


# ---------------------------------------------------------------- leads
CORPUS = os.path.join(os.path.dirname(__file__), "data", "lead_messages.json")
TIER_MIX = {"emergency": 0.02, "urgent": 0.18, "soon": 0.35, "routine": 0.45}
# Staff response delay (hours) by tier: lognormal median; out-of-hours leads wait for the morning.
RESPONSE_MEDIAN_H = {"emergency": 0.4, "urgent": 1.6, "soon": 5.0, "routine": 16.0}
CONVERSION_BASE = {"emergency": 0.05, "urgent": 0.60, "soon": 0.45, "routine": 0.28}
CONVERSION_DECAY_H = 30.0  # conversion halves roughly every 21h of waiting
NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn", "Drew", "Sam", "Robin",
         "Maria", "Wei", "Priya", "Diego", "Aisha", "Noah", "Emma", "Liam", "Olivia", "Mateo", "Sofia", "Kenji"]
SOURCES = (["website_form", "phone_inquiry", "referral", "social_media_ad", "insurance_directory", "walk_in",
            "community_event"], [0.32, 0.20, 0.15, 0.13, 0.10, 0.06, 0.04])
PAYERS = (["cash", "ppo", "medicare", "unknown"], [0.45, 0.38, 0.10, 0.07])


def load_corpus(path: str = CORPUS) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _next_business_morning(ts: dt.datetime) -> dt.datetime:
    if 8 <= ts.hour < 19 and ts.weekday() < 6:
        return ts
    nxt = ts if ts.hour < 8 else ts + dt.timedelta(days=1)
    nxt = nxt.replace(hour=8, minute=0, second=0, microsecond=0)
    while nxt.weekday() == 6:
        nxt += dt.timedelta(days=1)
    return nxt


def leads(corpus: list[dict], now: dt.datetime, locations: list[str], days: int = LEAD_DAYS,
          per_day: float = 24.0, seed: int = 7) -> tuple[list[dict], list[dict]]:
    """Inbound leads for the last `days` plus the staff responses already made to them."""
    rng = np.random.default_rng(seed + 2)
    by_tier = {t: [c for c in corpus if c["tier"] == t] for t in TIER_MIX}
    tiers = [t for t in TIER_MIX if by_tier[t]]
    mix = np.array([TIER_MIX[t] for t in tiers])
    lead_rows, action_rows = [], []
    n_leads = int(rng.poisson(per_day * days))
    for i in range(n_leads):
        # Arrivals peak in business hours.
        created = now - dt.timedelta(days=float(rng.random() * days))
        if rng.random() < 0.75:
            created = created.replace(hour=int(rng.integers(8, 20)), minute=int(rng.integers(60)))
        if created > now:
            created -= dt.timedelta(days=1)
        tier = tiers[int(rng.choice(len(tiers), p=mix / mix.sum()))]
        c = by_tier[tier][int(rng.integers(len(by_tier[tier])))]
        payer = str(rng.choice(PAYERS[0], p=PAYERS[1]))
        consent = rng.random() < 0.94
        lead = {"lead_id": f"GEN-{seed:02d}{i:05d}", "created_at": created.isoformat(),
                "first_name": NAMES[int(rng.integers(len(NAMES)))],
                "source": str(rng.choice(SOURCES[0], p=SOURCES[1])), "complaint": c["complaint"],
                "insurance_type": payer, "distance_miles": round(float(min(rng.lognormal(1.6, 0.6), 60)), 1),
                "message": c["message"], "consent_to_contact": bool(consent),
                "consent_sms": bool(consent and rng.random() < 0.8), "consent_email": bool(consent and rng.random() < 0.9),
                "first_response_hours": None, "status": "new", "converted": None,
                "num_touchpoints": 0, "assigned_location_id": locations[int(rng.integers(len(locations)))]}
        # Staff response: lognormal around the tier's median, out-of-hours waits for the morning.
        delay = float(rng.lognormal(np.log(RESPONSE_MEDIAN_H[tier]), 0.7))
        responded = _next_business_morning(created) + dt.timedelta(hours=delay)
        if consent and responded <= now - dt.timedelta(hours=2):
            hours = (responded - created).total_seconds() / 3600
            p = CONVERSION_BASE[tier] * np.exp(-hours / CONVERSION_DECAY_H) * (1.1 if payer == "ppo" else 1.0)
            resolved = responded <= now - dt.timedelta(days=2)
            converted = bool(rng.random() < p)
            lead.update(status=("booked" if converted else "lost") if resolved else "contacted",
                        converted=converted if resolved else None, num_touchpoints=int(rng.integers(1, 4)))
            drafted = created + (responded - created) * 0.6
            action_rows.append({
                "action_id": f"LA-GEN{seed:02d}{i:05d}", "run_id": RUN_ID, "lead_id": lead["lead_id"],
                "action_type": "refer_out" if tier == "emergency" else "outreach",
                "channel": "phone" if tier in ("emergency", "urgent") else ("sms" if lead["consent_sms"] else "email"),
                "priority": "high" if tier in ("emergency", "urgent") else "normal",
                "message": "[scenario: simulated past outreach]", "proposed_slot_id": None,
                "reasoning": f"[scenario] simulated staff response, {tier} lead", "status": "approved",
                "created_at": drafted.isoformat(), "reviewed_by": "scenario (simulated staff)",
                "reviewed_at": responded.isoformat()})
        lead_rows.append(lead)
    return lead_rows, action_rows


# ---------------------------------------------------------------- care-plan cohort
CARE_CORPUS = os.path.join(os.path.dirname(__file__), "data", "care_notes.json")
# Adherence patterns in an active care plan, with their share of the cohort.
ADHERENCE = {"steady": 0.35, "drifting": 0.2, "no_show_prone": 0.15, "stalled": 0.15, "finishing": 0.15}


def _visit_days(start: dt.date, gaps: list[int]) -> list[dt.date]:
    days, d = [], start
    for g in [0] + gaps:
        d += dt.timedelta(days=g)
        if d.weekday() == 6:  # closed on Sunday
            d += dt.timedelta(days=1)
        days.append(d)
    return days


def care_cohort(patients: list[dict], providers: dict[str, list[str]], corpus: list[dict], today: dt.date,
                seed: int = 7) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Weekly care plans for `patients` ({patient_id, location_id, complaint, plan_visits}) with an adherence
    pattern each, plus 2-4 staff notes from the corpus dated on their visits.
    Returns (appointments, care_notes, {patient_id: pattern})."""
    rng = np.random.default_rng(seed + 3)
    names, weights = list(ADHERENCE), np.array(list(ADHERENCE.values()))
    appts, notes, patterns = [], [], {}
    for n, p in enumerate(patients):
        provs = providers.get(p["location_id"]) or []
        if not provs:
            continue
        pattern = names[int(rng.choice(len(names), p=weights / weights.sum()))]
        plan = int(p["plan_visits"] or 8)
        jitter = lambda: int(rng.integers(-1, 2))  # noqa: E731
        if pattern == "steady":
            done = int(rng.integers(3, plan))
            days = _visit_days(today - dt.timedelta(days=7 * done - 3), [7 + jitter() for _ in range(done - 1)])
            statuses = ["Completed"] * done
        elif pattern == "drifting":
            gaps = [7, 7, 10, 14, 21][: int(rng.integers(3, 6))]
            start = today - dt.timedelta(days=sum(gaps) + int(rng.integers(18, 30)))
            days, statuses = _visit_days(start, gaps), ["Completed"] * (len(gaps) + 1)
        elif pattern == "no_show_prone":
            booked = int(rng.integers(6, 10))
            days = _visit_days(today - dt.timedelta(days=7 * booked - 2), [7 + jitter() for _ in range(booked - 1)])
            statuses = ["No-Show" if rng.random() < 0.4 else "Completed" for _ in days]
            statuses[0] = "Completed"
        elif pattern == "stalled":
            done = int(rng.integers(2, 5))
            start = today - dt.timedelta(days=7 * done + int(rng.integers(21, 40)))
            days, statuses = _visit_days(start, [7 + jitter() for _ in range(done - 1)]), ["Completed"] * done
        else:  # finishing
            done = plan - 1
            days = _visit_days(today - dt.timedelta(days=7 * done - 4), [7 + jitter() for _ in range(done - 1)])
            statuses = ["Completed"] * done
        prov = provs[n % len(provs)]
        for i, (day, status) in enumerate(zip(days, statuses)):
            if day > today:
                continue
            appts.append({"appointment_id": f"GC{seed:02d}{n:04d}{i:02d}", "patient_id": p["patient_id"],
                          "provider_id": prov, "location_id": p["location_id"], "appointment_date": day.isoformat(),
                          "appointment_type": "Initial Consultation" if i == 0 else "Spinal Adjustment",
                          "booked_channel": "Phone", "status": status, "lead_time_days": 7})
        visit_days = [d for d, s in zip(days, statuses) if s == "Completed" and d <= today]
        pool = [c for c in corpus if c["complaint"] == p["complaint"]] or corpus
        by_cat: dict[str, list[dict]] = {}
        for c in pool:
            by_cat.setdefault(c["category"], []).append(c)
        cats = sorted(by_cat)
        chosen = rng.choice(len(cats), size=min(int(rng.integers(2, 5)), len(cats)), replace=False)
        picked = [by_cat[cats[i]][int(rng.integers(len(by_cat[cats[i]])))] for i in chosen]
        for j, c in enumerate(picked):
            at = visit_days[min(j, len(visit_days) - 1)]
            notes.append({"note_id": f"CN-GEN{seed:02d}{n:04d}{j}", "patient_id": p["patient_id"], "author": prov,
                          "note": f"{c['advice']} If ignored: {c['if_ignored']}",
                          "created_at": dt.datetime.combine(at, dt.time(10 + j)).isoformat(),
                          "category": c["category"], "advice": c["advice"], "if_ignored": c["if_ignored"],
                          "importance": c["importance"]})
        patterns[p["patient_id"]] = pattern
    return appts, notes, patterns
