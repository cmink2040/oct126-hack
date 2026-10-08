"""Care guidance: what to keep doing, and what tends to happen if you don't.

Not a risk assessment and not diagnostic. Two kinds of source feed a patient's guidance:
  * staff care notes - structured: one piece of advice, the consequence of ignoring it, a category and an
    importance, written in the chart by the treating provider;
  * follow-through signals - facts from the patient's own attendance (visits spreading out, missed bookings,
    falling behind the plan's pace, nearly finished), each with fixed, staff-approved wording.
A model turns them into a staff version and a plain-language patient version. Every patient-facing item must
cite the sources it came from, and code rejects drafts that cite nothing, drift from their sources, add
outcomes staff never wrote (e.g. "surgery"), use diagnostic or certain language, or read above ~8th grade.
Staff approve before the patient sees anything; a new approved report supersedes the old one, and the patient
can answer each item ("I'm on it" / "I need help" / "not relevant") from the portal.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import statistics
import uuid

from chiro import guardrails
from chiro.config import Settings
from chiro.llm import max_output_tokens
from chiro.sql import SqlRunner

CATEGORIES = {
    "home_exercise": "stretches and strengthening exercises between visits",
    "posture_ergonomics": "desk setup, phone use, sitting and standing habits",
    "activity_modification": "what to ease off on, and returning to sport or work gradually",
    "lifting_technique": "how to lift, carry and bend",
    "aftercare": "ice or heat and what to do in the day or two after a visit",
    "sleep": "sleep position, pillow and mattress habits",
    "visit_rhythm": "keeping to the planned visit schedule",
    "daily_movement": "walking, breaks from sitting, staying active",
}
IMPORTANCE = ("high", "medium", "low")
RESPONSES = {"on_it": "I'm on it", "need_help": "I need help with this", "not_relevant": "This doesn't apply to me"}
MAX_ITEMS = 6
MAX_GRADE = 8.5  # Flesch-Kincaid grade for the patient version
MIN_GROUNDING = 0.3  # share of an item's content words found in the sources it cites

_CLINICAL = {
    r"\bdiagnos": "diagnostic language",
    r"\bprognos": "prognosis",
    r"\bprescri": "prescribing",
    r"\b(dosage|dose|\d+\s?mg)\b": "medication dosing",
    r"\byou (have|suffer from|are suffering from|'ve got) (a |an )?([\w-]+ ){0,3}(disease|disorder|syndrome|"
    r"condition|tear|fracture|herniat\w*|stenosis|scoliosis|arthritis|sciatica|subluxation)":
        "states a medical condition",
    r"\byour (spine|discs?|joints?|nerves?|vertebrae|neck|back) (is|are) (degenerat|damaged|herniat|misaligned|"
    r"out of (place|alignment)|wearing out)": "states a medical condition",
    r"\bwill (definitely|certainly|always|never)\b|\bcertain to\b|\birreversible\b|"
    r"\bnever (heal|recover|get better)": "certainty about outcomes",
    r"\b(stop|change|skip) (taking )?(your )?medication|\b(take|double|increase|reduce|stop|skip)\b.{0,25}"
    r"\b(painkillers?|ibuprofen|medications?|meds|pills|tablets)\b": "medication advice",
    r"\b(you could|you might|you may|risk) (end up )?(be )?(paralys|crippl|disabled for life)": "scare tactics",
}
# Outcomes a draft may only mention if a cited source already does: the model may soften, never escalate.
OUTCOME_TERMS = ("surgery", "surgical", "injection", "nerve damage", "permanent", "paralysis", "disability",
                 "degenerat", "arthritis", "herniat", "chronic", "fracture", "hospital", "emergency", "numbness",
                 "weakness", "disc")
_STOP = set("""a an the and or but if then so to of in on at for with from by as is are was were be been being it its
this that these those you your yours we our us they them their he she his her i me my can could may might will
would should do does did done have has had not no more most less very just also than about into over under up
down out off again each every any some such only own same too keep keeping help helps make makes get gets""".split())


# ---------------------------------------------------------------- language and readability
def check_care_language(text: str) -> list[str]:
    lowered = (text or "").lower()
    problems = [why for pattern, why in _CLINICAL.items() if re.search(pattern, lowered)]
    problems += [p for p in guardrails.check_message(text, "report") if "too short" not in p]
    return problems


def _syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(groups, 1)


def reading_grade(text: str) -> float:
    """Flesch-Kincaid grade level (heuristic syllable count; good enough to catch jargon and long sentences)."""
    sentences = max(len(re.findall(r"[.!?]+(\s|$)", text or "")), 1)
    words = re.findall(r"[A-Za-z']+", text or "")
    if not words:
        return 0.0
    return round(0.39 * len(words) / sentences + 11.8 * sum(map(_syllables, words)) / len(words) - 15.59, 1)


def _content(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 3 and w not in _STOP}


def grounding(item_text: str, source_text: str) -> float:
    """Share of the item's content words (5-letter stems) that appear in its cited sources."""
    words = _content(item_text)
    return round(len(words & _content(source_text)) / len(words), 2) if words else 1.0


# ---------------------------------------------------------------- follow-through signals
SIGNALS = {
    "visits_spreading": {
        "importance": "high",
        "advice": "Book your next visit soon to get back to your usual rhythm of about every {gap} days.",
        "if_ignored": "Long gaps partway through a plan often let stiffness and old habits creep back, so it can "
                      "take extra visits to get back to where you were.",
        "fact": "Last visit {since} days ago; visits are usually {gap} days apart."},
    "missed_visits": {
        "importance": "high",
        "advice": "If a time stops working, move the visit instead of missing it. We can text you a reminder the "
                  "day before.",
        "if_ignored": "Missed visits interrupt the plan, so progress tends to slow and the plan can take longer.",
        "fact": "{missed} of the last {booked} bookings were missed or cancelled."},
    "plan_behind": {
        "importance": "medium",
        "advice": "Try to keep to about one visit a week until your re-check.",
        "if_ignored": "Falling behind the plan's pace often means the improvement you have felt takes longer to "
                      "hold.",
        "fact": "{done} of {planned} planned visits done after {weeks} weeks."},
    "near_finish": {
        "importance": "medium",
        "advice": "Book your re-check at the end of your plan so we can agree on what comes next.",
        "if_ignored": "Stopping without a re-check makes it harder to know if your gains will hold and what to "
                      "keep doing on your own.",
        "fact": "{done} of {planned} planned visits done."},
    "on_track": {
        "importance": "low",
        "advice": "Keep the rhythm you have now; it is working.",
        "if_ignored": "Changing a rhythm that works often slows progress.",
        "fact": "{done} of {planned} planned visits done on a steady rhythm (about every {gap} days)."},
}


def follow_through_signals(history: list[dict], planned_visits: int | None, today: dt.date) -> list[dict]:
    """Signals from a patient's last 180 days of bookings ({appointment_date, status}). Each signal carries its
    fact and its fixed advice / consequence wording."""
    rows = sorted(history, key=lambda r: str(r["appointment_date"]))
    dates = [dt.date.fromisoformat(str(r["appointment_date"])[:10]) for r in rows if r["status"] == "Completed"]
    if not dates:
        return []
    gaps = [(b - a).days for a, b in zip(dates, dates[1:]) if (b - a).days > 0]
    gap = int(statistics.median(gaps)) if gaps else 7
    since = (today - dates[-1]).days
    planned = int(planned_visits or 0)
    done, weeks = len(dates), max(1, (today - dates[0]).days // 7)
    recent = rows[-6:]
    missed = sum(1 for r in recent if r["status"] in ("No-Show", "Cancelled"))
    facts = {"gap": gap, "since": since, "done": done, "planned": planned, "weeks": weeks, "missed": missed,
             "booked": len(recent)}
    found = []
    finished = planned and done >= planned
    if since > max(14, 1.5 * gap) and not finished:
        found.append("visits_spreading")
    if missed >= 2:
        found.append("missed_visits")
    if planned and not finished and weeks >= 3 and done < 0.6 * min(weeks, planned):
        found.append("plan_behind")
    if planned and planned - 1 <= done <= planned:
        found.append("near_finish")
    if not found and done >= 3:
        found.append("on_track")
    return [{"id": f"signal:{name}", "kind": "signal", "name": name, "importance": SIGNALS[name]["importance"],
             "fact": SIGNALS[name]["fact"].format(**facts), "advice": SIGNALS[name]["advice"].format(**facts),
             "if_ignored": SIGNALS[name]["if_ignored"]} for name in found]


# ---------------------------------------------------------------- drafting
REPORT_PROMPT = """You write care guidance for {clinic}, a chiropractic clinic: practical advice and what tends to
happen if it is ignored. It is NOT a risk assessment and NOT medical advice beyond what staff wrote.

You get JSON: the patient's first name, their treatment theme, care settings similar patients stuck with, and
`sources` - staff care notes (kind=note) and facts from their attendance (kind=signal). Each source has an id,
advice, if_ignored and importance.

Return ONLY a JSON object:
{{"staff_summary": "3-5 sentences for staff: where the patient is, the most important follow-through gap, what
   to raise at the next visit. Cite facts.",
  "patient_intro": "1-2 warm sentences to the patient by first name.",
  "items": [{{"advice": "what to keep doing, specific", "if_ignored": "what tends to happen if not",
              "why_now": "one short sentence linking it to their own situation (optional)",
              "sources": ["ids of the sources this item comes from"]}}]}}

Rules:
- 2 to {max_items} items, most important first. Every item cites at least one source id and says only what its
  sources say. You may merge two related sources into one item. Never add advice of your own.
- if_ignored may soften the sources, never escalate: no new outcomes, conditions or body parts.
- Hedge consequences ("often", "tends to", "may"); never "will". No diagnosis, no medication advice, no scare
  tactics. No "cure", "guarantee", "permanent", "100%", "risk-free".
- Patient text: second person, short sentences, everyday words (about 6th-8th grade). No jargon."""


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object in the response")
    data = json.loads(match.group(0))
    if not (isinstance(data.get("staff_summary"), str) and isinstance(data.get("patient_intro"), str)):
        raise ValueError("staff_summary and patient_intro must be strings")
    items = data.get("items") or []
    if not 1 <= len(items) <= MAX_ITEMS or not all(
            isinstance(i, dict) and str(i.get("advice", "")).strip() and str(i.get("if_ignored", "")).strip()
            for i in items):
        raise ValueError(f"items must be 1-{MAX_ITEMS} objects with advice and if_ignored")
    return data


def patient_text(data: dict) -> str:
    return " ".join([data["patient_intro"]] + [f"{i['advice']} {i['if_ignored']} {i.get('why_now') or ''}"
                                                for i in data["items"]])


def validate(data: dict, sources: dict[str, dict]) -> tuple[list[str], dict]:
    """Problems with a draft (empty = acceptable) and its quality measurements."""
    problems, scores = [], []
    for n, item in enumerate(data["items"], 1):
        cited = [s for s in item.get("sources") or [] if s in sources]
        if not cited:
            problems.append(f"item {n} cites no known source (use the ids given)")
            continue
        unknown = set(item.get("sources") or []) - set(cited)
        if unknown:
            problems.append(f"item {n} cites unknown sources {sorted(unknown)}")
        src_text = " ".join(f"{sources[s]['advice']} {sources[s]['if_ignored']} {sources[s].get('fact', '')}"
                            for s in cited)
        score = grounding(f"{item['advice']} {item['if_ignored']}", src_text)
        scores.append(score)
        if score < MIN_GROUNDING:
            problems.append(f"item {n} drifts from its sources (grounding {score:.2f}); restate what they say")
        item_text = f"{item['advice']} {item['if_ignored']} {item.get('why_now') or ''}".lower()
        added = [t for t in OUTCOME_TERMS if t in item_text and t not in src_text.lower()]
        if added:
            problems.append(f"item {n} adds outcomes its sources don't mention: {', '.join(added)}")
    texts = [data["patient_intro"], data["staff_summary"]]
    texts += [t for i in data["items"] for t in (i["advice"], i["if_ignored"], i.get("why_now") or "")]
    problems += sorted({p for t in texts if t for p in check_care_language(t)})
    grade = reading_grade(patient_text(data))
    if grade > MAX_GRADE:
        problems.append(f"patient text reads at grade {grade}; use shorter sentences and everyday words "
                        f"(target <= {MAX_GRADE})")
    quality = {"reading_grade": grade, "min_grounding": min(scores) if scores else None,
               "mean_grounding": round(sum(scores) / len(scores), 2) if scores else None}
    return problems, quality


def rank_items(items: list[dict], sources: dict[str, dict]) -> list[dict]:
    """Most important first; among equals, items about the patient's current behaviour (signals) first."""
    def key(item):
        cited = [sources[s] for s in item["sources"] if s in sources]
        importance = min(IMPORTANCE.index(s["importance"]) if s["importance"] in IMPORTANCE else 1 for s in cited)
        return importance, 0 if any(s["kind"] == "signal" and s["name"] != "on_track" for s in cited) else 1
    return sorted(items, key=key)


def item_key(item: dict) -> str:
    """Stable id for an item across report versions (same sources -> same key), for patient responses."""
    return hashlib.sha1(",".join(sorted(item["sources"])).encode()).hexdigest()[:10]


def render_patient(data: dict) -> str:
    lines = [data["patient_intro"].strip(), ""]
    for i in data["items"]:
        why = f"  \n  _{i['why_now'].strip()}_" if i.get("why_now") else ""
        lines.append(f"- **{i['advice'].strip()}**  \n  If this slips: {i['if_ignored'].strip()}{why}")
    return "\n".join(lines)


def render_staff(data: dict, sources: dict[str, dict], context: dict, quality: dict) -> str:
    lines = [data["staff_summary"].strip(), "", f"**Theme:** {context.get('theme') or 'n/a'}  "]
    signals = [s for s in sources.values() if s["kind"] == "signal"]
    if signals:
        lines.append("**Follow-through signals:** " + "; ".join(f"{s['name'].replace('_', ' ')} ({s['fact']})"
                                                               for s in signals))
    lines += ["", "**Items and their sources:**"]
    for i in data["items"]:
        cites = "; ".join(f"{sources[s]['kind']}: {sources[s].get('fact') or sources[s]['advice']}"
                          for s in i["sources"] if s in sources)
        lines.append(f"- {i['advice']} — if ignored: {i['if_ignored']}  \n  _from {cites}_")
    lines.append(f"\n_Reading grade {quality['reading_grade']}, grounding {quality['mean_grounding']}._")
    return "\n".join(lines)


class CareReports:
    def __init__(self, db: SqlRunner, settings: Settings, run_id: str = ""):
        self.db, self.s, self.run_id = db, settings, run_id or f"care-{uuid.uuid4().hex[:8]}"

    def t(self, name: str) -> str:
        return self.s.table(name)

    # -------------------------------------------------------------- notes
    def add_note(self, patient_id: str, author: str, advice: str, if_ignored: str = "", category: str = "",
                 importance: str = "medium") -> dict:
        advice, if_ignored = (advice or "").strip(), (if_ignored or "").strip()
        if not 10 <= len(advice) <= 500:
            return {"error": "advice must be 10-500 characters"}
        if not 10 <= len(if_ignored) <= 500:
            return {"error": "say what tends to happen if this is ignored (10-500 characters)"}
        if category not in CATEGORIES:
            return {"error": f"category must be one of {sorted(CATEGORIES)}"}
        if importance not in IMPORTANCE:
            return {"error": f"importance must be one of {IMPORTANCE}"}
        problems = check_care_language(advice) + check_care_language(if_ignored)
        if problems:
            return {"error": "care notes are follow-through advice, not diagnoses: " + "; ".join(sorted(set(problems)))}
        if not self.db.query(f"SELECT 1 FROM {self.t('patients')} WHERE patient_id = :id", {"id": patient_id}):
            return {"error": f"patient {patient_id} not found"}
        note_id = f"CN-{uuid.uuid4().hex[:10]}"
        self.db.execute(
            f"""INSERT INTO {self.t('care_notes')}
                (note_id, patient_id, author, note, created_at, category, advice, if_ignored, importance)
                VALUES (:nid, :pid, :author, :note, current_timestamp(), :cat, :advice, :ignored, :imp)""",
            {"nid": note_id, "pid": patient_id, "author": author or "staff", "note": f"{advice} If ignored: {if_ignored}",
             "cat": category, "advice": advice, "ignored": if_ignored, "imp": importance})
        return {"ok": True, "note_id": note_id}

    def notes(self, patient_id: str, limit: int = 12) -> list[dict]:
        return self.db.query(
            f"""SELECT note_id, author, created_at, coalesce(category, 'other') AS category,
                       coalesce(advice, note) AS advice, coalesce(if_ignored, '') AS if_ignored,
                       coalesce(importance, 'medium') AS importance
                FROM {self.t('care_notes')} WHERE patient_id = :id ORDER BY created_at DESC LIMIT {int(limit)}""",
            {"id": patient_id})

    # -------------------------------------------------------------- signals
    def history(self, patient_id: str) -> list[dict]:
        return self.db.query(
            f"""SELECT appointment_date, status FROM {self.t('appointments')}
                WHERE patient_id = :id AND appointment_date > date_sub(current_date(), 180)
                  AND appointment_date <= current_date() ORDER BY appointment_date""", {"id": patient_id})

    def signals(self, patient_id: str, planned_visits: int | None) -> list[dict]:
        return follow_through_signals(self.history(patient_id), planned_visits, dt.date.today())

    def sources(self, profile: dict) -> dict[str, dict]:
        notes = [{"id": f"note:{n['note_id']}", "kind": "note", "category": n["category"], "advice": n["advice"],
                  "if_ignored": n["if_ignored"], "importance": n["importance"], "author": n["author"],
                  "date": str(n["created_at"])[:10]} for n in self.notes(profile["patient_id"])]
        return {s["id"]: s for s in notes + self.signals(profile["patient_id"], profile.get("care_plan_visits"))}

    # -------------------------------------------------------------- reports
    def draft(self, client, model: str, profile: dict, recommendations: dict | None = None) -> dict:
        """Draft both versions from notes + signals; re-ask once with the problems if the draft fails checks."""
        sources = self.sources(profile)
        if not any(s["kind"] == "note" for s in sources.values()):
            return {"error": "add at least one care note first; guidance is built from staff notes"}
        strong = [r for r in (recommendations or {}).get("recommendations", [])
                  if r["evidence"] == "strong" and not r["keep_current"]]
        context = {"first_name": profile.get("first_name"), "theme": (recommendations or {}).get("theme"),
                   "care_settings_similar_patients_kept": strong,
                   "sources": [{k: v for k, v in s.items() if k != "author"} for s in sources.values()]}
        messages = [{"role": "system", "content": REPORT_PROMPT.format(clinic=self.s.clinic_name, max_items=MAX_ITEMS)},
                    {"role": "user", "content": json.dumps(context, default=str)}]
        problems, quality, attempts = [], {}, 0
        for attempts in (1, 2):
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2,
                                                  max_tokens=max_output_tokens())
            if getattr(resp.choices[0], "finish_reason", None) == "length":
                return {"error": "the model ran out of output tokens before finishing; raise LLM_MAX_TOKENS"}
            text = resp.choices[0].message.content or ""
            try:
                data = _parse(text)
                problems, quality = validate(data, sources)
            except (ValueError, json.JSONDecodeError) as e:
                problems, quality = [str(e)], {}
            if not problems:
                break
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Fix these problems and return the full JSON again: "
                                                     + "; ".join(problems)}]
        if problems:
            return {"error": "draft rejected: " + "; ".join(problems)}
        data["items"] = rank_items(data["items"], sources)
        for item in data["items"]:
            item["sources"] = [s for s in item["sources"] if s in sources]
            item["key"] = item_key(item)
        quality["attempts"] = attempts
        staff, patient = render_staff(data, sources, context, quality), render_patient(data)
        previous = self.db.query(f"SELECT coalesce(max(version), 0) AS v FROM {self.t('care_reports')} "
                                 "WHERE patient_id = :pid", {"pid": profile["patient_id"]})
        version = int((previous[0]["v"] if previous else 0) or 0) + 1
        report_id = f"CR-{uuid.uuid4().hex[:10]}"
        self.db.execute(
            f"""INSERT INTO {self.t('care_reports')}
                (report_id, run_id, patient_id, staff_report, patient_report, status, created_at, items, version,
                 quality)
                VALUES (:rid, :run, :pid, :staff, :patient, 'pending_review', current_timestamp(), :items, :version,
                        :quality)""",
            {"rid": report_id, "run": self.run_id, "pid": profile["patient_id"], "staff": staff, "patient": patient,
             "items": json.dumps({"intro": data["patient_intro"], "items": data["items"]}), "version": version,
             "quality": json.dumps(quality)})
        return {"ok": True, "report_id": report_id, "version": version, "status": "pending_review",
                "staff_report": staff, "patient_report": patient, "quality": quality}

    def due(self, limit: int = 20) -> list[dict]:
        """Patients whose care notes changed since their last drafted or approved report (or who have none)."""
        return self.db.query(
            f"""WITH n AS (SELECT patient_id, max(created_at) AS last_note, count(*) AS notes
                           FROM {self.t('care_notes')} GROUP BY patient_id),
                     r AS (SELECT patient_id, max(created_at) AS last_report FROM {self.t('care_reports')}
                           WHERE status IN ('pending_review', 'approved', 'superseded') GROUP BY patient_id)
                SELECT n.patient_id, n.last_note, n.notes, r.last_report
                FROM n LEFT JOIN r USING (patient_id)
                WHERE r.last_report IS NULL OR n.last_note > r.last_report
                ORDER BY n.last_note DESC LIMIT {int(limit)}""")

    def draft_due(self, client, model: str, tools, limit: int = 10) -> list[dict]:
        """Draft reports for the patients that are due (daily job and the Care tab)."""
        out = []
        for row in self.due(limit):
            profile = tools.get_patient_profile(row["patient_id"])
            if "error" in profile:
                continue
            recs = tools.get_care_recommendations(row["patient_id"])
            result = self.draft(client, model, profile, None if "error" in recs else recs)
            out.append({"patient_id": row["patient_id"], **{k: result.get(k) for k in ("report_id", "error", "quality")}})
        return out

    def history_of_reports(self, patient_id: str) -> list[dict]:
        return self.db.query(
            f"""SELECT r.report_id, r.version, r.status, r.created_at, r.reviewed_at, r.viewed_at, r.quality,
                       (SELECT count(*) FROM {self.t('care_report_responses')} x WHERE x.report_id = r.report_id)
                         AS responses
                FROM {self.t('care_reports')} r WHERE r.patient_id = :id ORDER BY r.version DESC""",
            {"id": patient_id})

    def latest_approved(self, patient_id: str) -> dict | None:
        rows = self.db.query(
            f"""SELECT report_id, patient_report, items, reviewed_at, viewed_at FROM {self.t('care_reports')}
                WHERE patient_id = :id AND status = 'approved' ORDER BY reviewed_at DESC LIMIT 1""", {"id": patient_id})
        if not rows:
            return None
        report = dict(rows[0])
        if isinstance(report.get("items"), str):
            report["items"] = json.loads(report["items"])
        return report

    def mark_viewed(self, report_id: str) -> None:
        self.db.execute(f"UPDATE {self.t('care_reports')} SET viewed_at = current_timestamp() "
                        "WHERE report_id = :id AND viewed_at IS NULL", {"id": report_id})

    # -------------------------------------------------------------- patient responses
    def respond(self, report_id: str, patient_id: str, key: str, response: str, comment: str = "") -> dict:
        if response not in RESPONSES:
            return {"error": f"response must be one of {sorted(RESPONSES)}"}
        report = self.latest_approved(patient_id)
        if not report or report["report_id"] != report_id or not report["items"] or \
                key not in {i["key"] for i in report["items"]["items"]}:
            return {"error": "that item is not on this patient's current guidance"}
        self.db.execute(
            f"""INSERT INTO {self.t('care_report_responses')}
                (response_id, report_id, patient_id, item_key, response, comment, created_at)
                VALUES (:id, :rid, :pid, :key, :resp, :comment, current_timestamp())""",
            {"id": f"RR-{uuid.uuid4().hex[:10]}", "rid": report_id, "pid": patient_id, "key": key, "resp": response,
             "comment": (comment or "").strip()[:500]})
        return {"ok": True}

    def responses(self, patient_id: str) -> dict[str, dict]:
        """Latest response per item key for a patient."""
        rows = self.db.query(
            f"""SELECT item_key, max_by(response, created_at) AS response, max_by(comment, created_at) AS comment,
                       max(created_at) AS at
                FROM {self.t('care_report_responses')} WHERE patient_id = :id GROUP BY item_key""", {"id": patient_id})
        return {r["item_key"]: r for r in rows}

    def help_requests(self, limit: int = 20) -> list[dict]:
        """Items patients said they need help with (latest response per item), newest first."""
        rows = self.db.query(
            f"""WITH latest AS (
                  SELECT patient_id, item_key, max_by(response, created_at) AS response,
                         max_by(comment, created_at) AS comment, max_by(report_id, created_at) AS report_id,
                         max(created_at) AS at
                  FROM {self.t('care_report_responses')} GROUP BY patient_id, item_key)
                SELECT l.patient_id, p.first_name, l.item_key, l.comment, l.at, l.report_id, r.items
                FROM latest l JOIN {self.t('patients')} p USING (patient_id)
                JOIN {self.t('care_reports')} r ON r.report_id = l.report_id
                WHERE l.response = 'need_help' ORDER BY l.at DESC LIMIT {int(limit)}""")
        for row in rows:  # show which advice the patient is stuck on
            items = json.loads(row.pop("items") or "{}").get("items", [])
            row["advice"] = next((i["advice"] for i in items if i.get("key") == row["item_key"]), "")
        return rows
