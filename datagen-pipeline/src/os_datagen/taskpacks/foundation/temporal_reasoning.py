from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision

TZS = ["America/New_York", "Europe/London", "Asia/Tokyo", "America/Los_Angeles", "Australia/Sydney", "Europe/Berlin"]
UTC = ZoneInfo("UTC")


def _resolve(src: dict[str, Any], events: dict[str, Any]) -> tuple[datetime, datetime] | None:
    """One source -> UTC interval [start, end). None = unknown."""
    if src.get("when") is None and "anchor" not in src:
        return None
    tz = ZoneInfo(src["tz"])
    if "anchor" in src:
        a = _event_interval(events[src["anchor"]], events)
        if not isinstance(a, tuple):
            return None
        local = a[0].astimezone(tz).date() + timedelta(days=src["offset_days"])
        start = datetime.combine(local, time.fromisoformat(src["at"]), tzinfo=tz)
        return start.astimezone(UTC), (start + timedelta(minutes=src.get("duration_min") or 0)).astimezone(UTC)
    if src["precision"] == "date":
        d = date.fromisoformat(src["when"])
        s = datetime.combine(d, time(0, 0), tzinfo=tz)
        return s.astimezone(UTC), (s + timedelta(days=1)).astimezone(UTC)
    s = datetime.fromisoformat(src["when"]).replace(tzinfo=tz)
    return s.astimezone(UTC), (s + timedelta(minutes=src.get("duration_min") or 0)).astimezone(UTC)


def _event_interval(ev: dict[str, Any], events: dict[str, Any]) -> tuple[datetime, datetime] | str:
    """'unknown' | 'conflict' | interval. Highest-authority sources decide; a tie between different values conflicts."""
    srcs = ev["sources"]
    if not srcs:
        return "unknown"
    top = max(s.get("authority", 1) for s in srcs)
    res = [_resolve(s, events) for s in srcs if s.get("authority", 1) == top]
    if any(r is None for r in res):
        return "unknown"
    if len({r for r in res}) > 1:  # type: ignore[misc]
        return "conflict"
    return res[0]  # type: ignore[return-value]


def relate(f: dict[str, Any]) -> tuple[str, str]:
    a = _event_interval(f["events"][f["question"]["a"]], f["events"])
    b = _event_interval(f["events"][f["question"]["b"]], f["events"])
    if a == "conflict" or b == "conflict":
        return "conflicting", "conflicting_timestamps"
    if a == "unknown" or b == "unknown":
        return "unknown", "missing_or_underspecified_time"
    (as_, ae), (bs, be) = a, b  # type: ignore[misc]
    if ae <= bs:
        return "before", "a_ends_before_b_starts"
    if as_ >= be and not (as_ == bs and be == bs):
        return "after", "a_starts_after_b_ends"
    if as_ >= bs and ae <= be:
        return "during", "a_within_b"
    return "unknown", "partial_overlap_or_underspecified"


def _fmt(s: dict[str, Any]) -> str:
    if "anchor" in s:
        return f"rel:{s['anchor']}{s['offset_days']:+d}d@{s['at']}|{s['tz']}"
    if s.get("when") is None:
        return "unknown"
    return (f"date:{s['when']}|{s['tz']}" if s["precision"] == "date" else f"{s['when'].replace('T', ' ')}|{s['tz']}")


class TemporalReasoning(BaseTaskPack):
    name = "foundation_temporal_reasoning_v1"
    namespace = "foundation"
    oracle_version = "temporal_calc_v1"
    label_source = "python_datetime_event_graph_v1"
    prompt_id = "foundation.temporal"
    prompt_dir = "foundation/temporal_reasoning"
    composition_family = "general_reasoning"
    id_prefix = "tmp"
    options = [
        ("before", "The first event ends before the second begins."),
        ("after", "The first event starts after the second ends."),
        ("during", "The first event happens within the second."),
        ("unknown", "The timing given does not settle the relation."),
        ("conflicting", "Equally authoritative sources disagree on the timing."),
    ]
    instruction_variants = [
        "How does the first event relate in time to the second?",
        "What is the temporal relation between the two events named in the question?",
        "Given the timeline, when does the first event fall relative to the second?",
        "Determine the temporal relation, taking time zones and precision into account.",
        "Compare the two events in absolute time and pick the relation that the timeline supports.",
    ]
    leak_exclude = {"before", "after", "during", "unknown", "conflicting"}
    families = ["deadlines", "time_zones", "relative_dates", "durations", "recurring_schedule", "policy_window",
                "before_after_ordering", "missing_time", "conflicting_sources"]
    challenge_families = ["dst_transition", "date_only_vs_timestamp", "conflicting_timestamps", "long_event_chain"]
    surface_fields = {"timeline": list[str], "question": str, "timezone_context": str | None, "distractor_tags": list[str]}
    verifier_fields = {"event_times": list[str], "question_events": str}
    challenge_note = "Stress time zones, DST, date-only vs timestamp precision and chained relative dates."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        tz1, tz2 = rng.sample(TZS, 2)
        base = date(2026, rng.randint(2, 10), rng.randint(3, 20))
        hh = rng.randint(8, 17)

        def ts(d: date, h: int, m: int = 0) -> str:
            return f"{d.isoformat()}T{h:02d}:{m:02d}"

        def src(when: str | None, tz: str, prec: str = "timestamp", dur: int | None = None, auth: int = 1, **kw: Any) -> dict[str, Any]:
            return {"when": when, "tz": tz, "precision": prec, "duration_min": dur, "authority": auth, **kw}

        A, B = "launch", "review"
        ev: dict[str, Any] = {}
        diff = "easy"
        if family in ("deadlines", "before_after_ordering", "time_zones"):
            tzb = tz2 if family == "time_zones" else tz1
            delta = rng.choice([-30, -3, 2, 40]) * 60
            b_local = datetime.fromisoformat(ts(base, hh)).replace(tzinfo=ZoneInfo(tzb))
            a_local = (b_local + timedelta(minutes=delta)).astimezone(ZoneInfo(tz1))
            ev = {A: {"sources": [src(a_local.replace(tzinfo=None).isoformat(timespec="minutes"), tz1, dur=60)]},
                  B: {"sources": [src(ts(base, hh), tzb, dur=None)]}}
            diff = "medium" if family == "time_zones" else "easy"
        elif family == "durations":
            ev = {A: {"sources": [src(ts(base, hh + 1), tz1, dur=30)]}, B: {"sources": [src(ts(base, hh), tz1, dur=180)]}}
        elif family == "policy_window":
            ev = {A: {"sources": [src(ts(base + timedelta(days=rng.choice([-5, 3, 20])), 10), tz1, dur=30)]},
                  B: {"sources": [src(ts(base, 0), tz1, dur=14 * 24 * 60)]}}
        elif family == "relative_dates":
            off = rng.choice([-2, 3, 10])
            ev = {"kickoff": {"sources": [src(ts(base, 9), tz1, dur=60)]},
                  A: {"sources": [{"anchor": "kickoff", "offset_days": off, "at": "09:00", "tz": tz1, "duration_min": 60, "authority": 1}]},
                  B: {"sources": [src(ts(base + timedelta(days=1), 9), tz1, dur=60)]}}
            diff = "medium"
        elif family == "recurring_schedule":
            n = rng.randint(2, 5)
            first = base
            occ = first + timedelta(weeks=n - 1)
            ev = {A: {"sources": [src(ts(occ, 15), tz1, dur=60)], "recurrence": f"weekly, starting {first.isoformat()}, occurrence #{n}"},
                  B: {"sources": [src(ts(base + timedelta(days=rng.choice([10, 21, 40])), 15), tz1, dur=60)]}}
            diff = "medium"
        elif family == "missing_time":
            ev = {A: {"sources": [src(None, tz1)]}, B: {"sources": [src(ts(base, hh), tz1, dur=60)]}}
        elif family in ("conflicting_sources", "conflicting_timestamps"):
            ev = {A: {"sources": [src(ts(base, hh), tz1, dur=60, auth=2), src(ts(base + timedelta(days=1), hh), tz1, dur=60, auth=2)]},
                  B: {"sources": [src(ts(base + timedelta(days=5), hh), tz1, dur=60)]}}
            diff = "hard" if family == "conflicting_timestamps" else "medium"
        elif family == "dst_transition":
            d = date(2026, 3, 8)  # US spring-forward: 02:00 EST -> 03:00 EDT
            tzn = "America/New_York"
            ev = {A: {"sources": [src(ts(d, 1, 30), tzn, dur=30)]}, B: {"sources": [src(ts(d, 3, 30), tzn, dur=30)]}}
            if rng.random() < 0.5:
                ev = {A: {"sources": [src(ts(d, 3, 30), tzn, dur=30)]}, B: {"sources": [src(ts(d, 1, 30), tzn, dur=30)]}}
            diff = "hard"
        elif family == "date_only_vs_timestamp":
            d = base
            same = rng.random() < 0.6
            ev = {A: {"sources": [src(d.isoformat(), tz1, prec="date")]},
                  B: {"sources": [src(ts(d if same else d + timedelta(days=3), 12), tz1, dur=60)]}}
            diff = "hard"
        elif family == "long_event_chain":
            ev = {"kickoff": {"sources": [src(ts(base, 9), tz1, dur=60)]},
                  "draft": {"sources": [{"anchor": "kickoff", "offset_days": 2, "at": "10:00", "tz": tz1, "duration_min": 60, "authority": 1}]},
                  A: {"sources": [{"anchor": "draft", "offset_days": 3, "at": "09:00", "tz": tz1, "duration_min": 60, "authority": 1}]},
                  B: {"sources": [{"anchor": "kickoff", "offset_days": rng.choice([4, 5, 6]), "at": "09:00", "tz": tz1, "duration_min": 60, "authority": 1}]}}
            diff = "hard"
        return {"events": ev, "question": {"a": A, "b": B}}, {"difficulty": diff, "tags": [family]}

    def decide(self, world: ScenarioWorld) -> Decision:
        lab, why = relate(world.facts)
        return Decision(lab, why)

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        return {"event_times": [f"{n}|{_fmt(s)}" for n, e in f["events"].items() for s in e["sources"]],
                "question_events": f"{f['question']['a']}|{f['question']['b']}"}

    def compare_facts(self, world: ScenarioWorld, extracted: Any) -> Any:
        import re

        d = extracted.model_dump()

        def canon(x: str) -> str:
            x = re.sub(r"^([^|]+)\|unknown\|.*$", r"\1|unknown", x.strip())  # a time zone on an unknown time carries no fact
            return re.sub(r"\+-(\d)", r"-\1", x)

        d["event_times"] = [canon(x) for x in d.get("event_times") or []]
        d["event_times"] = [re.sub(r"^(?:the )?([a-z0-9_]+?)(?: event)?\|", r"\1|", x, flags=re.I) for x in d["event_times"]]
        d["question_events"] = re.sub(r"\b(the|event)\b", "", str(d.get("question_events") or "").lower()).replace(" |", "|").replace("| ", "|").strip() or None
        return super().compare_facts(world, type(extracted).model_validate(d))

    def verifier_context(self, state: dict[str, Any]) -> dict[str, Any]:
        ctx = super().verifier_context(state)
        ctx["timezone_names"] = TZS
        return ctx

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"notes": "Every source entry of every event must appear in the timeline, tagged with its event name exactly as in "
                         "WORLD_FACTS (events 'launch', 'review', ...). Keep all clock times/dates exact in the stated local time zone; "
                         "state the time zone (city or abbreviation) for each entry. Sources with the same authority that disagree must "
                         "both be listed as separate statements. Relative entries must stay relative ('two days after kickoff'). "
                         "The question asks how event 'launch' relates in time to event 'review' without giving the answer. "
                         "timezone_context lists the time zones used."}

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        """The direction of the relation decides the label, so the question is rendered by code (first event -> second event)."""
        q = world.facts["question"]
        return {**surface, "question": f"How does the {q['a']} relate in time to the {q['b']}? (Answer for the {q['a']} relative to the {q['b']}.)"}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"timeline": surface["timeline"], "question": surface["question"], "timezone_context": surface["timezone_context"]}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        lines = []
        for n, e in f["events"].items():
            for s in e["sources"]:
                if "anchor" in s:
                    lines.append(f"{n}: {s['offset_days']} days after {s['anchor']}, at {s['at']} ({s['tz']})")
                elif s["when"] is None:
                    lines.append(f"{n}: date not yet announced")
                elif s["precision"] == "date":
                    lines.append(f"{n}: on {s['when']} (date only, {s['tz']})")
                else:
                    lines.append(f"{n}: {s['when'].replace('T', ' ')} {s['tz']}" + (f" for {s['duration_min']} minutes" if s["duration_min"] else ""))
        return {"timeline": lines, "question": f"How does {f['question']['a']} relate in time to {f['question']['b']}?",
                "timezone_context": ", ".join(sorted({s["tz"] for e in f["events"].values() for s in e["sources"]})), "distractor_tags": []}
