from __future__ import annotations

import random
from typing import Any

from ...config import policies
from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision


def rubric_text() -> str:
    u = policies()["urgency"]
    pts = lambda d: ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in d.items())  # noqa: E731
    dl = "; ".join(f"a real deadline within {h} hours {p}" for h, p in u["deadline_hour_cutoffs"])
    lo, mid, hi = u["level_cutoffs"]
    return (f"Score urgency from operational impact, not tone. Add points: scope ({pts(u['scope_points'])}); service ({pts(u['service_points'])}); "
            f"revenue impact ({pts(u['revenue_points'])}); {dl}, later or none 0 (a deadline the message itself says is not binding counts 0); "
            f"workaround ({pts(u['workaround_points'])}). Any risk of physical harm is critical (3) regardless of points. Level: total <= {lo} is low (0); "
            f"{lo + 1}-{mid} normal (1); {mid + 1}-{hi} high (2); above {hi} critical (3). If the message does not say who or what is affected, the level is exactly "
            f"{u['missing_info_level']} (normal).")


def bucket(hours: float | None) -> str:
    if hours is None:
        return "none"
    return "under_4h" if hours <= 4 else "4_to_24h" if hours <= 24 else "24_to_72h" if hours <= 72 else "over_72h"


def urgency_level(f: dict[str, Any]) -> tuple[int, str]:
    p = policies()["urgency"]
    if f["safety_impact"]:
        return 3, "safety_impact_override"
    pts = p["scope_points"][f["scope"]] + p["service_points"][f["service_state"]] + p["revenue_points"][f["revenue_impact"]]
    hrs = f["deadline_hours"] if f["deadline_is_real"] else None
    if hrs is not None:
        for cutoff, pt in p["deadline_hour_cutoffs"]:
            if hrs <= cutoff:
                pts += pt
                break
    pts += p["workaround_points"][f["workaround"]]
    lo, mid, hi = p["level_cutoffs"]
    lvl = 0 if pts <= lo else 1 if pts <= mid else 2 if pts <= hi else 3
    if f["missing_impact_info"]:
        lvl = p["missing_info_level"]  # exactly normal: with unstated impact the scope cannot matter (derivable from the visible rubric)
    return lvl, f"points:{pts}"


class CommunicationUrgency(BaseTaskPack):
    name = "foundation_communication_urgency_v1"
    namespace = "foundation"
    decision_type = "score"
    oracle_version = "urgency_policy_v1"
    label_source = "declared_urgency_policy_v1"
    label_quality = "controlled_world"
    prompt_id = "foundation.urgency"
    prompt_dir = "foundation/communication_urgency"
    composition_family = "communication_productivity"
    code_rendered_keys = ("decision_policy",)
    supportability = "reject"
    id_prefix = "urg"
    levels = [(0, "Low: no time pressure or impact."), (1, "Normal: routine handling."), (2, "High: prompt attention needed."),
              (3, "Critical: immediate action needed.")]
    instruction_variants = [
        "How urgent is this message operationally?",
        "Rate the operational urgency of the message on the rubric.",
        "Score the message's urgency using the levels provided; judge impact, not tone.",
        "Assign an urgency level based on scope, deadline, impact and available workarounds.",
        "What urgency level should this message be handled at?",
    ]
    families = ["broad_outage", "single_user_issue", "deadline_driven", "safety_report", "revenue_loss", "workaround_available",
                "partial_outage", "missing_info", "routine_request"]
    challenge_families = ["emotional_low_impact", "calm_critical_outage", "false_deadline", "workaround_reduces", "partial_outage_hard", "missing_impact_info_hard"]
    surface_fields = {"message": str, "channel": str, "context": list[str], "metadata": dict[str, Any], "distractor_tags": list[str]}
    verifier_defaults = {"revenue_impact": "none", "deadline_bucket": "none", "deadline_is_real": True, "safety_impact": False, "workaround": "none", "missing_impact_info": False}
    verifier_fields = {"scope": str, "service_state": str, "revenue_impact": str, "workaround": str, "safety_impact": bool,
                       "deadline_bucket": str, "deadline_is_real": bool, "missing_impact_info": bool}
    challenge_note = "Decouple emotional tone from real impact (angry-but-minor, calm-but-critical); the impact features must stay as given."

    def probability_semantics(self):  # type: ignore[no-untyped-def]
        s = super().probability_semantics()
        s.description = "Ordinal rubric level from a declared points policy; neighbouring levels are closer than distant ones."
        return s

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        f: dict[str, Any] = {"scope": "single", "affected_users": 1, "service_state": "normal", "revenue_impact": "none",
                             "workaround": "none", "safety_impact": False, "deadline_hours": None, "deadline_is_real": True,
                             "explicit_escalation": False, "missing_impact_info": False, "emotional_intensity": "low",
                             "topic": rng.choice(pool.topics), "sender": pool.person(rng), "channel": rng.choice(["email", "chat", "ticket"])}
        diff = "medium"
        R = rng.choice
        if family == "broad_outage":
            f.update(scope=R(["org", "all_customers"]), service_state="down", revenue_impact=R(["minor", "major"]))
        elif family == "single_user_issue":
            f.update(scope="single", service_state=R(["normal", "degraded"]), workaround=R(["none", "partial"]))
            diff = "easy"
        elif family == "deadline_driven":
            f.update(scope=R(["single", "team"]), deadline_hours=R([2, 8, 20, 60, 200]))
        elif family == "safety_report":
            f.update(safety_impact=True, scope=R(["single", "team", "org"]))
        elif family == "revenue_loss":
            f.update(scope="team", revenue_impact=R(["minor", "major"]), service_state=R(["degraded", "down"]))
        elif family == "workaround_available":
            f.update(scope=R(["team", "org"]), service_state=R(["degraded", "down"]), workaround="full")
        elif family == "partial_outage":
            f.update(scope="team", service_state="degraded", workaround="partial", revenue_impact=R(["none", "minor"]))
        elif family == "missing_info":
            f.update(missing_impact_info=True, scope=R(["team", "org"]), service_state=R(["degraded", "down"]))
        elif family == "routine_request":
            f.update(scope=R(["single", "team"]), deadline_hours=R([None, 200]))
            diff = "easy"
        elif family == "emotional_low_impact":
            f.update(scope="single", emotional_intensity="high", explicit_escalation=True, workaround=R(["none", "full"]))
        elif family == "calm_critical_outage":
            f.update(scope="all_customers", service_state="down", revenue_impact="major", emotional_intensity="calm")
        elif family == "false_deadline":
            f.update(scope="single", deadline_hours=R([1, 3]), deadline_is_real=False)
        elif family == "workaround_reduces":
            f.update(scope="org", service_state="down", revenue_impact="minor", workaround="full")
        elif family == "partial_outage_hard":
            f.update(scope="org", service_state="degraded", revenue_impact="minor", workaround="partial")
        elif family == "missing_impact_info_hard":
            f.update(missing_impact_info=True, scope="team", explicit_escalation=True)
        if f["scope"] != "single":
            f["affected_users"] = {"team": rng.randint(4, 30), "org": rng.randint(50, 900), "all_customers": rng.randint(2000, 50000)}[f["scope"]]
        return f, {"difficulty": diff, "tags": [family]}

    def decide(self, world: ScenarioWorld) -> Decision:
        lvl, why = urgency_level(world.facts)
        return Decision(lvl, why)

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        exp = {"scope": f["scope"], "service_state": f["service_state"], "revenue_impact": f["revenue_impact"], "workaround": f["workaround"],
                "safety_impact": f["safety_impact"], "deadline_bucket": bucket(f["deadline_hours"]), "deadline_is_real": f["deadline_is_real"],
                "missing_impact_info": f["missing_impact_info"]}
        if f["missing_impact_info"]:
            exp.pop("scope")  # deliberately unstated in the message
        return exp

    def verifier_context(self, state: dict[str, Any]) -> dict[str, Any]:
        return super().verifier_context(state)

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"notes": "scope: single (one person), team (a small group), org (a whole organisation), all_customers (the public). "
                         "affected_users is the exact number of affected people. service_state: normal | degraded | down. "
                         "revenue_impact: none | minor | major. workaround: none | partial | full. deadline_hours: hours until a stated "
                         "deadline (null = no deadline); if deadline_is_real is false the message states a deadline that the context "
                         "makes clearly non-binding (e.g. 'no rush from management'). missing_impact_info: the sender does not say who or "
                         "what is affected. emotional_intensity is tone only and does NOT change impact. Write a natural, human message of 2-5 sentences in the requested channel style (not a status line). Every non-default feature MUST be stated in the text: who is affected (and how many), the service state, any revenue effect, any workaround (or that there is none), any deadline (with the time left), any safety concern; if missing_impact_info is true the message must NOT say who or what is affected. Express features in everyday language ('the portal is down', 'we can work around it by hand', 'about 30 of us are stuck', 'the client call is at 3pm'), never as a checklist and never with literal phrases like 'service state', 'revenue impact' or 'workaround status'; do NOT mention features that are at their default (no revenue effect, normal service, no deadline, no safety issue) unless it is natural to. Do not put feature values in metadata. Never state the urgency level."}

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        md = {k: v for k, v in (surface.get("metadata") or {}).items() if k in ("sender", "subject", "topic")}
        return {**surface, "metadata": md, "decision_policy": rubric_text()}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"decision_policy": surface["decision_policy"], "message": surface["message"], "channel": surface["channel"], "context": surface["context"], "metadata": surface["metadata"]}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        who = {"single": "just me", "team": f"about {f['affected_users']} people on my team", "org": f"roughly {f['affected_users']} people across the company",
               "all_customers": f"about {f['affected_users']} customers"}[f["scope"]]
        parts = [f"The {f['topic']} system is {'completely down' if f['service_state'] == 'down' else 'running slowly' if f['service_state'] == 'degraded' else 'behaving oddly'}."]
        if not f["missing_impact_info"]:
            parts.append(f"This affects {who}.")
        if f["revenue_impact"] != "none":
            parts.append(f"We are losing {'a lot of' if f['revenue_impact'] == 'major' else 'a little'} revenue.")
        if f["workaround"] != "none":
            parts.append(f"There is {'a full' if f['workaround'] == 'full' else 'a partial'} workaround.")
        if f["safety_impact"]:
            parts.append("Someone could get hurt if this continues.")
        if f["deadline_hours"] is not None:
            parts.append(f"The deadline is in {f['deadline_hours']} hours." + ("" if f["deadline_is_real"] else " (No rush from management, though.)"))
        return {"message": " ".join(parts), "channel": f["channel"], "context": [], "metadata": {"sender": f["sender"]}, "distractor_tags": []}
