from __future__ import annotations

import random
from typing import Any

from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision

# controlled intent policy: communicative feature (speech act) -> intent label
ACT_TO_INTENT = {
    "asks_question": "request_information", "asks_for_action": "request_action", "reports_defect": "report_problem",
    "gives_status": "provide_update", "proposes_time_change": "schedule_or_reschedule",
    "withdraws_commitment": "cancel_or_decline", "expresses_dissatisfaction": "feedback_or_complaint",
}
ACT_TEXT = {
    "asks_question": "asks a factual question", "asks_for_action": "asks the recipient to do something",
    "reports_defect": "reports that something is broken or malfunctioning", "gives_status": "shares a progress or status update",
    "proposes_time_change": "proposes a new time or asks to move a meeting", "withdraws_commitment": "withdraws from or declines something",
    "expresses_dissatisfaction": "expresses dissatisfaction with a product or service",
}


def intent_of(f: dict[str, Any]) -> tuple[str, str]:
    live = [a["act"] for a in f["acts"] if not a["quoted"]]
    if not live:
        return "other", "no_live_communicative_act"
    prim = f["primary_act"]
    return (ACT_TO_INTENT[prim], f"primary_act:{prim}") if prim in live else (ACT_TO_INTENT[live[0]], f"first_live_act:{live[0]}")


class CommunicationIntent(BaseTaskPack):
    name = "foundation_communication_intent_v1"
    namespace = "foundation"
    oracle_version = "intent_policy_v1"
    label_source = "structured_scenario_intent_policy_v1"
    label_quality = "controlled_world"
    prompt_id = "foundation.intent"
    prompt_dir = "foundation/communication_intent"
    composition_family = "communication_productivity"
    code_checked_features = ("very_short_message",)
    id_prefix = "int"
    options = [
        ("request_information", "The sender wants information."),
        ("request_action", "The sender wants the recipient to do something."),
        ("report_problem", "The sender reports something broken or wrong."),
        ("provide_update", "The sender shares status or news."),
        ("schedule_or_reschedule", "The sender sets or changes a time."),
        ("cancel_or_decline", "The sender cancels or declines."),
        ("feedback_or_complaint", "The sender gives feedback or complains."),
        ("other", "None of the other intents."),
    ]
    instruction_variants = [
        "What is the main communicative intent of this message?",
        "Which intent best describes what the sender is doing in this message?",
        "Classify the primary intent of the sender's message.",
        "Choose the intent that captures the message's main purpose (ignore text quoted from someone else).",
        "Decide what the sender mainly wants to achieve with this message.",
    ]
    leak_exclude = {"other", "request_information", "request_action", "report_problem", "provide_update"}
    families = ["direct_request", "implicit_request", "status_update", "complaint", "scheduling", "cancellation",
                "multi_intent", "terse_chat", "typo_heavy", "formal_email", "problem_report", "information_request"]
    challenge_families = ["polite_complaint", "urgent_information_request", "multi_intent_primary_last",
                          "quoted_other_sender", "controlled_ambiguity"]
    surface_fields = {"message": str, "channel": str, "context": list[str], "metadata": dict[str, Any], "distractor_tags": list[str]}
    verifier_fields = {"live_acts": list[str], "quoted_acts": list[str], "primary_act": str, "surface_features": list[str]}
    challenge_note = "Blend tones and phrasing (polite complaints, urgency, quoted text) without changing the acts or which one is primary."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        acts_all = list(ACT_TO_INTENT)
        single = {"direct_request": "asks_for_action", "implicit_request": "asks_for_action", "status_update": "gives_status",
                  "complaint": "expresses_dissatisfaction", "scheduling": "proposes_time_change", "cancellation": "withdraws_commitment",
                  "problem_report": "reports_defect", "information_request": "asks_question", "polite_complaint": "expresses_dissatisfaction"}
        style = {"formality": rng.choice(["casual", "neutral", "formal"]), "typos": False, "length": "medium", "indirect": False}
        f: dict[str, Any] = {"channel": rng.choice(["email", "chat", "ticket", "note"]), "topic": rng.choice(pool.topics),
                             "sender": pool.person(rng), "style": style, "acts": [], "primary_act": "", "ambiguity": None, "phrasing_note": None}
        diff = "easy"
        if family in single:
            a = single[family]
            f["acts"], f["primary_act"] = [{"act": a, "quoted": False}], a
            if family == "implicit_request":
                style["indirect"] = True
                diff = "medium"
            if family == "polite_complaint":
                f["phrasing_note"] = "phrased politely as a request, but the substance is dissatisfaction"
                diff = "hard"
        elif family == "terse_chat":
            a = rng.choice(acts_all)
            f["acts"], f["primary_act"], f["channel"] = [{"act": a, "quoted": False}], a, "chat"
            style.update(length="very short", formality="casual")
        elif family == "typo_heavy":
            a = rng.choice(acts_all)
            f["acts"], f["primary_act"] = [{"act": a, "quoted": False}], a
            style["typos"] = True
            diff = "medium"
        elif family == "formal_email":
            a = rng.choice(acts_all)
            f["acts"], f["primary_act"], f["channel"] = [{"act": a, "quoted": False}], a, "email"
            style["formality"] = "formal"
        elif family in ("multi_intent", "multi_intent_primary_last", "urgent_information_request"):
            a, b = rng.sample(acts_all, 2)
            if family == "urgent_information_request":
                a, b = "asks_question", "gives_status"
                f["phrasing_note"] = "the question is urgent"
            f["acts"] = [{"act": a, "quoted": False}, {"act": b, "quoted": False}]
            f["primary_act"] = b if family == "multi_intent_primary_last" else a
            f["order_note"] = "the primary act comes " + ("last" if family == "multi_intent_primary_last" else "first")
            diff = "medium" if family == "multi_intent" else "hard"
        elif family == "quoted_other_sender":
            a, b = rng.sample(acts_all, 2)
            f["acts"] = [{"act": a, "quoted": False}, {"act": b, "quoted": True}]
            f["primary_act"] = a
            diff = "hard"
        elif family == "controlled_ambiguity":
            a, b = rng.sample(acts_all, 2)
            f["acts"] = [{"act": a, "quoted": False}, {"act": b, "quoted": False}]
            f["primary_act"], f["ambiguity"] = a, {"alternative_act": b}
            f["phrasing_note"] = "the two purposes are equally weighted; do not make either one clearly dominant"
            diff = "hard"
        f["required_surface_features"] = {"typo_heavy": ["typos_or_abbreviations"], "terse_chat": ["very_short_message"]}.get(family, [])
        return f, {"difficulty": diff, "tags": [family], "adjudicated": family == "controlled_ambiguity"}

    def decide(self, world: ScenarioWorld) -> Decision:
        lab, why = intent_of(world.facts)
        f = world.facts
        if f["ambiguity"]:
            alt = ACT_TO_INTENT[f["ambiguity"]["alternative_act"]]
            return Decision(lab, "controlled_ambiguity_primary_policy", acceptable=[lab, alt], quality="adjudicated")
        return Decision(lab, why)

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        return {"live_acts": [a["act"] for a in f["acts"] if not a["quoted"]], "quoted_acts": [a["act"] for a in f["acts"] if a["quoted"]],
                "primary_act": f["primary_act"]}  # surface_features are checked separately (subset test, see check_surface_features)

    def compare_facts(self, world: ScenarioWorld, extracted: Any) -> Any:
        # `primary_act` is a judgement call for controlled-ambiguity messages: not verifiable there
        c = super().compare_facts(world, extracted)
        got = extracted.model_dump()
        # Natural messages carry implied secondary acts ("Can we move it?" also asks a question): tolerate EXTRA live acts as long as every
        # world act is present; the label depends on the primary act, which is still verified strictly below.
        if "live_acts" in c.mismatched and set(self.expected_extraction(world)["live_acts"]) <= set(got.get("live_acts") or []):
            c.mismatched.remove("live_acts")
            c.matched.append("live_acts")
        self.check_surface_features(world, got, c)
        if world.facts["ambiguity"] and "primary_act" in c.mismatched:
            c.mismatched.remove("primary_act")
            c.matched.append("primary_act")
        return c

    def verifier_context(self, state: dict[str, Any]) -> dict[str, Any]:
        ctx = super().verifier_context(state)
        ctx["act_vocabulary"] = ACT_TEXT
        return ctx

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        d = {a: ACT_TEXT[a] for a in ACT_TO_INTENT}
        return {"notes": "WORLD_FACTS.acts lists the communicative acts in the message (a 'quoted' act appears only inside text quoted from "
                         "ANOTHER sender, e.g. a forwarded or replied-to message, and is not the sender's own act). primary_act is the sender's "
                         f"main purpose. Act meanings: {d}. Never name the act or intent explicitly. Put the message text in `message`. If WORLD_FACTS.style.typos is true, or the "
                         "message must be typo heavy, use several real misspellings, dropped letters and abbreviations ('u', 'thx', 'calender') while keeping "
                         "the meaning unambiguous."}

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        """Only whitelisted metadata reaches the state: the generator sometimes copies hidden WORLD_FACTS keys (primary_act, ...) into it."""
        md = {k: v for k, v in (surface.get("metadata") or {}).items() if k in ("sender", "subject", "topic")}
        return {**surface, "metadata": md}

    def check_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> list[str]:
        if "very_short_message" in world.facts.get("required_surface_features", []) and len(str(surface.get("message", "")).split()) > 14:
            return ["fact_mismatch:surface_features:very_short_message"]  # objective: a terse chat message is at most ~14 words
        return []

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"message": surface["message"], "channel": surface["channel"], "context": surface["context"], "metadata": surface["metadata"]}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        t = f["topic"]
        line = {"asks_question": f"Do you know what the current plan for the {t} is?", "asks_for_action": f"Please send me the {t} summary today.",
                "reports_defect": f"The {t} page is showing an error every time I open it.", "gives_status": f"Quick note: the {t} is on track for this week.",
                "proposes_time_change": f"Could we move our {t} sync to Thursday afternoon?", "withdraws_commitment": f"I have to pull out of the {t} session, sorry.",
                "expresses_dissatisfaction": f"Honestly the {t} experience has been disappointing."}
        parts = [(line[a["act"]] if not a["quoted"] else f"> Earlier they wrote: {line[a['act']]}") for a in f["acts"]]
        return {"message": " ".join(parts), "channel": f["channel"], "context": [], "metadata": {"sender": f["sender"]}, "distractor_tags": []}
