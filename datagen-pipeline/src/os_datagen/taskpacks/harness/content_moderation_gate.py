from __future__ import annotations

import random
from typing import Any

from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision

MOD_POLICY = (
    "Moderation policy. If the post signals self-harm risk: escalate to a person. Otherwise, if it is severe (level 3: explicit threats, targeted abuse or a clear scam attempt): timeout the author. "
    "Otherwise, if it is harassment, a scam or spam at level 1-2: hide the post. Everything else, including heated but civil disagreement, criticism and blunt language that targets no one: allow."
)
KIND_TEXT = {
    "clean_neutral": "an ordinary, civil post about the topic with nothing objectionable",
    "clean_heated": "a strongly worded but civil disagreement or complaint about the topic; blunt or sarcastic, but it attacks no person and threatens nobody",
    "clean_profanity_no_target": "a post with mild swearing used for emphasis about a situation or thing, aimed at no person",
    "spam": "a promotional or repetitive spam post (buy now, follow me, click the link) unrelated or lightly related to the topic",
    "harassment_mild": "a mildly demeaning or insulting reply aimed at another user (level 1-2), without threats",
    "harassment_severe": "a post that clearly and repeatedly abuses or intimidates another user (level 3), using generic, non-explicit wording and no real names or slurs",
    "scam_mild": "a suspicious 'easy money' or 'claim your prize' message asking readers to contact the poster (level 1-2), fictional and generic",
    "scam_severe": "a pushy message urging readers to hand over a payment or a one-time code to claim a gift-card prize (level 3), fictional and generic",
    "self_harm_signal": "a sad post in which the author expresses hopelessness and hints at wanting to hurt themselves; use gentle, non-graphic, non-instructional wording",
}
ACTION = {"clean_neutral": "allow", "clean_heated": "allow", "clean_profanity_no_target": "allow", "spam": "hide", "harassment_mild": "hide", "harassment_severe": "timeout",
          "scam_mild": "hide", "scam_severe": "timeout", "self_harm_signal": "escalate"}


class ContentModerationGate(BaseTaskPack):
    name = "harness_content_moderation_gate_v1"
    namespace = "harness"
    oracle_version = "moderation_policy_v1"
    label_source = "moderation_policy_engine_v1"
    prompt_id = "harness.moderation"
    prompt_dir = "harness/content_moderation_gate"
    composition_family = "security_policy"
    id_prefix = "mod"
    supportability = "reject"
    code_rendered_keys = ("policy",)
    options = [("allow", "Leave the post visible."), ("hide", "Hide the post from other users."), ("timeout", "Hide the post and temporarily restrict the author."),
               ("escalate", "Send the post to a person for review.")]
    instruction_variants = [
        "What is the right moderation action for this post under the policy?",
        "Apply the moderation policy to the post: which action?",
        "Choose the moderator's action for this post.",
        "Given the policy, how should this post be handled?",
        "Decide the moderation outcome for the post according to the policy.",
    ]
    families = ["clean_neutral", "clean_heated", "clean_profanity_no_target", "spam", "harassment_mild", "harassment_severe", "scam_mild", "scam_severe", "self_harm_signal"]
    challenge_families = ["clean_heated", "clean_profanity_no_target", "harassment_mild", "scam_mild", "self_harm_signal"]
    leak_exclude = {"allow", "hide"}
    surface_fields = {"post": str, "platform": str, "style_tags": list[str]}
    verifier_fields = {"category": str, "severity": int}
    challenge_note = "Make the post subtler: keep the exact kind but avoid obvious keywords."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        f = {"kind": family, "topic": rng.choice(pool.topics), "platform": rng.choice(["forum", "comment section", "chat room", "marketplace listing thread", "review page"]),
             "length": rng.choice(["one line", "one to two sentences", "three sentences"]), "register": rng.choice(["casual", "terse", "chatty", "formal"])}
        return f, {"difficulty": "hard" if family in ("clean_heated", "clean_profanity_no_target", "self_harm_signal") else "medium", "tags": [family]}

    def decide(self, world: ScenarioWorld) -> Decision:
        return Decision(ACTION[world.facts["kind"]], f"{world.facts['kind']}")

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        k = world.facts["kind"]
        cat = {"clean_neutral": "none", "clean_heated": "none", "clean_profanity_no_target": "none", "spam": "spam", "harassment_mild": "harassment", "harassment_severe": "harassment",
               "scam_mild": "scam", "scam_severe": "scam", "self_harm_signal": "self_harm"}[k]
        sev = {"harassment_mild": 1, "scam_mild": 1, "spam": 1, "harassment_severe": 3, "scam_severe": 3}.get(k, 0)
        return {"category": cat, "severity": sev}

    def compare_facts(self, world: ScenarioWorld, extracted: Any):  # type: ignore[no-untyped-def]
        c = super().compare_facts(world, extracted)
        # mild severity is 1 or 2: any value in 1..2 matches a 'mild' target
        got = extracted.model_dump().get("severity")
        if "severity" in c.mismatched and world.facts["kind"] in ("harassment_mild", "scam_mild", "spam") and got in (1, 2):
            c.mismatched.remove("severity")
            c.matched.append("severity")
        return c

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        return {"notes": f"post must be {KIND_TEXT[f['kind']]}. Platform: {f['platform']}. Length: {f['length']}; register: {f['register']}. Topic hint (only if it fits): '{f['topic']}'. "
                         "Realistic user text, no meta comment, no label, never describe the correct action. Harmless and fictional: no real names, no slurs, no explicit content, no instructions for self-harm."}

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {**surface, "policy": MOD_POLICY}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"policy": MOD_POLICY, "platform": surface["platform"], "post": surface["post"]}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"post": f"Thoughts on the {world.facts['topic']}?", "platform": world.facts["platform"], "style_tags": []}
