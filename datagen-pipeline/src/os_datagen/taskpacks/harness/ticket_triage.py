from __future__ import annotations

import random
from typing import Any

from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision

QUEUES = {
    "billing": "an invoice, charge, refund, payment method or subscription-price problem",
    "technical": "a product malfunction, error message, outage, login problem or how-to-fix request",
    "sales": "a prospective customer asking about pricing tiers, demos, quotes, upgrades or bulk purchases",
    "spam": "unsolicited promotion, phishing-style or bulk junk that has nothing to do with the company's service",
    "other": "a legitimate request that fits none of billing, technical or sales, such as a press enquiry, partnership proposal, job application or general feedback",
}


class TicketTriage(BaseTaskPack):
    name = "harness_ticket_triage_v1"
    namespace = "harness"
    oracle_version = "ticket_queue_construction_v1"
    label_source = "constructed_queue_v1"
    prompt_id = "harness.ticket"
    prompt_dir = "harness/ticket_triage"
    composition_family = "communication_productivity"
    id_prefix = "tkt"
    supportability = "reject"
    options = [("billing", "Route to the billing team."), ("technical", "Route to technical support."), ("sales", "Route to the sales team."),
               ("spam", "Discard as spam."), ("other", "Route to the general inbox.")]
    instruction_variants = [
        "Which queue should this message be routed to?",
        "Route the incoming email to the right team.",
        "Pick the queue that should handle this ticket.",
        "Decide where this customer message belongs.",
        "Triage the message: which queue owns it?",
    ]
    families = ["plain", "mixed_signal", "short_terse", "angry_tone", "forwarded_thread", "signature_noise"]
    challenge_families = ["mixed_signal", "short_terse", "forwarded_thread"]
    leak_exclude = {"other", "sales", "spam", "billing", "technical"}
    surface_fields = {"subject": str, "body": str, "style_tags": list[str]}
    verifier_fields = {"queue": str}
    challenge_note = "Make the message ambiguous on the surface (mentions other departments in passing) while its main request stays exactly as given."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        q = rng.choice(list(QUEUES))
        f = {"queue": q, "family": family, "product": rng.choice(pool.objects), "topic": rng.choice(pool.topics), "length": rng.choice(["one to two sentences", "three to five sentences", "a short paragraph"])}
        return f, {"difficulty": "hard" if family in ("mixed_signal", "forwarded_thread") else "medium", "tags": [family, q]}

    def decide(self, world: ScenarioWorld) -> Decision:
        return Decision(world.facts["queue"], f"queue:{world.facts['queue']}")

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"queue": world.facts["queue"]}

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        styles = {"plain": "a plain message", "mixed_signal": "a message whose main request is the given kind but which mentions another department in passing (e.g. a technical issue that also mentions being charged last month)",
                  "short_terse": "a very short, terse message with no greeting", "angry_tone": "an irritated or frustrated message", "forwarded_thread": "a forwarded email thread where the actual request is in the latest message",
                  "signature_noise": "a message ending with a long signature and legal disclaimer"}
        return {"notes": f"Write an email to a company's shared inbox. Its MAIN request is: {QUEUES[f['queue']]}. Style: {styles[f['family']]}. Length: {f['length']}. Product or subject hint (only if it fits): "
                         f"'{f['product']}'. Fictional, generic, no real companies or people. Never name a queue or team as the answer and never describe the correct routing."}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"subject": surface["subject"], "body": surface["body"]}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"subject": f"About the {world.facts['product']}", "body": f"Hello, I need help with the {world.facts['product']}.", "style_tags": []}
