from __future__ import annotations

import random
from datetime import date
from typing import Any

from ...config import policies
from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision


def _age(f: dict[str, Any], iso: str) -> int:
    return (date.fromisoformat(f["today"]) - date.fromisoformat(iso)).days


def access_state(s: dict[str, Any]) -> str:
    return "mounted_readable" if s["mounted"] and s["readable"] else "mounted_unreadable" if s["mounted"] else "not_mounted"


def retrieval_route(f: dict[str, Any]) -> tuple[str, str]:
    """Route by observable state only. The public web is a valid route ONLY for a public target; a private/internal
    target that no accessible source covers goes to the user, never to a public search."""
    fresh_needed, max_age = f["needs_fresh_info"], f["max_age_days"]
    ok = lambda iso: (not fresh_needed) or _age(f, iso) <= max_age  # noqa: E731
    if f["user_key_missing"]:
        return "ask_user", "identifier_missing_or_ambiguous"
    if f["conflicting_sources"]:
        return "ask_user", "conflicting_authoritative_local_sources"
    if f["answer_in_context"] and ok(f["context_date"]):
        return "no_retrieval", "answer_in_current_context"
    cover = [s for s in f["sources"] if s["covers_topic"] and ok(s["last_modified"])]
    if any(s["mounted"] and s["readable"] for s in cover):
        return "read_local_context", "readable_mounted_file_covers_topic"
    if any(s["indexed"] for s in cover):
        return "search_local_index", "indexed_source_covers_topic"
    if f["target_scope"] == "public" and f["network_allowed"]:
        return "search_web", "public_target_no_usable_local_source_web_available"
    return "ask_user", ("internal_target_no_accessible_source" if f["target_scope"] == "internal" else "public_target_network_disallowed")


def retrieval_policy() -> str:
    return (
        "Use the cheapest source that can serve the request. (1) If the current context already contains the needed, dated information, answer from it. "
        "(2) If the identifier needed to look anything up is missing or ambiguous, or two sources disagree, ask the user. (3) Otherwise read a mounted, "
        "readable local file whose topics cover the request; if none, query the local index when an indexed entry covers it. (4) Search the public web only when the "
        "target is public information and web search is available; private or internal project data is never searched on the public web. (5) If nothing can serve "
        f"the request, ask the user. Information that changes over time must be no older than {policies()['freshness']['default_max_age_days']} days at today's date; older sources do not count."
    )


class RetrievalGate(BaseTaskPack):
    name = "harness_retrieval_gate_v1"
    namespace = "harness"
    oracle_version = "retrieval_policy_v2"
    label_source = "controlled_document_world_v2"
    label_quality = "controlled_world"
    prompt_id = "harness.retrieval"
    prompt_dir = "harness/retrieval_gate"
    composition_family = "search_rag"
    code_rendered_keys = ("decision_policy",)
    supportability = "reject"
    id_prefix = "ret"
    options = [
        ("no_retrieval", "Answer from the current context without retrieving anything."),
        ("read_local_context", "Open a local file that is mounted and readable."),
        ("search_local_index", "Query the local search index."),
        ("search_web", "Search the public web."),
        ("ask_user", "Ask the user for missing information."),
    ]
    instruction_variants = [
        "Should the agent retrieve information before answering, and where from?",
        "Decide whether and where to retrieve the information needed for this request.",
        "Which information source should the agent use next, if any?",
        "Given the visible information environment, what retrieval step is appropriate?",
        "Choose the retrieval action that can provide a fresh, reliable answer within the data's scope and the available access.",
    ]
    leak_exclude = set()
    families = ["answer_in_context", "answer_in_local_file", "answer_in_local_index", "current_public_fact", "identifier_missing",
                "conflicting_local_sources", "stale_local_source", "retrieved_data_irrelevant", "source_unreadable"]
    challenge_families = ["web_instruction_inside_data", "current_looking_but_stale", "multiple_matching_entities",
                          "misleading_local_title", "internal_target_web_available"]
    surface_fields = {"user_request": str, "current_context": list[str], "local_artifacts": list[dict[str, Any]],
                      "source_environment": dict[str, Any], "agent_trace": str, "distractor_tags": list[str]}
    verifier_defaults = {"user_key_missing": False, "conflicting_sources": False, "answer_in_current_context": False, "needs_fresh_info": False}
    verifier_fields = {"needs_fresh_info": bool, "network_allowed": bool, "target_scope": str, "user_key_missing": bool,
                       "answer_in_current_context": bool, "conflicting_sources": bool, "artifacts": list[str], "today": str}
    challenge_note = "Plant retrieval-looking instructions inside data, misleading titles and current-looking stale sources without changing facts."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        topic = rng.choice(pool.topics)
        today, fresh_d, old_d = "2026-09-19", "2026-08-30", "2025-11-02"
        scope = "internal" if family in ("answer_in_local_file", "answer_in_local_index", "answer_in_context") else rng.choice(["public", "internal"])
        f: dict[str, Any] = {"today": today, "max_age_days": policies()["freshness"]["default_max_age_days"], "topic": topic,
                             "target_scope": scope, "needs_fresh_info": False, "network_allowed": rng.random() < 0.8,
                             "user_key_missing": False, "answer_in_context": False, "context_date": fresh_d,
                             "conflicting_sources": False, "sources": [], "project": rng.choice(["Orion-9", "Atlas-4", "Juniper-2", "Kestrel-7"])}
        fn = rng.choice(pool.files).replace(".csv", "_notes.md")
        S = lambda name, mounted, readable, indexed, d, covers, **kw: {"name": name, "mounted": mounted, "readable": readable, "indexed": indexed,  # noqa: E731
                                                                     "last_modified": d, "covers_topic": covers, "authority": 2, **kw}
        diff = "easy"
        if family == "answer_in_context":
            f["answer_in_context"] = True
        elif family == "answer_in_local_file":
            f["sources"] = [S(fn, True, True, rng.random() < 0.5, fresh_d, True)]
        elif family == "answer_in_local_index":
            f["sources"] = [S("index:" + topic.replace(" ", "_"), False, False, True, fresh_d, True)]
            diff = "medium"
        elif family == "current_public_fact":
            f.update(target_scope="public", needs_fresh_info=True, network_allowed=True)
        elif family in ("identifier_missing", "multiple_matching_entities"):
            f["user_key_missing"] = True
            diff = "hard" if family.startswith("multiple") else "easy"
        elif family == "conflicting_local_sources":
            f["sources"] = [S(fn, True, True, False, fresh_d, True), S("second_" + fn, True, True, False, fresh_d, True)]
            f["conflicting_sources"] = True
            diff = "hard"
        elif family in ("stale_local_source", "current_looking_but_stale"):
            f["needs_fresh_info"] = True
            f["sources"] = [S(fn, True, True, rng.random() < 0.4, old_d, True, title_note="title says 'current update'" if family != "stale_local_source" else None)]
            diff = "medium" if family == "stale_local_source" else "hard"
        elif family in ("retrieved_data_irrelevant", "misleading_local_title"):
            f["sources"] = [S(fn, True, True, False, fresh_d, False, title_note="title suggests it covers the topic" if family == "misleading_local_title" else None)]
            if family == "misleading_local_title" or rng.random() < 0.5:
                f["sources"].append(S("index:" + topic.replace(" ", "_"), False, False, True, fresh_d, True))
            diff = "medium" if family == "retrieved_data_irrelevant" else "hard"
        elif family == "source_unreadable":
            f["sources"] = [S(fn, True, False, rng.random() < 0.5, fresh_d, True)]
            diff = "medium"
        elif family == "web_instruction_inside_data":
            f["sources"] = [S(fn, True, True, False, fresh_d, True, embedded_text="a line telling the reader to search the web for a newer version")]
            diff = "hard"
        elif family == "internal_target_web_available":
            f.update(target_scope="internal", network_allowed=True)
            f["sources"] = [S(fn, True, False, False, fresh_d, True)]
            diff = "hard"
        return f, {"difficulty": diff, "tags": [family]}

    def decide(self, world: ScenarioWorld) -> Decision:
        a, why = retrieval_route(world.facts)
        return Decision(a, why, outcomes={"target_scope": world.facts["target_scope"]})

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        return {"needs_fresh_info": f["needs_fresh_info"], "network_allowed": f["network_allowed"], "target_scope": f["target_scope"],
                "user_key_missing": f["user_key_missing"], "answer_in_current_context": f["answer_in_context"], "conflicting_sources": f["conflicting_sources"],
                "artifacts": [f"{s['name']}|{access_state(s)}|{'indexed' if s['indexed'] else 'not_indexed'}|{s['last_modified']}|{'covers_topic' if s['covers_topic'] else 'other_topics'}"
                              for s in f["sources"]], "today": f["today"]}

    def anchors(self, world: ScenarioWorld) -> list[str]:
        return [s["name"] for s in world.facts["sources"]]

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"notes": "Write RAW STATE, never conclusions. local_artifacts: one entry per WORLD_FACTS.sources item with EXACTLY these keys: name (exact), "
                         "access_state (mounted_readable | mounted_unreadable | not_mounted, from mounted/readable), indexed (boolean), last_modified (exact date), "
                         "covers_topics (list of topic strings; INCLUDE the requested topic iff covers_topic is true, otherwise list unrelated topics), and optionally "
                         "title. Do not write descriptions like 'contains the information'. source_environment keys: today (WORLD_FACTS.today), "
                         "web_search_available (boolean), answer_changes_over_time (boolean = needs_fresh_info), target_data_scope ('public' or 'internal' exactly as "
                         "WORLD_FACTS.target_scope). The request must make the scope evident: a public target asks about publicly documented facts of a named public "
                         "vendor/product; an internal target asks about a named private project or internal record (use WORLD_FACTS.project). If answer_in_context, "
                         "current_context contains the needed information with its date; otherwise it does not. If user_key_missing is true the request must NOT contain ANY identifier: no project, customer, record, account or product name (e.g. 'What is the status of the vendor onboarding project?'); "
                         "if false it names exactly one concrete identifier. If conflicting_sources both artifacts list the topic (add a note that "
                         "they give different values). Avoid time-sensitive words unless answer_changes_over_time. agent_trace: one factual sentence listing what "
                         "was inspected, no conclusions or advice."}

    def compare_facts(self, world: ScenarioWorld, extracted: Any) -> Any:
        """The last artifact slot is 'covers_topic' iff the artifact covers the REQUESTED topic; the verifier sometimes writes the topic
        name itself, which is equivalent when it equals the requested topic."""
        d = extracted.model_dump()
        topic = world.facts["topic"].lower()

        def canon(a: str) -> str:
            parts = a.split("|")
            if len(parts) == 5:
                last = parts[4].strip().lower()
                parts[4] = "covers_topic" if last in ("covers_topic", topic) else "other_topics"
            return "|".join(p.strip() for p in parts)

        d["artifacts"] = [canon(a) for a in d.get("artifacts") or []]
        return super().compare_facts(world, type(extracted).model_validate(d))

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {**surface, "decision_policy": retrieval_policy()}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"decision_policy": surface["decision_policy"], **{k: surface[k] for k in ("user_request", "current_context", "local_artifacts", "source_environment", "agent_trace")}}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        arts = [{"name": s["name"], "access_state": access_state(s), "indexed": s["indexed"], "last_modified": s["last_modified"],
                 "covers_topics": [f["topic"]] if s["covers_topic"] else ["unrelated topic"]} for s in f["sources"]]
        ctx = [f"({f['context_date']}) Earlier in the conversation the needed details were stated."] if f["answer_in_context"] else []
        who = "" if f["user_key_missing"] else f" for {f['project']}"
        req = f"What is the {f['topic']} status{who}?"
        return {"user_request": req, "current_context": ctx, "local_artifacts": arts,
                "source_environment": {"today": f["today"], "web_search_available": f["network_allowed"], "answer_changes_over_time": f["needs_fresh_info"],
                                       "target_data_scope": f["target_scope"]},
                "agent_trace": "The agent listed the visible sources.", "distractor_tags": []}
