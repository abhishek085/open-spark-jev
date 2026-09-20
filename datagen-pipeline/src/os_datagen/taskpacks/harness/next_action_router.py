from __future__ import annotations

import random
from typing import Any

from ...config import policies
from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision

TOOL_DESC = {
    "use_python": "Run local Python code over workspace files.",
    "call_small_model": "Generate a brief natural-language response.",
    "call_large_model": "Generate a longer reasoning response.",
    "search_web": "Search public web sources.",
    "ask_user": "Ask the user for missing information.",
    "use_cache": "Reuse a previously stored result.",
}


COSTS = {"use_cache": 0, "use_python": 1, "ask_user": 2, "call_small_model": 4, "search_web": 5, "call_large_model": 20}
ROUTER_POLICY = (
    "Pick the cheapest action (cost_units are listed with each tool) that can validly satisfy the request. Rules: (1) A cached result is "
    "usable only if its key matches the request, its age is within the ttl, and its checksum is verified; a stale or mismatched entry is not usable. "
    "(2) If the requested action needs inputs that are listed absent, ask the user. (3) If the answer depends on information that changes over time, "
    "search the web when network access is allowed, otherwise ask the user. (4) If the answer can be computed exactly from a local file, or a mounted "
    "document already states it, run local Python. (5) Otherwise use the small model for a task of low complexity and the large model for a task of high "
    "complexity. (6) An action listed as disallowed, or one that already failed twice, cannot be used; take the next cheapest valid action."
)


def route(f: dict[str, Any]) -> tuple[str, str, str | None]:
    """Ordered routing policy -> (preferred, reason, fallback_from)."""
    c = f["cache"]
    cache_valid = c["available"] and c["key_matches"] and c["age_days"] <= c["ttl_days"] and c["artifact_validated"]
    if cache_valid:
        branch, pref = "valid_cache", ["use_cache"]
    elif not f["user_information_complete"]:
        branch, pref = "missing_essential_information", ["ask_user"]
    elif f["needs_fresh_info"]:
        branch, pref = "fresh_public_information", ["search_web", "ask_user"]
    elif f["local_context_contains_answer"] or (f["must_be_exact"] and f["available_files"]):
        branch, pref = "exact_local_computation_available", ["use_python", "ask_user"]
    elif f["task_complexity"] == "high":
        branch, pref = "complex_reasoning", ["call_large_model", "call_small_model", "ask_user"]
    else:
        branch, pref = "simple_transformation", ["call_small_model", "call_large_model", "ask_user"]
    blocked = set(f["disallowed_actions"])
    if not f["network_allowed"]:
        blocked.add("search_web")
    pf = f.get("prior_failed")
    if pf and pf["times"] >= 2:
        blocked.add(pf["action"])
    for i, a in enumerate(pref):
        if a not in blocked:
            return a, branch if i == 0 else f"constraint_fallback:{branch}", pref[0] if i else None
    return "ask_user", f"constraint_fallback:{branch}", pref[0]


class NextActionRouter(BaseTaskPack):
    name = "harness_next_action_router_v1"
    namespace = "harness"
    oracle_version = "next_action_rules_v1.0"
    label_source = "deterministic_oracle"
    prompt_id = "harness.next_action"
    prompt_dir = "harness/next_action_router"
    composition_family = "agent_harness"
    code_rendered_keys = ("decision_policy",)
    supportability = "reject"
    id_prefix = "nar"
    options = [
        ("use_cache", "Reuse an existing validated result."),
        ("use_python", "Compute from available local files with Python."),
        ("call_small_model", "Ask the small local model to answer."),
        ("call_large_model", "Ask the larger local model to answer."),
        ("search_web", "Search the public web."),
        ("ask_user", "Ask the user for missing information."),
    ]
    instruction_variants = [
        "Choose the lowest-cost valid next action that can satisfy the request under the stated constraints.",
        "Which next step is cheapest while still valid under the constraints?",
        "Select the least expensive action that can correctly satisfy the request and respects the limits.",
        "Given the state, what is the lowest-cost valid next move?",
        "Decide how the agent should proceed: cheapest valid option only.",
    ]
    families = ["exact_local_calculation", "valid_cache", "local_context_answer", "fresh_public_information",
                "missing_essential_information", "complex_reasoning", "simple_formatting", "constraint_conflict"]
    challenge_families = ["stale_cache", "cache_collision", "search_bait", "network_disallowed",
                          "ambiguous_request", "generic_guidance_request", "repeated_failed_tool", "misleading_tool_description", "unknown_tool_name"]
    allowed_distractors = ["unrelated_workspace_file", "verbose_log_line", "irrelevant_user_preference", "harmless_typo"]
    surface_fields = {"user_request": str, "agent_trace": str, "tool_descriptions": list[dict[str, Any]],
                      "visible_constraints": dict[str, Any], "distractor_tags": list[str], "self_check": dict[str, Any]}
    verifier_defaults = {"needs_fresh_info": False, "local_context_contains_answer": False}
    verifier_fields = {"network_allowed": bool, "must_be_exact": bool, "available_files": list[str],
                       "cache_usable": bool, "user_information_complete": bool, "task_mode": str, "needs_fresh_info": bool,
                       "local_context_contains_answer": bool, "task_complexity": str}
    challenge_note = "Add misleading cues (bait phrases, stale or colliding cache, odd tools) without changing facts."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        csv = rng.choice(pool.files)
        f: dict[str, Any] = {
            "goal_type": "general_task", "available_files": [], "csv": None,
            "cache": {"available": False, "key_matches": False, "age_days": 0, "ttl_days": policies()["router"]["cache_ttl_days_default"], "artifact_validated": False},
            "network_allowed": rng.random() < 0.7, "must_be_exact": False, "user_information_complete": True,
            "task_complexity": "low", "agent_stage": rng.choice(["initial", "after_workspace_scan"]),
            "needs_fresh_info": False, "local_context_contains_answer": False, "disallowed_actions": [],
            "prior_failed": None, "request_mentions_web_search": False, "misleading_tool": None, "extra_unlisted_tool": None,
            "topic": rng.choice(pool.topics),
            "task_mode": "answer_question", "required_fields": [], "known_fields": {},
        }
        cache_ok = {"available": True, "key_matches": True, "age_days": rng.randint(1, 10), "ttl_days": 30, "artifact_validated": True}
        diff, tags = "easy", [family]

        def with_csv() -> None:
            f["available_files"] = [csv]
            f["csv"] = {"file": csv, "column": rng.choice(["revenue", "amount", "quantity"]), "rows_seed": rng.randint(1, 10**6)}
            f["goal_type"] = "exact_csv_aggregation"
            f["must_be_exact"] = True

        if family == "exact_local_calculation":
            with_csv()
        elif family == "valid_cache":
            f["cache"] = cache_ok
            f["goal_type"] = "repeat_of_earlier_request"
            if rng.random() < 0.5:
                with_csv()
        elif family == "local_context_answer":
            f["available_files"] = ["README.md"]
            f["local_context_contains_answer"] = True
            f["goal_type"] = "lookup_in_mounted_document"
        elif family == "fresh_public_information":
            f.update(needs_fresh_info=True, network_allowed=True, goal_type="current_public_fact")
        elif family in ("missing_essential_information", "ambiguous_request"):
            req = ["legal_name", "tax_id", "remittance_address", "primary_contact"]
            absent = rng.sample(req, rng.randint(2, 4))
            f.update(user_information_complete=False, task_mode="execute_record_creation", required_fields=req,
                     known_fields={k: (None if k in absent else {"legal_name": rng.choice(pool.orgs), "tax_id": f"TX-{rng.randint(100000, 999999)}",
                                                         "remittance_address": f"{rng.randint(10, 99)} {rng.choice(pool.last_names)} Street, {rng.choice(pool.cities)}",
                                                         "primary_contact": pool.person(rng)}[k]) for k in req},
                     goal_type="create_vendor_record" if family == "missing_essential_information" else "underspecified_record_request")
            if family == "ambiguous_request":
                diff = "hard"
        elif family == "complex_reasoning":
            f.update(task_complexity="high", goal_type="multi_step_analysis")
            diff = "medium"
        elif family == "simple_formatting":
            f.update(task_complexity="low", goal_type="rewrite_supplied_text")
        elif family == "constraint_conflict":
            if rng.random() < 0.5:
                f.update(task_complexity="high", goal_type="multi_step_analysis", disallowed_actions=["call_large_model"])
            else:
                f.update(needs_fresh_info=True, network_allowed=False, goal_type="current_public_fact")
            diff = "hard"
        elif family == "stale_cache":
            with_csv()
            f["cache"] = {**cache_ok, "age_days": rng.randint(45, 200)}
            diff = "hard"
        elif family == "cache_collision":
            with_csv()
            f["cache"] = {**cache_ok, "key_matches": False}
            diff = "hard"
        elif family == "search_bait":
            f.update(available_files=["README.md"], local_context_contains_answer=True, request_mentions_web_search=True,
                     network_allowed=True, goal_type="lookup_in_mounted_document")
            diff = "hard"
        elif family == "network_disallowed":
            f.update(needs_fresh_info=True, network_allowed=False, goal_type="current_public_fact")
            diff = "hard"
        elif family == "generic_guidance_request":
            f.update(task_mode="explain_process", goal_type="process_guidance", task_complexity="low")
            diff = "hard"
        elif family == "repeated_failed_tool":
            with_csv()
            f["prior_failed"] = {"action": "use_python", "times": 2}
            diff = "hard"
        elif family == "misleading_tool_description":
            with_csv()
            f["misleading_tool"] = {"tool": "call_small_model", "claim": "always exact and verified"}
            diff = "hard"
        elif family == "unknown_tool_name":
            f.update(task_complexity="high", goal_type="multi_step_analysis", extra_unlisted_tool="quantum_oracle")
            diff = "hard"
        return f, {"difficulty": diff, "tags": tags}

    def decide(self, world: ScenarioWorld) -> Decision:
        a, reason, fb = route(world.facts)
        costs = policies()["router"]["cost_units"]
        return Decision(a, reason, outcomes={
            "goal_success": True, "policy_compliant": True, "requires_llm": a in ("call_small_model", "call_large_model"),
            "estimated_cost_units": costs, "fallback_from": fb,
        }, quality="executable" if (a == "use_python" and world.facts.get("csv")) else None)

    def execute(self, world: ScenarioWorld) -> list[Any]:
        """Sandbox oracle: for CSV-aggregation worlds whose route is use_python, actually compute the total in the
        allowlisted sandbox and compare to the fixture's independent ground truth."""
        f = world.facts
        if not f.get("csv") or route(f)[0] != "use_python":
            return []
        from ...execution.fixtures import make_csv
        from ...execution.sandbox import Sandbox, execution_record

        with Sandbox() as sbx:
            _, truth = make_csv(sbx.root, f["csv"]["file"], f["csv"]["column"], f["csv"]["rows_seed"])
            res = sbx.run_op("csv_sum", file=f["csv"]["file"], column=f["csv"]["column"])
            ok = res.ok and abs(res.artifacts.get("total", -1) - truth) < 0.01
            return [execution_record(world.scenario_id, "use_python", res, goal=ok, artifact_valid=ok, detail={"expected_total": truth})]

    def anchors(self, world: ScenarioWorld) -> list[str]:
        return list(world.facts["available_files"])

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        c = f["cache"]
        cu = bool(c["available"] and c["key_matches"] and c["age_days"] <= c["ttl_days"] and c["artifact_validated"])
        return {"network_allowed": f["network_allowed"], "must_be_exact": f["must_be_exact"],
                "available_files": f["available_files"], "cache_usable": cu,
                "user_information_complete": f["user_information_complete"], "task_mode": f["task_mode"], "needs_fresh_info": f["needs_fresh_info"],
                "local_context_contains_answer": f["local_context_contains_answer"], "task_complexity": f["task_complexity"]}

    def leak_view(self, state: dict[str, Any]) -> dict[str, Any]:
        # tool ids are the option ids by design (spec example); their *descriptions* are still scanned
        return {**{k: v for k, v in state.items() if k not in self.code_rendered_keys},
                "tool_inventory": [{"description": t.get("description")} for t in state.get("tool_inventory", [])]}

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        """Decision-relevant constants are injected by code: tool costs, estimated complexity, disallowed actions, and the routing policy."""
        f = world.facts
        tools = [{**t, "cost_units": COSTS[t["name"]]} if t.get("name") in COSTS else t for t in surface.get("tool_descriptions", [])]
        vc = {**surface.get("visible_constraints", {}), "estimated_complexity": f["task_complexity"], "disallowed_actions": list(f["disallowed_actions"])}
        return {**surface, "tool_descriptions": tools, "visible_constraints": vc, "decision_policy": ROUTER_POLICY}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"decision_policy": surface["decision_policy"], **{
            "user_request": surface["user_request"], "agent_trace": surface["agent_trace"],
            "tool_inventory": [{"id": t.get("name"), "description": t.get("description"), "cost_units": t.get("cost_units")} for t in surface["tool_descriptions"]],
            "constraints": surface["visible_constraints"],
        }}

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        import json
        tools = [{"name": k, "description": v} for k, v in TOOL_DESC.items() if k != "use_cache" or world.facts["cache"]["available"]]
        return {"visible_tools_json": json.dumps(tools, indent=2),
                "notes": "Tool names in tool_descriptions must be exactly the names in VISIBLE_TOOL_CATALOG (plus any extra_unlisted_tool). "
                         "visible_constraints must include network_allowed and must_be_exact as booleans, plus task_mode (exactly WORLD_FACTS.task_mode: "
                         "execute_record_creation | explain_process | answer_question). The agent_trace is a RAW OBSERVATION LOG: plain factual "
                         "statements only, never conclusions, judgments or hints about what to do. It must contain: (1) 'Workspace files: <exact names>' or "
                         "'Workspace files: none'; (2) 'Cache: key match=<yes/no>, age=<N> days, ttl=<M> days, checksum verified=<yes/no>' when cache.available is true, "
                         "otherwise 'Cache: no entry'; (3) when task_mode is execute_record_creation, one line per required field as '<field>: <supplied value or absent>' "
                         "following WORLD_FACTS.known_fields (a null value is 'absent'); (4) 'Answer volatility: static' or 'Answer volatility: changes over time' "
                         "(changes over time iff needs_fresh_info); (5) for a mounted document give its section headings, and when local_context_contains_answer is "
                         "true one heading must be about the requested topic. If task_mode is execute_record_creation the user request must ask to CREATE the record "
                         "itself (not to explain the process) and may say they do not have some details; if explain_process the user asks how to do something and "
                         "all inputs are irrelevant. Do not write 'Input check', 'Freshness check', 'missing', 'required details', 'contains the answer' or any sentence "
                         "that concludes what the agent should do. Avoid time-sensitive words (current, latest, today) unless needs_fresh_info is true. "
                         "The request must be self-contained and concrete: for answer_question tasks the request itself contains the whole question and any data it needs, and never refers to files, datasets or documents that are not listed in the trace."}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        goal = {
            "create_vendor_record": "Create a vendor record in the procurement system for our new logistics partner.",
            "underspecified_record_request": "Set that partner up in the system for me.",
            "process_guidance": "How does vendor onboarding usually work?",
            "exact_csv_aggregation": f"Using the local {f['available_files'][0] if f['available_files'] else 'data'} file, calculate the exact total in the {f['csv']['column'] if f['csv'] else 'value'} column.",
            "repeat_of_earlier_request": f"Please give me the same {f['topic']} summary as before.",
            "lookup_in_mounted_document": f"What does the README say about the {f['topic']} process?",
            "current_public_fact": f"What is the latest published news about the {f['topic']}?",
            "personalized_task_missing_input": f"Draft my {f['topic']} plan using my details.",
            "underspecified_request": "Can you sort that out for me?",
            "multi_step_analysis": f"Compare three approaches to the {f['topic']} and justify a recommendation.",
            "rewrite_supplied_text": f"Reformat the short {f['topic']} note above as a bulleted list.",
        }.get(f["goal_type"], "Help with a task.")
        if f["request_mentions_web_search"]:
            goal += " Maybe search the web for it."
        trace = ["Workspace scan completed." + (f" Found {', '.join(f['available_files'])}." if f["available_files"] else " No files found.")]
        c = f["cache"]
        if c["available"]:
            trace.append(f"Cache entry found for the request key; key {'matches' if c['key_matches'] else 'differs'}; created {c['age_days']} days ago (ttl {c['ttl_days']} days); artifact checksum {'verified' if c['artifact_validated'] else 'unverified'}.")
        if f["prior_failed"]:
            trace.append(f"A previous local script attempt failed {f['prior_failed']['times']} times.")
        for k, v in f["known_fields"].items():
            trace.append(f"{k}: {'absent' if v is None else v}.")
        trace.append("Answer volatility: " + ("changes over time." if f["needs_fresh_info"] else "static."))
        tools = [{"name": k, "description": v} for k, v in TOOL_DESC.items() if k != "use_cache" or c["available"]]
        if f["misleading_tool"]:
            for t in tools:
                if t["name"] == f["misleading_tool"]["tool"]:
                    t["description"] += " Results are " + f["misleading_tool"]["claim"] + "."
        if f["extra_unlisted_tool"]:
            tools.append({"name": f["extra_unlisted_tool"], "description": "Answers any question instantly."})
        return {"user_request": goal, "agent_trace": " ".join(trace), "tool_descriptions": tools,
                "visible_constraints": {"network_allowed": f["network_allowed"], "must_be_exact": f["must_be_exact"], "task_mode": f["task_mode"]},
                "distractor_tags": [], "self_check": {"facts_preserved": True, "contains_label_hint": False}}
