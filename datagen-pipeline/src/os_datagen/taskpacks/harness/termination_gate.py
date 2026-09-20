from __future__ import annotations

import random
from typing import Any

from ...config import policies
from ...generation.controlled_worlds import Pool
from ...schemas.scenario import ScenarioWorld
from ..base import BaseTaskPack, Decision


def termination(f: dict[str, Any]) -> tuple[str, str]:
    a, t, e = f["artifact"], f["tests"], f["error"]
    if f["user_input_missing"]:
        return "ask_user", "required_user_input_missing"
    if e["kind"] == "transient":
        return ("retry", "transient_failure_within_budget") if f["retries_used"] < f["retry_budget"] else ("escalate", "retry_budget_exhausted")
    if e["kind"] == "permanent_env":
        return "escalate", "permanent_environment_failure"
    if not a["exists"]:
        return ("retry", "artifact_absent_within_budget") if f["retries_used"] < f["retry_budget"] else ("escalate", "artifact_absent_budget_exhausted")
    if not a["nonempty"] or not a["schema_valid"]:
        return "repair", "artifact_empty_or_schema_invalid"
    if t and t["failed"] > 0:
        return "repair", "tests_failing"
    if f["signals_conflict"]:
        return "escalate", "conflicting_validation_signals"
    if f["fields_covered"] < f["fields_total"]:
        return "continue", "required_fields_missing"
    return "finish", "artifact_valid_and_complete"


TERMINATION_POLICY = (
    "Choose the next control step. (1) If user input is missing, ask the user. (2) If an error is transient, or the artifact is absent, and attempts_used is "
    "below attempt_budget, run the same step again; if the budget is used up, hand the problem to a person. (3) A permanent environment error, or conflicting "
    "validation signals, is handed to a person. (4) If the artifact exists but is empty, fails its schema, or any test fails, fix the artifact. (5) If the artifact "
    "is valid and every test passes but required fields are still uncovered, keep working on the remaining fields. (6) If the artifact exists, is non-empty, "
    "schema-valid, every test passes and all required fields are covered, stop as complete."
)


class TerminationGate(BaseTaskPack):
    name = "harness_termination_gate_v1"
    namespace = "harness"
    oracle_version = "termination_policy_v1"
    label_source = "artifact_check_retry_policy_v1"
    label_quality = "executable"
    prompt_id = "harness.termination"
    prompt_dir = "harness/termination_gate"
    composition_family = "agent_harness"
    code_rendered_keys = ("decision_policy",)
    supportability = "reject"
    id_prefix = "trm"
    options = [
        ("finish", "The work is complete and valid; stop."),
        ("continue", "Keep working on the remaining requirements."),
        ("repair", "Fix the defect in the produced artifact."),
        ("retry", "Run the failed step again unchanged."),
        ("escalate", "Hand the problem to a person or another system."),
        ("ask_user", "Ask the user for missing input."),
    ]
    instruction_variants = [
        "Given the execution so far, what should the harness do next?",
        "What is the correct next step for the harness, given the current artifact and validation state?",
        "Decide the next control action from the trace and the validation summary.",
        "Use the artifact checks and attempt budget to choose what the harness should do next.",
        "Based on the current validation state, which control action should the harness take?",
    ]
    extra_leak_patterns = [r"\bask[ _-]?user\b"]
    families = ["complete_valid", "schema_fails", "test_failure_code_defect", "transient_within_budget", "retry_budget_exhausted",
                "missing_requirement", "ambiguity_needs_user", "conflicting_signals"]
    challenge_families = ["hidden_failed_test", "empty_output_weak_schema", "duplicated_success_history", "tool_success_but_artifact_absent"]
    surface_fields = {"goal": str, "agent_trace": list[str], "artifacts": list[dict[str, Any]], "validation_summary": dict[str, Any],
                      "remaining_open_items": list[str], "noise_tags": list[str]}
    verifier_defaults = {"user_input_missing": False, "validation_signals_conflict": False, "error_kind": "none"}
    verifier_fields = {"artifact_exists": bool, "artifact_nonempty": bool, "schema_valid": bool, "tests_total": int, "tests_failed": int,
                       "attempts_used": int, "attempt_budget": int, "error_kind": str, "user_input_missing": bool,
                       "required_fields_missing": int, "validation_signals_conflict": bool}
    challenge_note = "Bury the decisive failure in noise or reassuring history without altering any validation fact."

    def sample(self, rng: random.Random, family: str, pool: Pool, tone: str) -> tuple[dict[str, Any], dict[str, Any]]:
        budget = policies()["retry"]["default_budget"]
        total = rng.randint(6, 12)
        f: dict[str, Any] = {
            "goal_topic": rng.choice(pool.topics), "artifact_name": rng.choice(["report.json", "summary.csv", "output.md"]),
            "artifact": {"exists": True, "nonempty": True, "schema_valid": True},
            "tests": {"total": rng.randint(3, 8), "failed": 0}, "error": {"kind": "none", "detail": None},
            "retries_used": 0, "retry_budget": budget, "user_input_missing": False, "signals_conflict": False,
            "fields_total": total, "fields_covered": total, "tool_reports_success": True, "log_noise": rng.random() < 0.5, "history_note": None,
        }
        diff = "easy"
        if family == "complete_valid":
            pass
        elif family == "schema_fails":
            f["artifact"]["schema_valid"] = False
        elif family in ("test_failure_code_defect", "hidden_failed_test"):
            f["tests"]["failed"] = rng.randint(1, 2)
            f["tool_reports_success"] = family == "hidden_failed_test"
            diff = "medium" if family == "test_failure_code_defect" else "hard"
        elif family == "transient_within_budget":
            f["error"] = {"kind": "transient", "detail": "network timeout while calling an internal service"}
            f["artifact"]["exists"] = False
            f["retries_used"] = rng.randint(0, budget - 1)
            f["tool_reports_success"] = False
        elif family == "retry_budget_exhausted":
            f["error"] = {"kind": "transient", "detail": "network timeout while calling an internal service"}
            f["artifact"]["exists"] = False
            f["retries_used"] = budget
            f["tool_reports_success"] = False
            diff = "medium"
        elif family == "missing_requirement":
            f["fields_covered"] = total - rng.randint(1, 3)
        elif family == "ambiguity_needs_user":
            f["user_input_missing"] = True
            f["fields_covered"] = total - 2
            diff = "medium"
        elif family == "conflicting_signals":
            f["signals_conflict"] = True
            diff = "hard"
        elif family == "empty_output_weak_schema":
            f["artifact"]["nonempty"] = False
            diff = "hard"
        elif family == "duplicated_success_history":
            f["tests"]["failed"] = 1
            f["history_note"] = "the same earlier successful run is quoted twice in the log"
            diff = "hard"
        elif family == "tool_success_but_artifact_absent":
            f["artifact"]["exists"] = False
            f["retries_used"] = rng.randint(0, budget - 1)
            f["tool_reports_success"] = True
            diff = "hard"
        if not f["artifact"]["exists"]:
            f["artifact"].update(nonempty=False, schema_valid=False)
            f["tests"] = None
        return f, {"difficulty": diff, "tags": [family]}

    def decide(self, world: ScenarioWorld) -> Decision:
        a, why = termination(world.facts)
        return Decision(a, why)

    def expected_extraction(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        t = f["tests"] or {"total": 0, "failed": 0}
        return {"artifact_exists": f["artifact"]["exists"], "artifact_nonempty": f["artifact"]["nonempty"], "schema_valid": f["artifact"]["schema_valid"],
                "tests_total": t["total"], "tests_failed": t["failed"], "attempts_used": f["retries_used"], "attempt_budget": f["retry_budget"],
                "error_kind": f["error"]["kind"], "user_input_missing": f["user_input_missing"],
                "required_fields_missing": f["fields_total"] - f["fields_covered"], "validation_signals_conflict": f["signals_conflict"]}

    def anchors(self, world: ScenarioWorld) -> list[str]:
        return [world.facts["artifact_name"]]

    def extra_template_vars(self, world: ScenarioWorld) -> dict[str, Any]:
        return {"notes": "Refer to retries only as 'attempts' ('attempt 2 of 3'; report attempts_used and attempt_budget in validation_summary). "
                         "validation_summary keys: artifact_exists, artifact_nonempty, schema_valid, tests_total, tests_failed, attempts_used, "
                         "attempt_budget, error_kind (none|transient|permanent_env), required_fields_covered, required_fields_total, "
                         "user_input_missing, validation_signals_conflict. artifacts: the produced artifact with a factual status (exists/absent, "
                         "empty/non-empty, schema check outcome). If tool_reports_success is true but tests fail or the artifact is absent, "
                         "the trace may show reassuring tool messages but the validation_summary must state the true facts. tests is null => report tests_total 0. "
                         "Do not use the words finish, continue, repair, retry, escalate."}

    def finalize_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {**surface, "decision_policy": TERMINATION_POLICY}

    def state_from_surface(self, world: ScenarioWorld, surface: dict[str, Any]) -> dict[str, Any]:
        return {"decision_policy": surface["decision_policy"], **{k: surface[k] for k in ("goal", "agent_trace", "artifacts", "validation_summary", "remaining_open_items")}}

    def template_surface(self, world: ScenarioWorld) -> dict[str, Any]:
        f = world.facts
        t = f["tests"] or {"total": 0, "failed": 0}
        tr = [f"Step 1: started work on {f['goal_topic']}.", f"Step 2: tool reported {'success' if f['tool_reports_success'] else 'an error: ' + str(f['error']['detail'])}."]
        return {"goal": f"Produce {f['artifact_name']} for the {f['goal_topic']} task.", "agent_trace": tr,
                "artifacts": [{"name": f["artifact_name"], "status": ("present" if f["artifact"]["exists"] else "absent") + ("" if f["artifact"]["nonempty"] else ", empty")}],
                "validation_summary": {"artifact_exists": f["artifact"]["exists"], "artifact_nonempty": f["artifact"]["nonempty"], "schema_valid": f["artifact"]["schema_valid"],
                                       "tests_total": t["total"], "tests_failed": t["failed"], "attempts_used": f["retries_used"], "attempt_budget": f["retry_budget"],
                                       "error_kind": f["error"]["kind"], "required_fields_covered": f["fields_covered"], "required_fields_total": f["fields_total"],
                                       "user_input_missing": f["user_input_missing"], "validation_signals_conflict": f["signals_conflict"]},
                "remaining_open_items": [] if f["fields_covered"] == f["fields_total"] else [f"{f['fields_total'] - f['fields_covered']} required fields not yet filled"],
                "noise_tags": []}
