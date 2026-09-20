from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SandboxOutcome(BaseModel):
    exit_code: int = 0
    goal_completed: bool = False
    policy_compliant: bool = True
    artifact_valid: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    runtime_ms: int = 0
    tool_calls: int = 0
    large_model_calls: int = 0


class ExecutionRecord(BaseModel):
    execution_id: str
    scenario_id: str
    candidate_action: str
    outcome: SandboxOutcome
    executor_version: str = "sandbox_v1"
    detail: dict[str, Any] = Field(default_factory=dict)


class RouteOutcome(BaseModel):
    runs: int
    successes: int
    success_rate: float
    lower_bound: float | None = None


class RolloutSummary(BaseModel):
    scenario_family: str
    route_outcomes: dict[str, RouteOutcome]
    policy: dict[str, Any]
    preferred_option: str | None


class SandboxResult(BaseModel):
    ok: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    artifacts: dict[str, Any] = Field(default_factory=dict)
