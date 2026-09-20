from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScenarioWorld(BaseModel):
    """Internal source of truth. `facts` are renderable (may go to the generator prompt);
    `hidden` is bookkeeping/oracle-only and must never reach a prompt or a model-facing record."""

    scenario_id: str
    task_pack: str
    scenario_family: str
    split_family: str  # template/style family, e.g. train_family_a; disjoint across splits
    split: str = "train"
    seed: int = 0
    facts: dict[str, Any]
    hidden: dict[str, Any] = Field(default_factory=dict)
    candidate_actions: list[str] = Field(default_factory=list)
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)
    is_challenge: bool = False


class RenderRequest(BaseModel):
    task_pack: str
    prompt_version: str
    template_path: str  # relative to prompts/
    template_vars: dict[str, Any]
    world_facts: dict[str, Any]  # exactly what the generator is allowed to see
    seed: int
    challenge_note: str | None = None


class RenderedState(BaseModel):
    surface: dict[str, Any]
    generator_model: str
    prompt_version: str
    raw_generation_path: str | None = None
    attempts: int = 1
    repair_attempt_count: int = 0
    lower_confidence_provenance: bool = False
