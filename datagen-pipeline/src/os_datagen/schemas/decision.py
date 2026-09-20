from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import DecisionType
from .provenance import ProvenanceRecord


class OptionDef(BaseModel):
    id: str
    definition: str


class LevelDef(BaseModel):
    value: int
    definition: str


class ProbabilitySemantics(BaseModel):
    """What a probability means for this pack (docs/probability_semantics.md)."""

    kind: Literal["deterministic_point_mass", "rollout_distribution", "annotator_distribution"]
    description: str


class TruthRecord(BaseModel):
    """Computed only by code, policy, solver, controlled world, sandbox or declared adjudication."""

    oracle_version: str
    decision_type: DecisionType
    preferred_option: str | None = None
    acceptable_options: list[str] = Field(default_factory=list)
    preferred_level: int | None = None
    acceptable_levels: list[int] = Field(default_factory=list)
    boolean_truth: bool | None = None
    reason_code: str
    expected_outcomes: dict[str, Any] = Field(default_factory=dict)
    label_quality: str = "deterministic"
    label_source: str
    distribution: dict[str, float] | None = None


class PublicTruth(BaseModel):
    """Truth as written to the model-facing record."""

    preferred_option: str | None = None
    acceptable_options: list[str] = Field(default_factory=list)
    preferred_level: int | None = None
    acceptable_levels: list[int] = Field(default_factory=list)
    truth: bool | None = None
    label_source: str
    oracle_version: str
    reason_code: str
    label_quality: str = "deterministic"
    distribution: dict[str, float] | None = None


class DecisionSpec(BaseModel):
    type: DecisionType
    state: dict[str, Any]
    question: dict[str, Any]  # instructions + options|levels (boolean: instructions only)


class QualityRecord(BaseModel):
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)
    semantic_verification: str = "pending"  # passed | escalated_passed | skipped
    dedupe_status: str = "pending"  # unique | near_duplicate | exact_duplicate
    supportability: str = "pending"  # supported | tagged_disagreement | skipped
    is_challenge: bool = False


class DatasetRecord(BaseModel):
    record_id: str
    task_pack: str
    task_version: str
    split: str
    decision: DecisionSpec
    truth: PublicTruth
    quality: QualityRecord
    provenance: ProvenanceRecord
    # Analysis-only slice keys (never shown to a model): namespace, scenario_family, split_family, ...
    meta: dict[str, Any] = Field(default_factory=dict)
