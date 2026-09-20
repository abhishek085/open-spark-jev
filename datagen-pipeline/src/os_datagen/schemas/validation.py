from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "error"
    message: str = ""


class FactComparison(BaseModel):
    matched: list[str] = Field(default_factory=list)
    mismatched: list[str] = Field(default_factory=list)
    unverifiable: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatched and not self.unverifiable


class JudgeVerdict(BaseModel):
    verdict: Literal["accept", "reject", "human_review"]
    fact_preservation: Literal["pass", "fail", "uncertain"]
    visible_decision_ambiguity: Literal["none", "permitted", "unexpected"]
    label_leakage: Literal["none", "possible", "confirmed"]
    reasons: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class RejectionRecord(BaseModel):
    candidate_id: str
    status: str = "rejected"
    reasons: list[str]
    scenario_id: str
    task_pack: str
    split: str | None = None
    generator_model: str
    generator_prompt_version: str
    validator_versions: dict[str, str]
    raw_generation_path: str | None = None
    created_at: str
