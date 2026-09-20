from __future__ import annotations

from pydantic import BaseModel


class ProvenanceRecord(BaseModel):
    synthetic: bool = True
    scenario_id: str
    scenario_seed: int
    generator_model: str
    generator_prompt_version: str
    verifier_model: str | None = None
    verifier_prompt_version: str | None = None
    judge_model: str | None = None
    repair_attempt_count: int = 0
    lower_confidence_provenance: bool = False
    created_at: str
    source_code_commit: str
    config_hash: str | None = None
