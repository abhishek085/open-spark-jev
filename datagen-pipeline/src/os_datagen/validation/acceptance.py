from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..schemas.common import utc_now
from ..schemas.scenario import ScenarioWorld
from ..schemas.validation import RejectionRecord
from ..utils.logging import get_logger
from . import VALIDATOR_VERSIONS

log = get_logger()


class Stage(StrEnum):
    SAMPLED = "sampled"
    ORACLE_OK = "oracle_ok"
    GENERATED = "generated"
    SCHEMA_OK = "schema_ok"
    LEAK_OK = "leak_ok"
    ANCHORS_OK = "anchors_ok"
    VERIFIED = "verified"
    POLICY_OK = "policy_ok"
    UNIQUE = "unique"
    SPLIT_OK = "split_ok"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass
class Candidate:
    candidate_id: str
    world: ScenarioWorld
    variant: int = 0
    stage: Stage = Stage.ORACLE_OK
    reasons: list[str] = field(default_factory=list)
    surface: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    raw_path: str | None = None
    generator_model: str = ""
    repair_attempts: int = 0
    lower_confidence: bool = False
    verifier_model: str | None = None
    judge_model: str | None = None
    judged: bool = False
    trace: dict[str, Any] = field(default_factory=dict)

    def reject(self, *reasons: str) -> None:
        self.stage = Stage.REJECTED
        self.reasons += [r for r in reasons if r not in self.reasons]

    @property
    def alive(self) -> bool:
        return self.stage != Stage.REJECTED

    def rejection(self, prompt_version: str) -> RejectionRecord:
        return RejectionRecord(
            candidate_id=self.candidate_id, reasons=self.reasons, scenario_id=self.world.scenario_id,
            task_pack=self.world.task_pack, split=self.world.split, generator_model=self.generator_model,
            generator_prompt_version=prompt_version, validator_versions=VALIDATOR_VERSIONS,
            raw_generation_path=self.raw_path, created_at=utc_now(),
        )
