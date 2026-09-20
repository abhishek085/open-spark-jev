from __future__ import annotations

from typing import Any

from ..schemas.decision import DatasetRecord
from ..validation.acceptance import Candidate


def lineage_row(rec: DatasetRecord, c: Candidate, prompt_hashes: dict[str, str], prompt_template: str) -> dict[str, Any]:
    return {
        "record_id": rec.record_id, "scenario_id": rec.provenance.scenario_id, "scenario_seed": rec.provenance.scenario_seed,
        "task_pack": rec.task_pack, "split": rec.split, "split_family": rec.meta.get("split_family"),
        "generator_model": rec.provenance.generator_model, "verifier_model": rec.provenance.verifier_model,
        "judge_model": rec.provenance.judge_model, "raw_generation_path": c.raw_path,
        "surface_prompt_sha256": prompt_hashes.get(prompt_template), "repair_attempt_count": c.repair_attempts,
        "source_code_commit": rec.provenance.source_code_commit,
    }
