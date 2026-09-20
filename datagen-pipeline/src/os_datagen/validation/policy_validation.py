from __future__ import annotations

from ..schemas.decision import DatasetRecord, TruthRecord
from ..schemas.scenario import ScenarioWorld
from ..taskpacks.base import BaseTaskPack


class OracleError(RuntimeError):
    """Oracle failures are pipeline errors, never data rejections."""


def rerun_oracle(pack: BaseTaskPack, world: ScenarioWorld, truth: TruthRecord) -> list[str]:
    try:
        again = pack.solve_oracle(world)
    except Exception as e:  # noqa: BLE001
        raise OracleError(f"{pack.name}/{world.scenario_id}: {e}") from e
    return [] if again.model_dump() == truth.model_dump() else ["policy:oracle_not_reproducible"]


def validate_record(pack: BaseTaskPack, record: DatasetRecord) -> list[str]:
    return [f"policy:{i.code}" for i in pack.validate_semantics(record) if i.severity == "error"]
