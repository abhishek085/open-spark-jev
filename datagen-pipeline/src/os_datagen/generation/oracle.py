from __future__ import annotations

from ..schemas.decision import TruthRecord
from ..schemas.scenario import ScenarioWorld
from ..taskpacks.base import BaseTaskPack
from ..validation.policy_validation import OracleError


def solve(pack: BaseTaskPack, world: ScenarioWorld) -> TruthRecord:
    """Truth comes only from the pack's code oracle. Errors are pipeline errors (OracleError)."""
    try:
        truth = pack.solve_oracle(world)
    except Exception as e:  # noqa: BLE001
        raise OracleError(f"{pack.name}/{world.scenario_id}: {type(e).__name__}: {e}") from e
    if pack.decision_type == "choice":
        legal = set(pack.option_ids())
        if truth.preferred_option not in legal or not set(truth.acceptable_options) <= legal:
            raise OracleError(f"{pack.name}: oracle returned option outside the legal set: {truth.preferred_option}")
    return truth
