from __future__ import annotations

from ..schemas.scenario import ScenarioWorld
from ..utils.hashing import stable_int


def variant_seed(world: ScenarioWorld, variant: int) -> int:
    """Generator sampling seed: same world, different surface realization per variant."""
    return stable_int(world.seed, "variant", variant, bits=31)


def candidate_id(world: ScenarioWorld, variant: int) -> str:
    return f"osj-{world.scenario_id}-v{variant + 1:03d}"
