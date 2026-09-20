from __future__ import annotations

from typing import Any

from ..schemas.scenario import ScenarioWorld
from ..taskpacks.base import BaseTaskPack
from .leakage import flatten


def check_contradictions(pack: BaseTaskPack, world: ScenarioWorld, surface: dict[str, Any]) -> list[str]:
    """Deterministic structured-content checks (pack-defined) plus trivial degenerate-text checks.
    Semantic contradiction/ambiguity is the verifier's and (if routed) the judge's job."""
    issues = list(pack.check_surface(world, surface))
    parts = [p for p in flatten(surface, keys=False) if isinstance(p, str)]
    if not any(len(p.strip()) > 3 for p in parts):
        issues.append("degenerate:empty_text")
    return issues
