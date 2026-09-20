from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from ..schemas.scenario import ScenarioWorld
from ..schemas.validation import FactComparison
from ..taskpacks.base import BaseTaskPack
from .leakage import flatten


def check_anchors(pack: BaseTaskPack, world: ScenarioWorld, state: dict[str, Any]) -> list[str]:
    """Deterministic fact anchors: required strings must appear verbatim (case-insensitive)."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", "\n".join(flatten(state, keys=False)).lower())  # 4,357 -> 4357
    return [f"fact_mismatch:anchor_missing:{a}"[:80] for a in pack.anchors(world) if a.lower() not in text]


def compare(pack: BaseTaskPack, world: ScenarioWorld, extracted: BaseModel) -> FactComparison:
    return pack.compare_facts(world, extracted)


def comparison_reasons(c: FactComparison) -> list[str]:
    return [f"fact_mismatch:{k}" for k in c.mismatched] + [f"fact_unverifiable:{k}" for k in c.unverifiable]
