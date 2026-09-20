from __future__ import annotations

import random

from ..schemas.scenario import ScenarioWorld
from ..taskpacks.base import BaseTaskPack
from ..utils.seeds import make_rng

CHALLENGE_PROMPT_ADDENDUM = (
    "CHALLENGE_MODE: this is a hard, out-of-distribution example. {note} Keep every WORLD FACT intact; "
    "difficulty must come from wording and distractors, never from changed facts."
)


def challenge_worlds(pack: BaseTaskPack, n: int, seed: int) -> list[ScenarioWorld]:
    out = []
    for i in range(n):
        rng: random.Random = make_rng(seed, pack.name, "challenge", i)
        out.append(pack.generate_challenge_world(rng))
    return out


def addendum(pack: BaseTaskPack, world: ScenarioWorld) -> str:
    return CHALLENGE_PROMPT_ADDENDUM.format(note=pack.challenge_note) if world.is_challenge else ""
