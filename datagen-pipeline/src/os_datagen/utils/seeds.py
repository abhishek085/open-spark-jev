from __future__ import annotations

import random

from .hashing import stable_int


def make_rng(*parts: object) -> random.Random:
    return random.Random(stable_int(*parts, bits=64))
