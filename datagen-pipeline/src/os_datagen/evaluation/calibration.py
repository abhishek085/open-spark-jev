from __future__ import annotations

import math
from typing import Any

from .metrics import EPS, nll, normalize


def softmax_from_scores(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    m = max(scores.values())
    ex = {k: math.exp((v - m) / temperature) for k, v in scores.items()}
    return normalize(ex)


def fit_temperature(rows: list[dict[str, Any]], lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature scaling by golden-section search on NLL. `rows` must come from the CALIBRATION split
    (each with raw_scores and acceptable). Never call this with locked-test rows."""
    def loss(T: float) -> float:
        return sum(nll(softmax_from_scores(r["raw_scores"], T), r["acceptable"]) for r in rows) / max(1, len(rows))

    g = (math.sqrt(5) - 1) / 2
    a, b = math.log(lo), math.log(hi)
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(60):
        if loss(math.exp(c)) < loss(math.exp(d)):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return math.exp((a + b) / 2)


_ = EPS
