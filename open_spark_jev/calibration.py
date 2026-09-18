"""Calibration metrics and post-hoc temperature scaling.

All functions take plain Python / numpy inputs so they can be used both in the training loop
(on detached tensors) and in the evaluation harness (on JSON results).

Metrics
-------
* ``brier``  - multi-class Brier score, mean over samples of sum_k (p_k - y_k)^2. Proper.
* ``nll``    - negative log-likelihood of the true label. Proper.
* ``ece``    - expected calibration error with equal-width confidence bins (top-label).
* ``reliability`` - per-bin (confidence, accuracy, count) for reliability diagrams.
* ``brier_noul`` / ``ece_noul`` - binary versions on P(true), for Noul questions.

Temperature scaling
-------------------
``fit_temperature`` finds T minimising NLL of softmax(logits / T). One scalar per question
type is the default in ``configs/train/*.yaml``; it is stored in the checkpoint's
``calibration.json`` and applied at serve time by dividing label logits before softmax.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def softmax(logits, axis=-1) -> np.ndarray:
    z = _np(logits)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def brier(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    p = _np(probs)
    y = np.zeros_like(p)
    y[np.arange(len(labels)), _np(labels).astype(int)] = 1.0
    return float(((p - y) ** 2).sum(-1).mean())


def brier_soft(probs, target_probs) -> float:
    """Brier against a *soft* target distribution (e.g. a known posterior or teacher dist)."""
    p, t = _np(probs), _np(target_probs)
    return float(((p - t) ** 2).sum(-1).mean())


def nll(probs, labels, eps: float = 1e-12) -> float:
    p = _np(probs)
    idx = _np(labels).astype(int)
    return float(-np.log(p[np.arange(len(idx)), idx] + eps).mean())


def accuracy(probs, labels) -> float:
    return float((_np(probs).argmax(-1) == _np(labels).astype(int)).mean())


def macro_f1(probs, labels) -> float:
    pred = _np(probs).argmax(-1)
    y = _np(labels).astype(int)
    f1s = []
    for c in np.unique(np.concatenate([pred, y])):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    confidence: float
    accuracy: float
    count: int


def reliability(probs, labels, n_bins: int = 10) -> list[ReliabilityBin]:
    p = _np(probs)
    conf = p.max(-1)
    correct = (p.argmax(-1) == _np(labels).astype(int)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            out.append(ReliabilityBin(float(lo), float(hi), 0.0, 0.0, 0))
        else:
            out.append(
                ReliabilityBin(float(lo), float(hi), float(conf[m].mean()), float(correct[m].mean()), int(m.sum()))
            )
    return out


def ece(probs, labels, n_bins: int = 10) -> float:
    bins = reliability(probs, labels, n_bins)
    n = sum(b.count for b in bins) or 1
    return float(sum(b.count / n * abs(b.confidence - b.accuracy) for b in bins))


def brier_noul(p_true: Sequence[float], truth: Sequence[int]) -> float:
    p, y = _np(p_true), _np(truth)
    return float(((p - y) ** 2).mean())


def ece_noul(p_true, truth, n_bins: int = 10) -> float:
    p, y = _np(p_true), _np(truth)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(total)


def fit_temperature(logits, labels, grid=None) -> float:
    """1-D search for the NLL-optimal temperature. Grid + local refinement; no torch needed."""
    z, y = _np(logits), _np(labels).astype(int)
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))

    def _nll(t):
        return nll(softmax(z / t), y)

    best = min(grid, key=_nll)
    # local refinement
    lo, hi = best / 1.3, best * 1.3
    for _ in range(40):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if _nll(m1) < _nll(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2)


def summary(probs, labels, n_bins: int = 10) -> dict:
    return {
        "n": len(labels),
        "accuracy": accuracy(probs, labels),
        "macro_f1": macro_f1(probs, labels),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels),
        "ece": ece(probs, labels, n_bins),
    }
