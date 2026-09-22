"""Confidence calibrator ("is the top answer right?") fitted on a held-out calibration split.

The softmax maximum, even after temperature scaling, is a poor gate: it mixes how peaked the distribution is with how often peaked answers are
right. This fits a small logistic model on features of the option logits (max probability, margin, entropy, number of options, answer type,
gap of the top logit to the mean) to predict P(top answer correct). It needs only the logits, so it works with any backend, and it is
fitted on calibration rows only. Use the predicted value, not the softmax maximum, to decide whether a decision may run automatically.
"""

from __future__ import annotations

import json
import math

import numpy as np

QTYPES = ("choice", "score", "noul")


def features(logits, qtype: str, T: float = 1.0) -> list[float]:
    z = np.asarray(logits, dtype=float) / T
    p = np.exp(z - z.max())
    p /= p.sum()
    s = np.sort(p)[::-1]
    ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
    return [float(s[0]), float(s[0] - (s[1] if len(s) > 1 else 0.0)), ent, math.log(len(p)), float(z.max() - z.mean()), *[1.0 if qtype == q else 0.0 for q in QTYPES]]


def fit(rows: list[dict], T: dict[str, float], C: float = 1.0) -> dict:
    """rows: dicts with logits, qtype, gold_idx. Returns a JSON-serialisable model."""
    from sklearn.linear_model import LogisticRegression

    X = np.array([features(r["logits"], r["qtype"], T.get(r["qtype"], 1.0)) for r in rows])
    y = np.array([int(np.argmax(r["logits"]) == r["gold_idx"]) for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=2000).fit((X - mu) / sd, y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "coef": clf.coef_[0].tolist(), "bias": float(clf.intercept_[0]), "T": T, "n": int(len(y)), "base_rate": float(y.mean())}


def predict(model: dict, logits, qtype: str) -> float:
    x = (np.array(features(logits, qtype, model["T"].get(qtype, 1.0))) - np.array(model["mu"])) / np.array(model["sd"])
    return float(1 / (1 + math.exp(-(float(np.dot(model["coef"], x)) + model["bias"]))))


def save(model: dict, path: str) -> None:
    json.dump(model, open(path, "w"), indent=1)


def load(path: str) -> dict:
    return json.load(open(path))
