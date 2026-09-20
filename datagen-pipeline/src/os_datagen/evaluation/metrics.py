from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

EPS = 1e-9


def normalize(p: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, v) for v in p.values())
    if s <= 0:
        return {k: 1 / len(p) for k in p}
    return {k: max(0.0, v) / s for k, v in p.items()}


def top1(p: dict[str, float]) -> tuple[str, float]:
    k = max(p, key=lambda x: p[x])
    return k, p[k]


def nll(p: dict[str, float], acceptable: list[str]) -> float:
    """-log P(acceptable set): with a single acceptable option this is the usual NLL."""
    return -math.log(max(EPS, sum(p.get(a, 0.0) for a in acceptable)))


def brier(p: dict[str, float], preferred: str) -> float:
    return sum((p[k] - (1.0 if k == preferred else 0.0)) ** 2 for k in p)


def ece(rows: list[dict[str, Any]], bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    """Expected calibration error on top-1 confidence vs acceptable-set correctness. Returns (ece, reliability bins)."""
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        buckets[min(bins - 1, int(r["confidence"] * bins))].append((r["confidence"], 1.0 if r["correct"] else 0.0))
    n = len(rows) or 1
    total, table = 0.0, []
    for b in range(bins):
        items = buckets.get(b, [])
        if items:
            conf = sum(c for c, _ in items) / len(items)
            acc = sum(a for _, a in items) / len(items)
            total += len(items) / n * abs(conf - acc)
            table.append({"bin": b, "lo": b / bins, "hi": (b + 1) / bins, "n": len(items), "mean_confidence": conf, "accuracy": acc})
        else:
            table.append({"bin": b, "lo": b / bins, "hi": (b + 1) / bins, "n": 0, "mean_confidence": None, "accuracy": None})
    return total, table


def per_option_pr(rows: list[dict[str, Any]], options: list[str]) -> dict[str, dict[str, float | int]]:
    out = {}
    for o in options:
        tp = sum(1 for r in rows if r["selected"] == o and r["preferred"] == o)
        fp = sum(1 for r in rows if r["selected"] == o and r["preferred"] != o)
        fn = sum(1 for r in rows if r["selected"] != o and r["preferred"] == o)
        out[o] = {"support": tp + fn, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0}
    return out


def selective_risk(rows: list[dict[str, Any]], steps: int = 20) -> list[dict[str, float]]:
    """Risk (error rate among answered) vs coverage as the confidence threshold rises."""
    out = []
    for i in range(steps):
        t = i / steps
        kept = [r for r in rows if r["confidence"] >= t]
        out.append({"threshold": t, "coverage": len(kept) / max(1, len(rows)),
                    "risk": (sum(1 for r in kept if not r["correct"]) / len(kept)) if kept else 0.0})
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    e, _ = ece(rows)
    by_class: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_class[r["preferred"]].append(r["selected"] == r["preferred"])
    return {
        "n": len(rows),
        "top1_accuracy": sum(r["selected"] == r["preferred"] for r in rows) / len(rows),
        "acceptable_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "macro_accuracy": sum(sum(v) / len(v) for v in by_class.values()) / len(by_class),
        "nll": sum(r["nll"] for r in rows) / len(rows),
        "brier": sum(r["brier"] for r in rows) / len(rows),
        "ece": e,
        "mean_confidence": sum(r["confidence"] for r in rows) / len(rows),
    }


def cost_quality_frontier(rows: list[dict[str, Any]], costs: dict[str, float], defer_option: str = "ask_user", steps: int = 10) -> list[dict[str, float]]:
    """For routed actions: deferring low-confidence predictions to `defer_option` trades cost for quality."""
    out = []
    for i in range(steps + 1):
        t = i / steps
        cost, ok = 0.0, 0
        for r in rows:
            a = r["selected"] if r["confidence"] >= t else defer_option
            cost += costs.get(a, 0.0)
            ok += a in r["acceptable"]
        out.append({"threshold": t, "mean_cost": cost / max(1, len(rows)), "quality": ok / max(1, len(rows))})
    return out
