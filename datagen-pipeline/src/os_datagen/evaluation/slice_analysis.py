from __future__ import annotations

from collections import defaultdict
from typing import Any

from .metrics import summarize_rows


def length_bucket(n: int) -> str:
    return "short" if n < 300 else "medium" if n < 900 else "long"


def slices(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Slice metrics by namespace, task pack, scenario family, difficulty, option count, length and truth tier."""
    keys = ("namespace", "task_pack", "scenario_family", "difficulty", "option_count", "length_bucket", "truth_tier", "label_source")
    out: dict[str, dict[str, Any]] = {}
    for k in keys:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[str(r["slice"][k])].append(r)
        out[k] = {g: summarize_rows(v) for g, v in sorted(groups.items())}
    return out
