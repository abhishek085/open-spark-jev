"""Mixture balancing: trim accepted rows so the composition matches the target shares (deterministic, split-proportional)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..utils.hashing import stable_int


def balance(rows_by_split: dict[str, list[dict[str, Any]]], targets: dict[str, float], family_key: str = "composition_family"
            ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Targets are renormalised over the families that actually have accepted rows (a family with no task pack, e.g. data_code_workflows,
    cannot be filled and is reported separately). The final size is set by the binding family: total = min_f available_f / share_f.
    Within a family, rows are kept per split in proportion to that split's share, ordered by a stable hash of the record id."""
    avail: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for split, rows in rows_by_split.items():
        for r in rows:
            avail[r["meta"].get(family_key, "?")][split].append(r)
    fams = {f: t for f, t in targets.items() if t > 0 and f in avail}
    z = sum(fams.values())
    share = {f: t / z for f, t in fams.items()}
    n_avail = {f: sum(len(v) for v in avail[f].values()) for f in fams}
    total = int(min(n_avail[f] / share[f] for f in fams)) if fams else 0
    keep_f = {f: round(share[f] * total) for f in fams}
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in rows_by_split}
    for f in fams:
        for split, rows in avail[f].items():
            k = round(keep_f[f] * len(rows) / max(1, n_avail[f]))
            out[split] += sorted(rows, key=lambda r: stable_int("mixture", r["record_id"]))[:k]
    for s in out:
        out[s].sort(key=lambda r: r["record_id"])
    kept = sum(len(v) for v in out.values())
    rep = {"binding_family": min(fams, key=lambda f: n_avail[f] / share[f]) if fams else None, "total_target_rows": total, "kept_rows": kept,
           "families": {f: {"target_share_renormalised": round(share[f], 4), "available": n_avail[f], "kept": sum(1 for s in out.values() for r in s if r["meta"].get(family_key) == f)} for f in fams},
           "unfilled_families": {f: t for f, t in targets.items() if t > 0 and f not in avail}}
    return out, rep
