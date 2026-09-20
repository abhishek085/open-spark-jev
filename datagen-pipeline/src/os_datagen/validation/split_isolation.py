from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..generation.controlled_worlds import SPLIT_STYLES
from ..schemas.decision import DatasetRecord
from ..schemas.scenario import ScenarioWorld
from .deduplication import jaccard, shingles, visible_text


def check_worlds(worlds: list[ScenarioWorld]) -> list[str]:
    """Assign-time isolation: scenario ids unique; template (split_family) and entity pools disjoint by split."""
    issues: list[str] = []
    seen: dict[str, str] = {}
    fam: dict[str, set[str]] = defaultdict(set)
    pools: dict[str, set[str]] = defaultdict(set)
    for w in worlds:
        if w.scenario_id in seen and seen[w.scenario_id] != w.split:
            issues.append(f"split_leak:scenario_id_reused:{w.scenario_id}")
        seen[w.scenario_id] = w.split
        fam[w.split_family].add(w.split)
        pools[w.hidden.get("pool", "?")].add(w.split)
    issues += [f"split_leak:template_family_shared:{f}" for f, s in fam.items() if len(s) > 1]
    issues += [f"split_leak:entity_pool_shared:{p}" for p, s in pools.items() if len(s) > 1]
    return sorted(set(issues))


def check_records(by_split: dict[str, list[DatasetRecord]], near_threshold: float = 0.92,
                  heldout_families: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """Post-render isolation report: ids, template families, pools, exact + near-duplicate visible text across splits."""
    violations: list[str] = []
    base_ids: dict[str, str] = {}
    fam: dict[str, set[str]] = defaultdict(set)
    pools: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, dict[str, set[int]]] = defaultdict(dict)
    for split, recs in by_split.items():
        for r in recs:
            base = r.provenance.scenario_id
            if base in base_ids and base_ids[base] != split:
                violations.append(f"scenario_id_reused:{base}:{base_ids[base]}/{split}")
            base_ids[base] = split
            fam[r.meta.get("split_family", "?")].add(split)
            pools[str(r.meta.get("pool"))].add(split)
            texts[r.task_pack][f"{split}|{r.record_id}"] = shingles(visible_text(r))
    violations += [f"template_family_shared:{f}" for f, s in fam.items() if len(s) > 1]
    violations += [f"entity_pool_shared:{p}" for p, s in pools.items() if len(s) > 1]
    for pack_name, held in (heldout_families or {}).items():  # scenario-structure isolation: held-out families never in train
        seen_in_train = {r.meta.get("scenario_family") for r in by_split.get("train", []) if r.task_pack == pack_name}
        violations += [f"heldout_family_in_train:{pack_name}:{f}" for f in held if f in seen_in_train]
    for d in texts.values():
        items = sorted(d.items())
        for i, (ka, a) in enumerate(items):
            for kb, b in items[i + 1:]:
                if ka.split("|")[0] != kb.split("|")[0] and jaccard(a, b) >= near_threshold:
                    violations.append(f"cross_split_near_duplicate:{ka}~{kb}")
    return {"ok": not violations, "violations": sorted(set(violations)),
            "template_families": {k: sorted(v) for k, v in fam.items()},
            "entity_pools": {k: sorted(v) for k, v in pools.items()},
            "heldout_families": heldout_families or {},
            "declared_style_families": {s: [x[0] for x in v] for s, v in SPLIT_STYLES.items()}}
