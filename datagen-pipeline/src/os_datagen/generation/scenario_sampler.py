from __future__ import annotations

import math
from collections import Counter

from ..config import GenerationConfig
from ..schemas.scenario import ScenarioWorld
from ..taskpacks.base import BaseTaskPack
from ..utils.hashing import sha256_obj, stable_int
from ..utils.seeds import make_rng
from .oracle import solve

_FAMILY_CACHE: dict[tuple[str, float], tuple[list[str], list[str]]] = {}


def family_split(pack: BaseTaskPack, max_share: float = 0.34) -> tuple[list[str], list[str]]:
    """(train_families, heldout_families). Held-out families feed calibration and locked_test only, so their scenario
    structure is unseen in training. A family is held out only if the remaining train families still produce EVERY label that
    the pack's non-challenge families can produce (so no label becomes unlearnable) and never more than `max_share` of families."""
    key = (pack.name, max_share)
    if key in _FAMILY_CACHE:
        return _FAMILY_CACHE[key]
    labels: dict[str, set[str]] = {}
    for fam in pack.families:
        ls: set[str] = set()
        for i in range(120):
            w = pack.sample_world(make_rng("famsplit", pack.name, fam, i), "train", families=[fam])
            ls.add(pack.label_key(solve(pack, w)))
        labels[fam] = ls
    all_labels = set().union(*labels.values())
    held: list[str] = []
    limit = max(1, int(len(pack.families) * max_share))
    for fam in sorted(pack.families, key=lambda f: stable_int("holdout", pack.name, f)):
        if len(held) >= limit:
            break
        rest = [f for f in pack.families if f != fam and f not in held]
        if set().union(*(labels[f] for f in rest)) >= all_labels:
            held.append(fam)
    train = [f for f in pack.families if f not in held]
    _FAMILY_CACHE[key] = (train, held)
    return train, held


def families_for(pack: BaseTaskPack, split: str, gen_cfg: GenerationConfig) -> list[str] | None:
    if not gen_cfg.holdout_families or split == "challenge":
        return None
    train, held = family_split(pack, gen_cfg.holdout_max_share)
    if not held:
        return None
    return train if split == "train" else held


def sample_split(pack: BaseTaskPack, split: str, n: int, base_seed: int, id_start: int,
                 gen_cfg: GenerationConfig) -> list[ScenarioWorld]:
    """Sample n worlds for one split with per-label caps (balanced label distribution) and no duplicate facts.
    Deterministic in (pack, split, base_seed)."""
    k = max(1, len(pack.option_ids()))
    cap = math.ceil(n / k * gen_cfg.label_balance_slack)
    counts: Counter[str] = Counter()
    n_easy = 0
    easy_cap = math.ceil(n * gen_cfg.max_easy_share) if n >= 20 else n
    seen: set[str] = set()
    out: list[ScenarioWorld] = []
    overflow: list[ScenarioWorld] = []
    for i in range(n * gen_cfg.oversample_factor):
        if len(out) >= n:
            break
        seed = stable_int(base_seed, pack.name, split, i, bits=31)
        w = pack.sample_world(make_rng(seed), split, families=families_for(pack, split, gen_cfg))
        w.seed, w.split = seed, split
        h = sha256_obj(w.facts)
        if h in seen:
            continue
        seen.add(h)
        label = pack.label_key(solve(pack, w))
        is_easy = w.difficulty == "easy"
        if counts[label] < cap and not (is_easy and n_easy >= easy_cap):
            counts[label] += 1
            n_easy += is_easy
            out.append(w)
        else:
            overflow.append(w)
    out += overflow[: n - len(out)]  # unreachable labels: fill rather than under-deliver
    for j, w in enumerate(out):
        w.scenario_id = f"{pack.id_prefix}-{id_start + j:06d}"
    return out
