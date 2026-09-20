"""Regression: for seed 42 the sampled worlds (ids, families) and oracle labels must not drift silently.
If you change a sampler/oracle on purpose, regenerate labels_seed42.json (see docs/task_pack_authoring.md) and bump oracle_version."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from os_datagen.config import GenerationConfig
from os_datagen.generation.oracle import solve
from os_datagen.generation.scenario_sampler import sample_split
from os_datagen.taskpacks.registry import get_pack

GOLD = json.loads((Path(__file__).parent / "labels_seed42.json").read_text())


@pytest.mark.parametrize("key", sorted(GOLD))
def test_labels_stable(key):
    name, split = key.split("/")
    p = get_pack(name)
    ws = sample_split(p, split, 8, 42, 1, GenerationConfig())
    assert [[w.scenario_id, w.scenario_family, p.label_key(solve(p, w))] for w in ws] == GOLD[key]
