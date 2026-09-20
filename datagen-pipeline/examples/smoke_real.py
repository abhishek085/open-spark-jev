"""Manual smoke test against real local endpoints. Phase-wise so only one model needs to be up:
  python examples/smoke_real.py generate PACK [N]   # generator only: render + deterministic gates, prints surfaces
  python examples/smoke_real.py verify PACK [N]     # generator output re-used from artifacts/smoke, verifier up
Run from datagen-pipeline/ with PYTHONPATH=src."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from os_datagen.config import load_config
from os_datagen.generation.pipeline import Pipeline
from os_datagen.llm.openai_compatible import OpenAICompatibleClient

phase, pack = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
cfg = load_config(None, "configs/models.local.yaml")
gen = OpenAICompatibleClient(cfg.models["generator"])
ver = OpenAICompatibleClient(cfg.models["verifier"])
p = Pipeline(cfg, Path("artifacts/smoke") / pack, gen, ver, None)
# n worlds spread over splits: 1 challenge + the rest train
plan = {pack: {"train": max(1, n - 1), "challenge": 1}}
p._sample(plan, seed=11)
if phase == "generate":
    p._generate(1)
    for c in p.cands:
        print(f"\n### {c.candidate_id} [{c.world.split}/{c.world.scenario_family}] stage={c.stage.value} reasons={c.reasons}")
        print("truth:", p.truths[c.world.scenario_id].preferred_option or p.truths[c.world.scenario_id].preferred_level or p.truths[c.world.scenario_id].boolean_truth)
        print(json.dumps(c.state or c.surface, indent=1, ensure_ascii=False)[:1500])
elif phase == "verify":
    p._generate(1)
    p._verify()
    for c in p.cands:
        print(f"\n### {c.candidate_id} stage={c.stage.value} reasons={c.reasons}")
        print(json.dumps(c.trace.get("verification"), ensure_ascii=False)[:800])
