"""Repeated-route rollouts -> empirical success rates with a Wilson lower confidence bound.

Executors: `use_python` runs the real sandbox on the world's CSV fixture. Model routes use a *declared success
profile* (configs/rollout_profiles.yaml) unless a real endpoint executor is supplied: results are labelled with
their executor so simulated rates are never mistaken for measured ones."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..config import policies
from ..schemas.execution import RolloutSummary, RouteOutcome
from ..taskpacks.base import BaseTaskPack
from ..utils.jsonl import read_jsonl, write_jsonl
from ..utils.paths import project_path
from ..utils.seeds import make_rng
from .fixtures import make_csv
from .sandbox import Sandbox


def wilson_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    if n == 0:
        return 0.0
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def _profiles() -> dict[str, Any]:
    p = project_path("configs", "rollout_profiles.yaml")
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def run_route(route: str, facts: dict[str, Any], rng_seed: int, trial: int) -> bool:
    if route == "use_python":
        csv_ = facts.get("csv")
        if not csv_:
            return False
        with Sandbox() as sbx:
            _, truth = make_csv(sbx.root, csv_["file"], csv_["column"], csv_["rows_seed"])
            res = sbx.run_op("csv_sum", file=csv_["file"], column=csv_["column"])
            return res.ok and abs(res.artifacts["total"] - truth) < 0.01
    prof = _profiles().get("simulated_model_success", {"call_small_model": {"low": 0.9, "high": 0.4}, "call_large_model": {"low": 0.97, "high": 0.9}})
    p = prof.get(route, {}).get(facts.get("task_complexity", "low"), 0.5)
    if facts.get("must_be_exact"):
        p = min(p, 0.6)  # models are not exact calculators
    return make_rng(rng_seed, route, trial).random() < p


def summarize(family: str, facts: dict[str, Any], routes: list[str], runs: int, seed: int) -> RolloutSummary:
    pol = policies()
    need, conf = pol["rollout"]["required_success_probability"], pol["rollout"]["confidence"]
    costs = pol["router"]["cost_units"]
    outs: dict[str, RouteOutcome] = {}
    for r in routes:
        ok = sum(run_route(r, facts, seed, t) for t in range(runs))
        outs[r] = RouteOutcome(runs=runs, successes=ok, success_rate=ok / runs, lower_bound=wilson_lower_bound(ok, runs, conf))
    valid = [r for r in routes if (outs[r].lower_bound or 0) >= need]
    pref = min(valid, key=lambda r: costs.get(r, 99)) if valid else None
    return RolloutSummary(scenario_family=family, route_outcomes=outs,
                          policy={"required_success_probability": need, "confidence": conf,
                                  "objective": "minimum_cost_subject_to_reliability_lower_bound",
                                  "executor": "sandbox(use_python)+simulated_model_profile"},
                          preferred_option=pref)


def run_dataset_rollouts(pack: BaseTaskPack, dataset: Path, routes: list[str], runs: int, out: Path) -> int:
    worlds = {r["world"]["scenario_id"]: r["world"] for r in read_jsonl(dataset.parent / "scenario_worlds.jsonl")}
    rows = []
    for rec in read_jsonl(dataset):
        w = worlds.get(rec["provenance"]["scenario_id"])
        if not w:
            continue
        s = summarize(w["scenario_family"], w["facts"], routes, runs, w["seed"])
        rows.append({"scenario_id": w["scenario_id"], "record_id": rec["record_id"], **s.model_dump(mode="json")})
    return write_jsonl(out / "rollout_summaries.jsonl", rows)


_ = tempfile  # (kept for executors that need scratch space)
