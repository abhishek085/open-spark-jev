from __future__ import annotations

import json
from pathlib import Path

import pytest

from os_datagen.config import load_config, policies
from os_datagen.datasets.manifest import build_manifest, write_checksums
from os_datagen.datasets.splitter import split_counts
from os_datagen.execution.fixtures import make_csv
from os_datagen.execution.rollouts import summarize, wilson_lower_bound
from os_datagen.execution.sandbox import Sandbox, SandboxViolation
from os_datagen.llm.scheduler import Phase, PhaseScheduler
from os_datagen.taskpacks.registry import get_pack
from os_datagen.utils.hashing import sha256_file


def test_sandbox_runs_allowlisted_ops_and_blocks_others(tmp_path):
    with Sandbox() as sbx:
        _, truth = make_csv(sbx.root, "sales.csv", "revenue", 7)
        r = sbx.run_op("csv_sum", file="sales.csv", column="revenue")
        assert r.ok and abs(r.artifacts["total"] - truth) < 0.01
        with pytest.raises(SandboxViolation):
            sbx.run_op("shell", cmd="rm -rf /")
        with pytest.raises(SandboxViolation):
            sbx.run_op("csv_sum", file="../../etc/passwd", column="x")
        with pytest.raises(SandboxViolation):
            sbx.run_snippet("import os; os.system('id')")  # not in the fixture allowlist
        with pytest.raises(SandboxViolation):
            sbx.run_op("sqlite_query_readonly", file="x.db", sql="DROP TABLE orders")


def test_wilson_lower_bound_and_routing_policy():
    assert wilson_lower_bound(0, 0) == 0
    assert wilson_lower_bound(88, 100) < 0.88 < 0.95
    assert wilson_lower_bound(5, 5) < 1.0  # small n is not treated as exact
    p = get_pack("harness_next_action_router_v1")
    for i in range(500):
        w = p.sample_world(__import__("random").Random(i), "train")
        if w.facts.get("csv") and w.scenario_family == "exact_local_calculation":
            routes = ["use_python", "call_small_model", "call_large_model"]
            few = summarize(w.scenario_family, w.facts, routes, 40, i)
            assert few.route_outcomes["use_python"].success_rate == 1.0
            assert few.preferred_option is None  # 40/40 => lower bound 0.91 < 0.95: not enough evidence to certify
            s = summarize(w.scenario_family, w.facts, routes, 100, i)
            assert s.preferred_option == "use_python"  # 100/100 => lower bound 0.96 >= 0.95, cheapest qualifying route
            return
    raise AssertionError


def test_split_counts_sum():
    for total in (4, 10, 100, 1001):
        sc = split_counts(total, {"train": 0.7, "calibration": 0.1, "locked_test": 0.1, "challenge": 0.1})
        assert sum(sc.values()) == total and all(v >= 1 for v in sc.values())


def test_scheduler_runs_phases_sequentially_and_warns_on_shared_endpoint():
    cfg = load_config(None, Path(__file__).parents[2] / "configs" / "models.example.yaml")
    order = []
    PhaseScheduler(cfg, lambda p: order.append(("s", p.name)), lambda p: order.append(("e", p.name))).run(
        [Phase("a", "generator", lambda: 1), Phase("b", "verifier", lambda: 2)])
    assert order == [("s", "a"), ("e", "a"), ("s", "b"), ("e", "b")]
    cfg.models["verifier"].base_url = cfg.models["generator"].base_url
    assert PhaseScheduler(cfg).warn_correlated()


def test_config_and_policies_load():
    cfg = load_config(Path(__file__).parents[2] / "configs" / "pipeline.example.yaml", Path(__file__).parents[2] / "configs" / "models.example.yaml")
    assert set(cfg.models) == {"generator", "verifier", "semantic_judge", "triage"}
    assert cfg.models["generator"].temperature == 0.8 and cfg.models["verifier"].temperature == 0.0
    assert policies()["retry"]["default_budget"] == 2
    assert "api_key" not in json.dumps(cfg.public_models())


def test_probability_semantics_declared(pack):
    s = pack.probability_semantics()
    assert s.kind and s.description


def test_manifest_has_required_fields_and_checksums(tmp_path):
    from os_datagen.config import PipelineConfig

    (tmp_path / "accepted_train.jsonl").write_text('{"a": 1}\n')
    m = build_manifest(tmp_path, PipelineConfig(), run_id="r", started="t", dry_run=True, packs={"p": get_pack("foundation_document_type_v1")},
                       prompt_hashes={"x": "y"}, seed=1, counts={"accepted": 1}, split_counts={"train": 1}, namespace_counts={"foundation": 1},
                       tier_counts={"deterministic": 1}, notes=[])
    d = m.model_dump()
    for k in ("run_id", "created_at", "source_code_commit", "config", "config_hash", "models", "prompt_hashes", "taskpacks", "seeds",
              "counts", "split_counts", "namespace_counts", "truth_tier_counts", "files", "license_policy"):
        assert d[k] not in (None, "") or k in ("models",), k
    assert d["files"]["accepted_train.jsonl"]["sha256"] == sha256_file(tmp_path / "accepted_train.jsonl")
    write_checksums(tmp_path)
    assert "accepted_train.jsonl" in (tmp_path / "checksums.sha256").read_text()


def test_redact_scrubs_secret_patterns_and_can_delete(tmp_path):
    from os_datagen.utils.redact import redact_run

    raw = tmp_path / "raw" / "p"
    raw.mkdir(parents=True)
    (raw / "a.json").write_text('[{"text": "key sk-ABCDEFGHIJKLMNOPQRSTUVWX and Bearer abcdefghijklmnopqrstuvwxyz012"}]')
    st = redact_run(tmp_path)
    assert st["redactions"] == 2 and "sk-ABC" not in (raw / "a.json").read_text()
    assert redact_run(tmp_path, delete_raw=True)["deleted"] == 1 and not (tmp_path / "raw").exists()


def test_scheduler_skips_phases_that_have_nothing_to_do_without_starting_their_server():
    cfg = load_config(None, Path(__file__).parents[2] / "configs" / "models.example.yaml")
    started = []
    PhaseScheduler(cfg, lambda p: started.append(p.name), lambda p: None).run(
        [Phase("verify", "verifier", lambda: 1), Phase("judge", "semantic_judge", lambda: 2, when=lambda: False),
         Phase("support", "verifier", lambda: 3, when=lambda: True)])
    assert started == ["verify", "support"]  # the judge phase never started (no model swap for an empty phase)


def test_mixture_balance_hits_targets_and_is_deterministic():
    from os_datagen.datasets.mixture import balance

    rows = {"train": [{"record_id": f"a{i}", "meta": {"composition_family": "A"}} for i in range(80)] +
                     [{"record_id": f"b{i}", "meta": {"composition_family": "B"}} for i in range(30)],
            "locked_test": [{"record_id": f"c{i}", "meta": {"composition_family": "A"}} for i in range(10)] +
                           [{"record_id": f"d{i}", "meta": {"composition_family": "B"}} for i in range(5)],
            "calibration": [], "challenge": []}
    targets = {"A": 0.5, "B": 0.4, "C": 0.1}  # C has no rows: reported, not fillable
    out, rep = balance(rows, targets)
    n = {f: sum(1 for s in out.values() for r in s if r["meta"]["composition_family"] == f) for f in "AB"}
    assert abs(n["A"] / (n["A"] + n["B"]) - 0.5 / 0.9) < 0.05 and rep["unfilled_families"] == {"C": 0.1}
    assert out["locked_test"] and balance(rows, targets)[0] == out  # split-proportional and deterministic
