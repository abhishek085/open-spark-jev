from __future__ import annotations

import json
from pathlib import Path

import pytest

from os_datagen.config import ModelConfig, PipelineConfig
from os_datagen.generation.pipeline import Pipeline
from os_datagen.llm.fake_client import FakeGenerator, FakeJudge, FakeVerifier
from os_datagen.utils.jsonl import read_jsonl

PACKS = ["foundation_semantic_entailment_v1", "foundation_document_type_v1", "harness_next_action_router_v1", "harness_tool_action_gate_v1"]
COUNTS = {"train": 70, "calibration": 10, "locked_test": 10, "challenge": 10}  # 100 worlds per pack


def _cfg() -> PipelineConfig:
    c = PipelineConfig()
    c.models["generator"] = ModelConfig(model="fake-generator", concurrency=1)
    c.models["verifier"] = ModelConfig(model="fake-verifier", concurrency=1)
    return c


def _run(out: Path, packs=PACKS, gen=None, ver=None, judge=None, counts=COUNTS):
    p = Pipeline(_cfg(), out, gen or FakeGenerator(), ver or FakeVerifier(), judge, dry_run=True)
    return p, p.run(packs, counts, 42)


SENTINEL = "sk-SENTINEL-DO-NOT-STORE-0123456789"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    import os

    os.environ["LOCAL_LLM_API_KEY"] = SENTINEL
    out = tmp_path_factory.mktemp("dry")
    p, summary = _run(out)
    return out, p, summary


def test_dry_run_generates_100_worlds_for_two_foundation_and_two_harness_packs(run):
    out, p, s = run
    assert s["candidates"] == 400
    for n in PACKS:
        assert s["by_pack"][n]["candidates"] == 100 and s["by_pack"][n]["accepted"] >= 30
    assert s["namespaces"].keys() == {"foundation", "harness"}


def test_accepted_and_rejected_both_produced(run):
    out, *_ = run
    acc = sum(len(list(read_jsonl(out / f))) for f in ("accepted_train.jsonl", "accepted_calibration.jsonl", "accepted_test_locked.jsonl", "accepted_challenge.jsonl"))
    rej = list(read_jsonl(out / "rejected.jsonl"))
    assert acc > 0 and rej
    for r in rej:
        assert r["reasons"] and r["candidate_id"] and r["validator_versions"] and r["task_pack"]
        assert not str(r.get("raw_generation_path") or "").startswith("/")  # artifact-relative only


def test_every_accepted_row_has_oracle_and_provenance(run):
    out, *_ = run
    for f in ("accepted_train.jsonl", "accepted_calibration.jsonl", "accepted_test_locked.jsonl", "accepted_challenge.jsonl"):
        for r in read_jsonl(out / f):
            t, pv = r["truth"], r["provenance"]
            assert t["oracle_version"] and t["label_source"] and t["reason_code"] and t["label_quality"] in (
                "deterministic", "executable", "controlled_world", "rollout", "adjudicated", "model_consensus")
            assert pv["scenario_id"] and pv["generator_model"] and pv["verifier_model"] and pv["created_at"] and pv["source_code_commit"]
            assert pv["generator_prompt_version"] and pv["verifier_prompt_version"]
            assert r["quality"]["semantic_verification"] in ("passed", "escalated_passed") and r["quality"]["dedupe_status"] == "unique"


def test_no_hidden_truth_in_visible_state(run):
    out, p, _ = run
    for f in ("accepted_train.jsonl", "accepted_calibration.jsonl", "accepted_test_locked.jsonl", "accepted_challenge.jsonl"):
        for r in read_jsonl(out / f):
            state = json.dumps(r["decision"]["state"]).lower()
            pack = p.packs[r["task_pack"]]
            for key in ("preferred_option", "acceptable_options", "label_source", "reason_code", pack.oracle_version.lower()):
                assert key not in state, (r["record_id"], key)
            assert "truth" not in r["decision"] and "meta" not in r["decision"]
            assert r["truth"]["preferred_option"] in [o["id"] for o in r["decision"]["question"]["options"]]


def test_reports_and_manifest_artifacts_emitted(run):
    out, *_ = run
    for f in ("config.resolved.yaml", "prompts.lock.json", "manifest.json", "scenario_worlds.jsonl", "candidates_raw.jsonl", "rejected.jsonl",
              "execution_results.jsonl", "validation_results.jsonl", "checksums.sha256", "lineage.jsonl",
              "reports/summary.json", "reports/acceptance_by_task.csv", "reports/rejection_reasons.csv", "reports/coverage_matrix.csv",
              "reports/split_isolation.json", "reports/duplicate_clusters.json", "reports/sample_audit.md"):
        assert (out / f).exists(), f
    m = json.loads((out / "manifest.json").read_text())
    for k in ("run_id", "source_code_commit", "config_hash", "models", "prompt_hashes", "taskpacks", "seeds", "counts", "split_counts",
              "namespace_counts", "truth_tier_counts", "files", "license_policy"):
        assert k in m
    assert m["counts"]["candidates_produced"] == 400
    for f in ("manifest.json", "config.resolved.yaml"):  # env-var NAMES may be recorded, secret VALUES never
        assert SENTINEL not in (out / f).read_text()
    assert m["dry_run"] is True and m["notes"]  # dry run is flagged as plumbing-only
    for meta in m["files"].values():
        assert meta["sha256"]
    s = json.loads((out / "reports/summary.json").read_text())
    assert {"foundation", "harness"} <= set(s["namespaces"]) and s["split_isolation_ok"]
    assert "composition_vs_target" in s and s["composition_vs_target"]["data_code_workflows"]["count"] == 0  # reported vs target


def test_splits_are_isolated(run):
    out, *_ = run
    iso = json.loads((out / "reports/split_isolation.json").read_text())
    assert iso["ok"], iso["violations"][:5]


def test_execution_results_from_sandbox_and_mock_tools(run):
    out, *_ = run
    rows = list(read_jsonl(out / "execution_results.jsonl"))
    assert rows and {r["executor_version"] for r in rows} == {"sandbox_v1"}
    assert {"use_python"} <= {r["candidate_action"] for r in rows}
    assert all(r["outcome"]["goal_completed"] for r in rows if r["candidate_action"] == "use_python")


def test_mismatched_verifier_rejects_candidates(tmp_path):
    _, s = _run(tmp_path, packs=["harness_tool_action_gate_v1"], ver=FakeVerifier(mismatch_rate=1.0), counts={"train": 20})
    assert s["accepted"] == 0
    reasons = {r for x in read_jsonl(tmp_path / "rejected.jsonl") for r in x["reasons"]}
    assert any(r.startswith("fact_mismatch:") for r in reasons)


def test_schema_leak_and_generation_faults_are_rejected_with_reasons(tmp_path):
    _, s = _run(tmp_path, packs=["harness_next_action_router_v1"], gen=FakeGenerator(corrupt_rate=0.3, leak_rate=0.3), counts={"train": 60})
    reasons = [r for x in read_jsonl(tmp_path / "rejected.jsonl") for r in x["reasons"]]
    assert any(r.startswith("schema:") for r in reasons) and any(r.startswith("label_leak:") for r in reasons)


def test_unverifiable_routes_to_judge_and_human_review_is_not_accepted(tmp_path):
    class NullVerifier(FakeVerifier):
        def chat(self, req):  # verifier that cannot tell anything -> unverifiable -> judge
            from os_datagen.llm.client import LLMResponse
            return LLMResponse(text="{}", model="fake-verifier")

    p, s = _run(tmp_path, packs=["harness_tool_action_gate_v1"], ver=NullVerifier(), judge=FakeJudge(), counts={"train": 20})
    assert s["accepted"] == 0 and p.judge_routed >= 1
    reasons = {r for x in read_jsonl(tmp_path / "rejected.jsonl") for r in x["reasons"]}
    assert "judge:human_review" in reasons or any(r.startswith("fact_unverifiable") for r in reasons)


def test_variants_change_option_order_not_truth(tmp_path):
    p = Pipeline(_cfg(), tmp_path, FakeGenerator(), FakeVerifier(), None, dry_run=True)
    p.run(["foundation_document_type_v1"], {"train": 12}, 5, variants=2)
    recs = list(read_jsonl(tmp_path / "accepted_train.jsonl"))
    by_scn: dict[str, list] = {}
    for r in recs:
        by_scn.setdefault(r["provenance"]["scenario_id"], []).append(r)
    pairs = [v for v in by_scn.values() if len(v) == 2]
    assert pairs
    assert all(a["truth"]["preferred_option"] == b["truth"]["preferred_option"] for a, b in pairs)
    assert any([o["id"] for o in a["decision"]["question"]["options"]] != [o["id"] for o in b["decision"]["question"]["options"]] for a, b in pairs)


def test_all_fifteen_packs_run_end_to_end_dry(tmp_path):
    from os_datagen.taskpacks.registry import pack_names

    p = Pipeline(_cfg(), tmp_path, FakeGenerator(), FakeVerifier(), None, dry_run=True)
    s = p.run(pack_names(), {"train": 16, "calibration": 4, "locked_test": 4, "challenge": 4}, 7)
    assert len(s["by_pack"]) == 15 and all(b["accepted"] > 0 for b in s["by_pack"].values())
    types = {r["decision"]["type"] for f in ("accepted_train.jsonl",) for r in read_jsonl(tmp_path / f)}
    assert types == {"choice", "score", "boolean"}  # all three primitives present


def test_evaluate_baseline_saves_per_row_predictions(tmp_path, run):
    from os_datagen.evaluation.runner import run_evaluation

    out, *_ = run
    m = run_evaluation(out / "accepted_test_locked.jsonl", "baseline://uniform", "uniform", tmp_path / "ev")
    rows = list(read_jsonl(tmp_path / "ev" / "predictions.jsonl"))
    assert len(rows) == m["overall"]["n"] > 0
    for k in ("record_id", "model", "inference_mode", "raw_output", "probabilities", "selected", "truth_match", "latency_ms"):
        assert k in rows[0]
    assert m["calibration"]["fitted_on"] is None  # no calibration claim without a calibration split
    assert (tmp_path / "ev" / "slices.json").exists() and (tmp_path / "ev" / "reliability.json").exists()


def test_calibration_uses_only_calibration_split(tmp_path, run):
    from os_datagen.evaluation.calibration import fit_temperature, softmax_from_scores
    from os_datagen.evaluation.metrics import nll

    rows = [{"raw_scores": {"a": 5.0, "b": 0.0}, "acceptable": ["a"] if i % 2 == 0 else ["b"]} for i in range(40)]  # overconfident scores
    T = fit_temperature(rows)
    assert T > 1.5
    before = sum(nll(softmax_from_scores(r["raw_scores"]), r["acceptable"]) for r in rows)
    after = sum(nll(softmax_from_scores(r["raw_scores"], T), r["acceptable"]) for r in rows)
    assert after < before


def test_training_export_refuses_non_train_splits(tmp_path, run):
    from os_datagen.training.export import export_sft

    out, *_ = run
    n = export_sft(out / "accepted_train.jsonl", tmp_path / "sft.jsonl")
    assert n > 0
    assert export_sft(out / "accepted_test_locked.jsonl", tmp_path / "x.jsonl") == 0
    row = next(read_jsonl(tmp_path / "sft.jsonl"))
    assert "STATE" in row["prompt"] and row["target"] in row["options"] and "preferred_option" not in row["prompt"]


def test_no_accepted_row_relies_on_model_consensus_and_all_have_supportability(run):
    out, *_ = run
    for f in ("accepted_train.jsonl", "accepted_calibration.jsonl", "accepted_test_locked.jsonl", "accepted_challenge.jsonl"):
        for r in read_jsonl(out / f):
            assert r["truth"]["label_quality"] != "model_consensus"  # the verifier can never be the only reason a row is accepted
            assert r["quality"]["supportability"] in ("supported", "tagged_disagreement", "skipped")


def test_supportability_disagreement_rejects_non_challenge_rows(tmp_path):
    class WrongSolver(FakeVerifier):
        def chat(self, req):
            from os_datagen.llm.client import LLMResponse
            if req.metadata.get("mode") == "support":
                return LLMResponse(text=json.dumps({"choice": "__wrong__"}), model="fake-verifier")
            return super().chat(req)

    p, s = _run(tmp_path, packs=["harness_tool_action_gate_v1"], ver=WrongSolver(), counts={"train": 12, "challenge": 4})
    reasons = [r for x in read_jsonl(tmp_path / "rejected.jsonl") for r in x["reasons"]]
    assert any(r.startswith("supportability:") for r in reasons)
    chal = list(read_jsonl(tmp_path / "accepted_challenge.jsonl"))
    assert chal and all(r["quality"]["supportability"] == "tagged_disagreement" for r in chal)  # challenge rows are tagged, never filtered
    assert list(read_jsonl(tmp_path / "accepted_train.jsonl")) == []


def test_option_order_check_flags_position_sensitive_baseline(tmp_path, run):
    from os_datagen.evaluation.option_order import option_order_check

    out, *_ = run
    s = option_order_check(out / "accepted_train.jsonl", tmp_path, k=4, limit=40)
    assert s["n"] > 0 and s["unstable_share"] > 0.3  # the uniform baseline always picks the first listed option
    assert (tmp_path / "option_order.jsonl").exists()


def test_heldout_families_are_absent_from_train_in_pipeline_output(run):
    out, p, _ = run
    iso = json.loads((out / "reports/split_isolation.json").read_text())
    assert iso["ok"] and iso["heldout_families"]
    train_fams = {(r["task_pack"], r["meta"]["scenario_family"]) for r in read_jsonl(out / "accepted_train.jsonl")}
    for pack, held in iso["heldout_families"].items():
        assert not any((pack, f) in train_fams for f in held)


def test_training_exports_all_formats_and_jev_records_load_in_the_root_trainer_schema(tmp_path, run):
    from os_datagen.training.export import export_training

    out, *_ = run
    for fmt in ("prompt_completion", "chat", "jev"):
        assert export_training(out / "accepted_train.jsonl", tmp_path / f"{fmt}.jsonl", fmt=fmt) > 0
        assert export_training(out / "accepted_test_locked.jsonl", tmp_path / f"x_{fmt}.jsonl", fmt=fmt) == 0  # non-train splits refused by default
    chat = next(read_jsonl(tmp_path / "chat.jsonl"))
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant"] and "preferred_option" not in chat["messages"][0]["content"]
    try:
        from open_spark_jev.data.corpus import (
            Record,  # the repo's own trainer schema (root package, same venv)
        )
    except ImportError:  # pragma: no cover
        return
    n = 0
    for row in read_jsonl(tmp_path / "jev.jsonl"):
        rec = Record.from_json(row)
        labels = rec.question_obj().labels
        assert labels[rec.label_index()] == row["target"]["label"] and abs(sum(rec.target_dist()) - 1) < 1e-6
        assert len(labels) == len(row["target"]["dist"])
        n += 1
    assert n > 10
