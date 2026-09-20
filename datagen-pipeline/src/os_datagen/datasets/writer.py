from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import yaml

from ..utils.jsonl import write_jsonl
from ..validation.acceptance import Stage
from . import lineage, manifest, reporting
from .splitter import FILE_FOR_SPLIT

if TYPE_CHECKING:
    from ..generation.pipeline import Pipeline


def write_run(p: Pipeline, st: dict[str, Any], seed: int, counts: dict[str, int]) -> dict[str, Any]:
    out = p.out
    started = p.started
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(p.cfg.model_dump(mode="json"), sort_keys=True))
    hashes = p.prompts.lock()
    (out / "prompts.lock.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))
    write_jsonl(out / "scenario_worlds.jsonl", ({"world": w.model_dump(mode="json"), "truth": p.truths[w.scenario_id].model_dump(mode="json")} for w in p.worlds))
    write_jsonl(out / "candidates_raw.jsonl", ({
        "candidate_id": c.candidate_id, "scenario_id": c.world.scenario_id, "task_pack": c.world.task_pack,
        "split": c.world.split, "variant": c.variant, "stage": c.stage.value, "generator_model": c.generator_model,
        "raw_generation_path": c.raw_path, "repair_attempt_count": c.repair_attempts, "surface": c.surface} for c in p.cands))
    recs = st["records"]
    for split, fname in FILE_FOR_SPLIT.items():
        write_jsonl(out / fname, recs[split])
    rejected = [c for c in p.cands if not c.alive]
    write_jsonl(out / "rejected.jsonl", (c.rejection(p.packs[c.world.task_pack].prompt_version) for c in rejected))
    exec_rows: list[dict[str, Any]] = []
    for w in p.worlds:
        hook = getattr(p.packs[w.task_pack], "execute", None)
        if hook:
            exec_rows += [r.model_dump(mode="json") for r in hook(w)]
    write_jsonl(out / "execution_results.jsonl", exec_rows)
    write_jsonl(out / "validation_results.jsonl", ({
        "candidate_id": c.candidate_id, "task_pack": c.world.task_pack, "stage": c.stage.value, "reasons": c.reasons,
        "leak_audit": c.trace.get("leak_audit", []), "verification": c.trace.get("verification", {}),
        "judge": c.trace.get("judge"), "support": c.trace.get("support"), "repair_attempt_count": c.repair_attempts} for c in p.cands))
    all_recs = [r for s in recs.values() for r in s]
    by_id = st["by_cand"]
    write_jsonl(out / "lineage.jsonl", (lineage.lineage_row(r, by_id[r.record_id], hashes, f"{p.packs[r.task_pack].prompt_dir}/surface.j2") for r in all_recs))
    extra = {"judge_routed": p.judge_routed, "human_review_routed": p.human_review,
             "lower_confidence_provenance": bool(p.correlated), "dry_run": p.dry_run}
    summary = reporting.build_reports(out, cands=p.cands, records=all_recs, clusters=st["clusters"], isolation=st["isolation"],
                                      packs=p.packs, extra=extra)
    from collections import Counter
    counts_out = {"candidates_produced": len(p.cands), "accepted": len(all_recs), "rejected": len(rejected),
                  "judge_routed": p.judge_routed, "human_review_routed": p.human_review, "worlds": len(p.worlds)}
    notes = ["fake generator/verifier used: plumbing test only, not evidence of data quality"] if p.dry_run else []
    if p.correlated:
        notes.append(p.correlated)
    m = manifest.build_manifest(
        out, p.cfg, run_id=out.name, started=started or "", dry_run=p.dry_run, packs=p.packs, prompt_hashes=hashes, seed=seed,
        counts=counts_out, split_counts=dict(Counter(r.split for r in all_recs)),
        namespace_counts=dict(Counter(r.meta.get("namespace") for r in all_recs)),
        tier_counts=dict(Counter(r.truth.label_quality for r in all_recs)), notes=notes)
    (out / "manifest.json").write_text(m.model_dump_json(indent=2))
    manifest.write_checksums(out)
    summary["stage_counts"] = dict(Counter(c.stage.value for c in p.cands))
    return summary


__all__ = ["write_run", "Stage"]
