"""Write a code-only pack run in the same layout as an LLM-backed run (accepted_<split>.jsonl of DatasetRecord, manifest, checksums)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...schemas.decision import DatasetRecord, DecisionSpec, PublicTruth, QualityRecord
from ...schemas.provenance import ProvenanceRecord
from . import tool_call_risk as T

PACK = "harness_tool_call_risk_v1"
FILES = {"train": "accepted_train.jsonl", "calibration": "accepted_calibration.jsonl", "locked_test": "accepted_test_locked.jsonl", "challenge": "accepted_challenge.jsonl"}


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def to_record(row: dict[str, Any], seed: int, when: str, commit: str) -> DatasetRecord:
    rid = f"osj-tcr-{row['rid']}"
    truth = PublicTruth(preferred_option=row["posture"], acceptable_options=[row["posture"]], label_source="tool_call_risk_grammar_v1", oracle_version=T.ORACLE_VERSION,
                        reason_code=row["template"], label_quality="deterministic")
    prov = ProvenanceRecord(synthetic=True, scenario_id=rid, scenario_seed=seed, generator_model="code-only", generator_prompt_version="tool_call_risk.grammar.v1",
                            created_at=when, source_code_commit=commit)
    quality = QualityRecord(difficulty="hard" if row["frame_kind"] == "misleading" else "medium", tags=[row["frame_kind"], row["tool"]], semantic_verification="skipped",
                            dedupe_status="unique", supportability="skipped", is_challenge=row["split"] == "challenge")
    opts = [{"id": p, "definition": T.DEFINITIONS[p]} for p in row["option_order"]]
    return DatasetRecord(
        record_id=rid, task_pack=PACK, task_version="1.0.0", split=row["split"],
        decision=DecisionSpec(type="choice", state={"text": T.state_text(row["frame"], row["command"])}, question={"instructions": row["instructions"], "options": opts}),
        truth=truth, quality=quality, provenance=prov,
        meta={"namespace": "harness", "scenario_family": row["group"], "split_family": row["split"], "composition_family": "agent_harness", "pool": row["split"],
              "variant": 0, "pair_id": row["pair_id"], "frame_kind": row["frame_kind"], "tool": row["tool"], "template": row["template"], "features": row["features"]},
    )


def write_run(out: Path, count: int, seed: int) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    rows = T.make_rows(count, seed)
    when = datetime.now(timezone.utc).isoformat()
    commit = _commit()
    per: dict[str, list[dict]] = {k: [] for k in FILES}
    for r in rows:
        per[r["split"]].append(to_record(r, seed, when, commit).model_dump(mode="json"))
    sums = {}
    for split, fname in FILES.items():
        (out / fname).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in per[split]))
        sums[fname] = hashlib.sha256((out / fname).read_bytes()).hexdigest()
    manifest = {"run_id": out.name, "pack": PACK, "generator": "code-only grammar (no LLM)", "oracle_version": T.ORACLE_VERSION, "seed": seed, "created_at": when,
                "counts": {k: len(v) for k, v in per.items()}, "postures": {k: dict(Counter(x["truth"]["preferred_option"] for x in v)) for k, v in per.items()},
                "unique_commands": len({r["command"] for r in rows}), "pairs": len({r["pair_id"] for r in rows if r["pair_id"]}),
                "note": "Labels are computed by rules from declared command features; framings never affect the label."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "checksums.sha256").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    return manifest


# ---------------------------------------------------------------- structured Jev-use-case packs
def _use_case_record(r: dict[str, Any], seed: int, when: str, commit: str) -> DatasetRecord:
    from ...schemas.decision import DecisionSpec  # noqa: F401
    from . import jev_use_cases as J

    rid = f"osj-{r['rid']}"
    rng = __import__("random").Random(f"{rid}-{seed}")
    q: dict[str, Any] = {"instructions": J.instr(r["instr"], r["split"], rng)}
    if r["type"] == "choice":
        opts = list(r["options"])
        rng.shuffle(opts)
        q["options"] = [{"id": i, "definition": d} for i, d in opts]
        truth = PublicTruth(preferred_option=r["label"], acceptable_options=[r["label"]], label_source="jev_use_case_rules_v1", oracle_version=J.ORACLE, reason_code=str(r["features"]), label_quality="deterministic")
    elif r["type"] == "score":
        q["levels"] = [{"value": v, "definition": d} for v, d in r["levels"]]
        truth = PublicTruth(preferred_level=r["label"], acceptable_levels=[r["label"]], label_source="jev_use_case_rules_v1", oracle_version=J.ORACLE, reason_code=str(r["features"]), label_quality="deterministic")
    else:
        truth = PublicTruth(truth=bool(r["label"]), label_source="jev_use_case_rules_v1", oracle_version=J.ORACLE, reason_code=str(r["features"]), label_quality="deterministic")
    prov = ProvenanceRecord(synthetic=True, scenario_id=rid, scenario_seed=seed, generator_model="code-only", generator_prompt_version=f"{r['pack']}.rules.v1", created_at=when, source_code_commit=commit)
    quality = QualityRecord(difficulty="medium", tags=[r["pack"]], semantic_verification="skipped", dedupe_status="unique", supportability="skipped", is_challenge=r["split"] == "challenge")
    return DatasetRecord(record_id=rid, task_pack=r["pack"], task_version="1.0.0", split=r["split"], decision=DecisionSpec(type=r["type"], state=r["state"], question=q),
                         truth=truth, quality=quality, provenance=prov,
                         meta={"namespace": "harness", "scenario_family": r["pack"], "split_family": r["split"], "composition_family": J.PACKS[r["pack"]][1], "pool": f"pool{r['pool']}", "variant": 0})


def write_use_case_runs(out: Path, count_per_pack: int, seed: int, packs: list[str] | None = None) -> dict[str, Any]:
    from . import jev_use_cases as J

    out.mkdir(parents=True, exist_ok=True)
    when = datetime.now(timezone.utc).isoformat()
    commit = _commit()
    per: dict[str, list[dict]] = {k: [] for k in FILES}
    for pack in packs or list(J.PACKS):
        for r in J.make_rows(pack, count_per_pack, seed):
            per[r["split"]].append(_use_case_record(r, seed, when, commit).model_dump(mode="json"))
    sums = {}
    for split, fname in FILES.items():
        (out / fname).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in per[split]))
        sums[fname] = hashlib.sha256((out / fname).read_bytes()).hexdigest()
    manifest = {"run_id": out.name, "packs": packs or list(J.PACKS), "generator": "code-only rules (no LLM)", "oracle_version": J.ORACLE, "seed": seed, "created_at": when,
                "counts": {k: len(v) for k, v in per.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "checksums.sha256").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()))
    return manifest
