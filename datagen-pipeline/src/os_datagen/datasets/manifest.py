from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import PipelineConfig
from ..schemas.common import utc_now
from ..schemas.dataset import DatasetManifest
from ..utils.hashing import sha256_file
from ..utils.jsonl import read_jsonl
from ..utils.paths import git_commit


def file_entries(out: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("checksums.sha256", "manifest.json") and "raw" not in p.relative_to(out).parts:
            rel = str(p.relative_to(out))
            rows = sum(1 for _ in read_jsonl(p)) if p.suffix == ".jsonl" else None
            entries[rel] = {"sha256": sha256_file(p), "rows": rows}
    return entries


def build_manifest(out: Path, cfg: PipelineConfig, *, run_id: str, started: str, dry_run: bool, packs: dict[str, Any],
                   prompt_hashes: dict[str, str], seed: int, counts: dict[str, Any], split_counts: dict[str, int],
                   namespace_counts: dict[str, int], tier_counts: dict[str, int], notes: list[str]) -> DatasetManifest:
    return DatasetManifest(
        run_id=run_id, created_at=started, finished_at=utc_now(), source_code_commit=git_commit(),
        config_hash=cfg.config_hash(), config=cfg.model_dump(mode="json"), models=cfg.public_models(),
        prompt_hashes=prompt_hashes,
        taskpacks={n: {"version": p.version, "oracle_version": p.oracle_version} for n, p in packs.items()},
        seeds={"base_seed": seed, "method": "per-world seed = sha256(base_seed|pack|split|index) mod 2^31; option order and "
                                            "instruction wording seeded from (world seed, variant)"},
        counts=counts, split_counts=split_counts, namespace_counts=namespace_counts, truth_tier_counts=tier_counts,
        files=file_entries(out), dry_run=dry_run, notes=notes,
    )


def write_checksums(out: Path) -> None:
    lines = [f"{sha256_file(p)}  {p.relative_to(out)}" for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "checksums.sha256" and "raw" not in p.relative_to(out).parts]
    (out / "checksums.sha256").write_text("\n".join(lines) + "\n")
