from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetManifest(BaseModel):
    run_id: str
    created_at: str
    finished_at: str | None = None
    source_code_commit: str
    config_hash: str
    config: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)  # role -> {model, base_url}; no secrets
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    taskpacks: dict[str, dict[str, str]] = Field(default_factory=dict)  # pack -> {version, oracle_version}
    seeds: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, Any] = Field(default_factory=dict)
    split_counts: dict[str, int] = Field(default_factory=dict)
    namespace_counts: dict[str, int] = Field(default_factory=dict)
    truth_tier_counts: dict[str, int] = Field(default_factory=dict)
    files: dict[str, dict[str, Any]] = Field(default_factory=dict)  # relpath -> {sha256, rows}
    license_policy: str = "Synthetic, fictional data only; Apache-2.0; no real personal data."
    dry_run: bool = False
    notes: list[str] = Field(default_factory=list)
