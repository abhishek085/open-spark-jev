from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .utils.hashing import sha256_obj
from .utils.paths import project_path

ROLES = ("generator", "verifier", "semantic_judge", "triage")


class ModelConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    base_urls: list[str] = Field(default_factory=list)  # several instances of the SAME model; requests are round-robined
    api_key_env: str = "LOCAL_LLM_API_KEY"
    model: str = "local-model"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1000
    concurrency: int = 2  # in-flight requests across all instances of this role
    json_mode: bool = True
    json_schema_mode: bool = True  # use guided json_schema when the endpoint supports it
    disable_thinking: bool = True  # sends chat_template_kwargs.enable_thinking=false
    timeout_s: float = 300.0
    start_cmd: str | None = None  # optional: lets the scheduler start this role's server for its phase
    stop_cmd: str | None = None
    startup_timeout_s: float = 900.0

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "local-not-a-secret")


class SchedulerConfig(BaseModel):
    mode: str = "sequential_phase"
    unload_between_roles: bool = True
    fail_if_memory_headroom_gb_below: float = 16
    generation_batch_size: int = 16
    verification_batch_size: int = 32
    judge_batch_size: int = 8


class GenerationConfig(BaseModel):
    max_render_attempts: int = 2  # first attempt + optional repair
    repair_enabled: bool = True
    variants_per_world: int = 1
    label_balance_slack: float = 1.6  # per-label cap = ceil(n / k * slack)
    oversample_factor: int = 6
    max_easy_share: float = 0.55  # easy worlds are capped at sampling time (rendering rows that the diversity cap would drop is wasted generation)
    holdout_families: bool = True  # calibration + locked_test use scenario families never seen in train
    holdout_max_share: float = 0.34
    supportability: str = "reject"  # reject | tag | off: blind solvability audit by the verifier model


class ValidationConfig(BaseModel):
    dedupe_near_threshold: float = 0.90
    use_judge: bool = True
    judge_max_share: float = 0.25
    fail_closed: bool = True


class RetentionConfig(BaseModel):
    keep_raw_generations: bool = True
    max_raw_bytes: int = 20_000  # per raw file; longer is truncated
    redact_patterns: list[str] = Field(default_factory=lambda: [r"sk-[A-Za-z0-9]{16,}", r"Bearer\s+[A-Za-z0-9._-]{16,}"])


class PipelineConfig(BaseModel):
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    scheduler: SchedulerConfig = SchedulerConfig()
    generation: GenerationConfig = GenerationConfig()
    validation: ValidationConfig = ValidationConfig()
    retention: RetentionConfig = RetentionConfig()
    split_shares: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "calibration": 0.1, "locked_test": 0.1, "challenge": 0.1}
    )

    def config_hash(self) -> str:
        return sha256_obj(self.model_dump(mode="json"))[:16]

    def public_models(self) -> dict[str, dict[str, Any]]:
        """Models by role with no secrets (only env var *names* are ever stored)."""
        return {r: {"model": m.model, "base_url": m.base_url, "instances": len(m.base_urls) or 1, "temperature": m.temperature}
                for r, m in self.models.items()}


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(pipeline_path: Path | str | None = None, models_path: Path | str | None = None) -> PipelineConfig:
    """Load pipeline yaml; models come from the `models:` block of the pipeline file and/or a models yaml."""
    raw: dict[str, Any] = {}
    for p in (models_path, pipeline_path):
        if p:
            with open(p) as f:
                raw = _deep_merge(raw, yaml.safe_load(f) or {})
    return PipelineConfig.model_validate(raw)


def load_policies(path: Path | None = None) -> dict[str, Any]:
    """Policy parameters read by the oracles (configs/policies.yaml)."""
    path = path or project_path("configs", "policies.yaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


_POLICIES: dict[str, Any] | None = None


def policies() -> dict[str, Any]:
    global _POLICIES
    if _POLICIES is None:
        _POLICIES = load_policies()
    return _POLICIES
