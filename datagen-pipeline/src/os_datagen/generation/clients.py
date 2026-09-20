from __future__ import annotations

from ..config import PipelineConfig
from ..llm.fake_client import FakeGenerator, FakeJudge, FakeVerifier
from ..llm.openai_compatible import OpenAICompatibleClient


def make_clients(cfg: PipelineConfig, dry_run: bool, corrupt: float = 0.0, leak: float = 0.0, mismatch: float = 0.0):
    """Returns (generator, verifier, judge|None). --dry-run uses the deterministic fakes (CI plumbing only)."""
    if dry_run:
        return FakeGenerator(corrupt, leak), FakeVerifier(mismatch), FakeJudge()
    for r in ("generator", "verifier"):
        if r not in cfg.models:
            raise SystemExit(f"config has no models.{r}; pass --config/--models (see configs/models.example.yaml)")
    gen = OpenAICompatibleClient(cfg.models["generator"])
    ver = OpenAICompatibleClient(cfg.models["verifier"])
    judge = OpenAICompatibleClient(cfg.models["semantic_judge"]) if "semantic_judge" in cfg.models else None
    return gen, ver, judge
