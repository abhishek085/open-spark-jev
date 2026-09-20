"""Deterministic fake LLM for CI / --dry-run ONLY. It renders from the pack's offline template and
echoes the expected facts. It is never evidence of real data quality."""
from __future__ import annotations

import json
from typing import Any

from ..utils.hashing import stable_int
from .client import LLMRequest, LLMResponse


class FakeGenerator:
    def __init__(self, corrupt_rate: float = 0.0, leak_rate: float = 0.0, name: str = "fake-generator"):
        self.model_name, self.base_url = name, "fake://generator"
        self.corrupt_rate, self.leak_rate = corrupt_rate, leak_rate

    def chat(self, req: LLMRequest) -> LLMResponse:
        from ..taskpacks.registry import get_pack

        world = req.metadata["world"]
        pack = get_pack(world.task_pack)
        surface: dict[str, Any] = pack.template_surface(world)
        v = req.metadata.get("variant", 0)
        if v:  # a real sampler renders each variant differently; mimic that so variants are not exact duplicates
            k0 = next(k for k, x in surface.items() if isinstance(x, str) and k in ("document_text", "user_request", "message", "user_goal", "query", "document", "claim"))
            surface[k0] = surface[k0] + f" Revision {v + 1}."
        u = (stable_int("fake-gen", world.scenario_id, v) % 1000) / 1000
        if u < self.corrupt_rate:
            surface.pop(next(iter(surface)))  # break the schema
        elif u < self.corrupt_rate + self.leak_rate:
            k = next(k for k, v in surface.items() if isinstance(v, str))
            surface[k] = surface[k] + f" (answer: {pack.options[0][0] if pack.options else 'yes'})"
        return LLMResponse(text=json.dumps(surface), model=self.model_name)


class FakeVerifier:
    def __init__(self, mismatch_rate: float = 0.0, name: str = "fake-verifier"):
        self.model_name, self.base_url = name, "fake://verifier"
        self.mismatch_rate = mismatch_rate

    def chat(self, req: LLMRequest) -> LLMResponse:
        from ..taskpacks.registry import get_pack

        world = req.metadata["world"]
        if req.metadata.get("mode") == "support":  # plumbing-only: a perfect solver that echoes the oracle label
            return LLMResponse(text=json.dumps({"choice": req.metadata["oracle_label"]}), model=self.model_name)
        pack = get_pack(world.task_pack)
        # Plumbing-only fake: echoes the world's expected facts (it does NOT test verification quality).
        out = dict(pack.expected_extraction(world))
        if "surface_features" in pack.verifier_fields:
            out["surface_features"] = pack.required_surface_features(world)
        u = (stable_int("fake-ver", world.scenario_id) % 1000) / 1000
        if u < self.mismatch_rate and out:
            k = next(iter(out))
            v = out[k]
            out[k] = (not v) if isinstance(v, bool) else ("__changed__" if isinstance(v, str) else v)
        return LLMResponse(text=json.dumps(out), model=self.model_name)


class FakeJudge:
    model_name, base_url = "fake-judge", "fake://judge"

    def chat(self, req: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=json.dumps({"verdict": "human_review", "fact_preservation": "uncertain",
                             "visible_decision_ambiguity": "none", "label_leakage": "none",
                             "reasons": ["fake_judge"], "confidence": 0.5}),
            model=self.model_name,
        )
