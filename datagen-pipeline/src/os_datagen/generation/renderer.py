from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..config import GenerationConfig, RetentionConfig
from ..llm.client import LLMClient, LLMRequest
from ..schemas.scenario import RenderedState, RenderRequest, ScenarioWorld
from ..taskpacks.base import BaseTaskPack
from ..utils.hashing import sha256_file
from ..utils.paths import project_path
from ..validation.schema_validation import validate_surface
from .adversary import addendum
from .variants import variant_seed


class PromptLibrary:
    def __init__(self, root: Path | None = None):
        self.root = root or project_path("prompts")
        self.env = Environment(loader=FileSystemLoader(str(self.root)), undefined=StrictUndefined, autoescape=False,
                               trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)

    def render(self, template: str, ctx: dict[str, Any]) -> str:
        return self.env.get_template(template).render(**{"notes": "", **ctx})

    def lock(self) -> dict[str, str]:
        """sha256 of every prompt template (prompts.lock.json)."""
        return {str(p.relative_to(self.root)): sha256_file(p) for p in sorted(self.root.rglob("*.j2"))}


def redact(text: str, patterns: list[str]) -> str:
    for p in patterns:
        text = re.sub(p, "[REDACTED]", text)
    return text


class SurfaceRenderer:
    def __init__(self, client: LLMClient, prompts: PromptLibrary, gen_cfg: GenerationConfig, retention: RetentionConfig,
                 raw_dir: Path, temperature: float, top_p: float, max_tokens: int, lower_confidence: bool = False):
        self.client, self.prompts, self.gen_cfg, self.retention = client, prompts, gen_cfg, retention
        self.raw_dir, self.temperature, self.top_p, self.max_tokens = raw_dir, temperature, top_p, max_tokens
        self.lower_confidence = lower_confidence

    def _save_raw(self, cand_id: str, pack: str, attempts: list[dict[str, Any]]) -> str | None:
        if not self.retention.keep_raw_generations:
            return None
        d = self.raw_dir / pack
        d.mkdir(parents=True, exist_ok=True)
        clipped = [{**a, "text": redact(a["text"], self.retention.redact_patterns)[: self.retention.max_raw_bytes]} for a in attempts]
        path = d / f"{cand_id}.json"
        path.write_text(json.dumps(clipped, ensure_ascii=False, indent=1))
        return str(path.relative_to(self.raw_dir.parent))

    def render(self, pack: BaseTaskPack, world: ScenarioWorld, req: RenderRequest, cand_id: str, variant: int
               ) -> tuple[RenderedState | None, list[str], str | None]:
        schema = pack.surface_json_schema()
        user = self.prompts.render(req.template_path, req.template_vars)
        extra = addendum(pack, world)
        if extra:
            user += "\n" + extra + "\n"
        messages = [{"role": "system", "content": self.prompts.render("system/generator_system.j2", {})},
                    {"role": "user", "content": user}]
        attempts: list[dict[str, Any]] = []
        issues: list[str] = []
        for attempt in range(max(1, self.gen_cfg.max_render_attempts)):
            llm_req = LLMRequest(messages=messages, temperature=self.temperature, top_p=self.top_p,
                                 max_tokens=self.max_tokens, seed=variant_seed(world, variant) + attempt,
                                 json_schema=schema, metadata={"world": world, "variant": variant})
            try:
                resp = self.client.chat(llm_req)
            except Exception as e:  # noqa: BLE001 - endpoint failure is a rejection with an audit trail
                attempts.append({"attempt": attempt, "text": "", "error": str(e)[:200]})
                issues = [f"generation_failed:{type(e).__name__}"]
                break
            surface, issues = validate_surface(pack, resp.text)
            attempts.append({"attempt": attempt, "text": resp.text, "model": resp.model, "issues": issues,
                             "latency_ms": resp.latency_ms})
            if surface is not None:
                path = self._save_raw(cand_id, pack.name, attempts)
                return RenderedState(surface=surface, generator_model=resp.model, prompt_version=pack.prompt_version,
                                     raw_generation_path=path, attempts=attempt + 1, repair_attempt_count=attempt,
                                     lower_confidence_provenance=self.lower_confidence), [], path
            if not self.gen_cfg.repair_enabled:
                break
            messages = [{"role": "system", "content": self.prompts.render("system/repair_system.j2", {})},
                        {"role": "user", "content": self.prompts.render("system/repair_user.j2", {
                            "schema_json": json.dumps(schema, indent=2), "previous_output": resp.text[:6000],
                            "errors": ", ".join(issues), "original_prompt": user})}]
        return None, issues or ["generation_failed"], self._save_raw(cand_id, pack.name, attempts)
