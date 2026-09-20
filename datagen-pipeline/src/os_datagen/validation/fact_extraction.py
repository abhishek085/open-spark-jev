from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from ..generation.renderer import PromptLibrary
from ..llm.client import LLMClient, LLMRequest
from ..llm.structured_output import StructuredOutputError, parse_model
from ..schemas.scenario import ScenarioWorld
from ..taskpacks.base import BaseTaskPack


def extract_facts(pack: BaseTaskPack, world: ScenarioWorld, state: dict[str, Any], client: LLMClient,
                  prompts: PromptLibrary, max_tokens: int = 900, temperature: float = 0.0
                  ) -> tuple[BaseModel | None, str, str | None]:
    """Independent verifier call. It sees the visible state and an extraction schema, never the truth or world."""
    schema = pack.verifier_schema()
    ctx = pack.verifier_context(state)
    ctx["schema_json"] = json.dumps(schema.model_json_schema(), indent=2)
    ctx["task_family"] = pack.name
    user = prompts.render(f"{pack.prompt_dir}/verify.j2", ctx)
    req = LLMRequest(
        messages=[{"role": "system", "content": prompts.render("system/verifier_system.j2", {})},
                  {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens, seed=world.seed, json_schema=schema.model_json_schema(),
        metadata={"world": world},
    )
    raw = client.chat(req).text
    try:
        return parse_model(raw, schema), raw, None
    except StructuredOutputError as e:
        return None, raw, f"verifier_unparseable:{str(e)[:60]}"
