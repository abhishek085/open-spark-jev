from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..llm.structured_output import StructuredOutputError, extract_json
from ..taskpacks.base import BaseTaskPack


def validate_surface(pack: BaseTaskPack, raw_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Strict JSON parse + Pydantic validation (extra fields forbidden). No free-form parsing."""
    try:
        obj = extract_json(raw_text)
    except StructuredOutputError as e:
        return None, [f"schema:{e}".split(":", 2)[0] + ":" + str(e).split(":")[0]]
    try:
        model = pack.surface_model().model_validate(obj)
    except ValidationError as e:
        codes = []
        for x in e.errors()[:8]:
            loc = ".".join(str(p) for p in x["loc"])
            kind = {"missing": "missing", "extra_forbidden": "extra_field"}.get(x["type"], "invalid")
            codes.append(f"schema:{kind}:{loc}")
        return None, codes
    return model.model_dump(), []
