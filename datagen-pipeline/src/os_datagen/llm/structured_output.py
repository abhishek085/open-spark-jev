from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(ValueError):
    pass


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def extract_json(text: str) -> dict[str, Any]:
    """Strict-ish JSON parse. Only strips a code fence and leading/trailing whitespace; it never
    guesses at free-form prose (no silent parsing in the default path)."""
    s = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise StructuredOutputError(f"invalid_json: {e.msg}") from e
    if not isinstance(obj, dict):
        raise StructuredOutputError("invalid_json: top-level value is not an object")
    return obj


def parse_model(text: str, model: type[T]) -> T:
    obj = extract_json(text)
    try:
        return model.model_validate(obj)
    except ValidationError as e:
        errs = ";".join(f"{'.'.join(str(p) for p in x['loc'])}:{x['type']}" for x in e.errors()[:6])
        raise StructuredOutputError(f"schema: {errs}") from e
