"""Deterministic validators used as truth sources (tool-argument schemas, JSON checks, path safety)."""
from __future__ import annotations

import json
import re
from typing import Any

_TYPES: dict[str, type | tuple[type, ...]] = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate_arguments(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """schema = {"required": [...], "types": {name: json-type}}. Returns issue codes ([] = schema-valid)."""
    if "_raw_arguments" in args:
        try:
            json.loads(args["_raw_arguments"])
        except (ValueError, TypeError):
            return ["malformed_json"]
    issues = [f"missing_required:{k}" for k in schema.get("required", []) if k not in args or args[k] in (None, "")]
    for k, t in schema.get("types", {}).items():
        if k in args and args[k] is not None:
            ok = isinstance(args[k], _TYPES[t]) and not (t in ("integer", "number") and isinstance(args[k], bool))
            if not ok:
                issues.append(f"wrong_type:{k}")
    return issues


_TRAVERSAL = re.compile(r"(\.\./|\.\.\\|^/etc/|^/root/|~/\.ssh)")


def has_path_traversal(args: dict[str, Any]) -> bool:
    return any(isinstance(v, str) and _TRAVERSAL.search(v) for v in args.values())


def json_schema_valid(instance: Any, schema: dict[str, Any]) -> list[str]:
    """Tiny JSON-Schema subset: type, required, properties, items (enough for fixtures)."""
    errs: list[str] = []
    t = schema.get("type")
    if t and not isinstance(instance, _TYPES[t]):
        return [f"type:{t}"]
    if isinstance(instance, dict):
        errs += [f"required:{k}" for k in schema.get("required", []) if k not in instance]
        for k, sub in schema.get("properties", {}).items():
            if k in instance:
                errs += [f"{k}.{e}" for e in json_schema_valid(instance[k], sub)]
    if isinstance(instance, list) and "items" in schema:
        for i, x in enumerate(instance):
            errs += [f"[{i}].{e}" for e in json_schema_valid(x, schema["items"])]
    return errs
