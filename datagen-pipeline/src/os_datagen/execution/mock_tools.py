"""Mock tools with predefined behavior (no real network / side effects). Used to verify tool-gate truth."""
from __future__ import annotations

from typing import Any

from .validators import has_path_traversal, validate_arguments


class ToolError(RuntimeError):
    pass


class MockToolRegistry:
    def __init__(self) -> None:
        self.effects: list[dict[str, Any]] = []  # recorded (mock) side effects
        self.tools: dict[str, dict[str, Any]] = {
            "search_notes": {"schema": {"required": ["query"], "types": {"query": "string", "limit": "integer"}}, "effect": "none"},
            "read_file": {"schema": {"required": ["path"], "types": {"path": "string"}}, "effect": "none"},
            "send_message": {"schema": {"required": ["recipient", "body"], "types": {"recipient": "string", "body": "string"}}, "effect": "external_write"},
        }

    def call(self, name: str, args: dict[str, Any], *, approved: bool = False, available: bool = True) -> dict[str, Any]:
        if name not in self.tools or not available:
            raise ToolError("unavailable")
        spec = self.tools[name]
        issues = validate_arguments(spec["schema"], args)
        if issues:
            raise ToolError("invalid_arguments:" + issues[0])
        if has_path_traversal(args):
            raise ToolError("path_traversal")
        if spec["effect"] == "external_write":
            if not approved:
                raise ToolError("approval_required")
            self.effects.append({"tool": name, "args": args})
            return {"status": "sent(mock)"}
        return {"status": "ok", "echo": args}
