from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..taskpacks.base import BaseTaskPack

FORBIDDEN_KEYS = re.compile(r"(preferred|acceptable|gold|truth|label|oracle|reason_code|answer_key|correct_(?:answer|option)|primary_(?:act|intent)|^intent$|urgency_level|^target$|^target_(?:option|level|label|value)$)", re.I)


def flatten(obj: Any, keys: bool = True) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if keys:
                out.append(str(k))
            out += flatten(v, keys)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += flatten(v, keys)
    elif obj is not None:
        out.append(str(obj))
    return out


@dataclass
class LeakageResult:
    codes: list[str] = field(default_factory=list)
    matches: list[dict[str, str]] = field(default_factory=list)  # audit trail incl. allowed (false-positive) matches

    @property
    def ok(self) -> bool:
        return not self.codes


def check_leakage(pack: BaseTaskPack, state: dict[str, Any], allow_patterns: list[str] | None = None,
                  extra_deny: list[str] | None = None) -> LeakageResult:
    """Deterministic label-leak detection over the visible state (keys and values)."""
    res = LeakageResult()
    allow = [re.compile(a, re.I) for a in (allow_patterns or [])]
    # field names that look like gold-label fields
    def walk_keys(o: Any) -> list[str]:
        if isinstance(o, dict):
            return [str(k) for k in o] + [x for v in o.values() for x in walk_keys(v)]
        if isinstance(o, list):
            return [x for v in o for x in walk_keys(v)]
        return []

    for k in walk_keys(state):
        if FORBIDDEN_KEYS.search(k):
            res.codes.append(f"label_leak:gold_field_name:{k}")
    text = "\n".join(flatten(state, keys=True))
    for pat in pack.leak_patterns() + list(extra_deny or []):
        for m in re.finditer(pat, text, re.I):
            snippet = text[max(0, m.start() - 20): m.end() + 20].replace("\n", " ")
            allowed = any(a.search(m.group(0)) or a.search(snippet) for a in allow)
            res.matches.append({"pattern": pat, "match": m.group(0), "context": snippet, "allowed": str(allowed)})
            if not allowed:
                res.codes.append(f"label_leak:{re.sub(r'[^a-z0-9_]+', '_', m.group(0).lower()).strip('_')[:40]}")
    res.codes = sorted(set(res.codes))
    return res
