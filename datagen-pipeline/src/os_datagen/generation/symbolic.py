"""Small symbolic solvers: literal forward chaining with source authority and default-exceptions.

Literal = (predicate, entity, positive). A rule fires per entity when all body literals are known.
Only quantifiers "all"/"no" fire; "some"/"most"/etc. license no conclusion about an individual. `unless` predicates are exceptions evaluated against the current closure."""
from __future__ import annotations

from typing import Any

Key = tuple[str, str, bool]


def _saturate(known: dict[Key, int], rules: list[dict[str, Any]]) -> None:
    changed = True
    while changed:
        changed = False
        for r in rules:
            if r.get("quant") not in ("all", "no", None):
                continue
            for e in {e for (_, e, _) in list(known)}:
                if not all((p, e, pos) in known for p, pos in r["body"]):
                    continue
                if any((u, e, True) in known for u in r.get("unless", [])):
                    continue
                auth = min(known[(p, e, pos)] for p, pos in r["body"])
                head: Key = (r["head"][0], e, bool(r["head"][1]))
                if known.get(head, -1) < auth:
                    known[head] = auth
                    changed = True


def closure(facts: list[dict[str, Any]], rules: list[dict[str, Any]]) -> dict[Key, int]:
    """Stratified: rules without exceptions are saturated first, then default rules (with `unless`) are
    applied against that closure. This makes the result independent of rule order."""
    known: dict[Key, int] = {}
    for f in facts:
        k = (f["pred"], f["ent"], bool(f.get("pos", True)))
        known[k] = max(known.get(k, 0), int(f.get("auth", 1)))
    _saturate(known, [r for r in rules if not r.get("unless")])
    _saturate(known, rules)
    return known


def literal_status(known: dict[Key, int], pred: str, ent: str, pos: bool) -> str:
    """'true' | 'false' | 'conflict' | 'unknown' for literal (pred, ent, pos)."""
    a = known.get((pred, ent, pos))
    b = known.get((pred, ent, not pos))
    if a is not None and b is not None:
        return "true" if a > b else "false" if b > a else "conflict"
    if a is not None:
        return "true"
    if b is not None:
        return "false"
    return "unknown"


def entailment_verdict(facts: list[dict[str, Any]], rules: list[dict[str, Any]], claim: dict[str, Any]) -> str:
    """entailed | contradicted | unknown. Source conflicts of equal authority resolve to unknown."""
    st = literal_status(closure(facts, rules), claim["pred"], claim["ent"], bool(claim.get("pos", True)))
    return {"true": "entailed", "false": "contradicted"}.get(st, "unknown")
