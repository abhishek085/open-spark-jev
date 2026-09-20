"""The public /v1/decide contract:

  request : {"state": {...}, "questions": [{"id","type": choice|boolean|score,"instructions", "options":[{"id","definition"}] | "levels":[{"value","definition"}]}], "model": optional}
  response: {"model", "decisions": {id: {"selected","probabilities","confidence","margin","entropy","latency_ms"}}, "latency_ms"}

Questions are mapped to the same typed prompts the models were trained on (choice ids as options with the definitions appended,
boolean -> noul, score levels as string values with the rubric "value: definition").
"""

from __future__ import annotations

from typing import Any

from ..schema import Answer, Choice, Noul, Score, State


def is_contract(body: dict[str, Any]) -> bool:
    qs = body.get("questions")
    return isinstance(qs, list) and bool(qs) and all(isinstance(q, dict) and "instructions" in q for q in qs) and "content" not in (body.get("state") or {})


def parse(body: dict[str, Any]):
    state = State(content=body["state"])
    ids, qs = [], []
    for q in body["questions"]:
        qid, typ, ins = q["id"], q["type"], q["instructions"]
        ids.append(qid)
        if typ == "boolean":
            qs.append(Noul(id=qid, prompt=ins))
        elif typ == "choice":
            opts = [o if isinstance(o, dict) else {"id": o, "definition": ""} for o in q["options"]]
            defs = "\n".join(f"- {o['id']}: {o['definition']}" for o in opts if o.get("definition"))
            qs.append(Choice(id=qid, prompt=ins + (f"\nOption definitions:\n{defs}" if defs else ""), options=[o["id"] for o in opts]))
        elif typ == "score":
            lv = q["levels"]
            qs.append(Score(id=qid, prompt=ins, levels=[str(x["value"]) for x in lv], rubric="\n".join(f"{x['value']}: {x['definition']}" for x in lv)))
        else:
            raise ValueError(f"{qid}: unknown question type {typ!r}")
    return state, qs, ids


def format_answer(a: Answer, latency_ms: float) -> dict[str, Any]:
    labels = ["true", "false"] if a.type == "noul" else list(a.labels)
    probs = list(a.probs)[: len(labels)]
    order = sorted(probs, reverse=True)
    sel = labels[max(range(len(probs)), key=probs.__getitem__)]
    return {"selected": sel, "probabilities": {k: round(float(p), 6) for k, p in zip(labels, probs)}, "confidence": round(float(order[0]), 6),
            "margin": round(float(order[0] - (order[1] if len(order) > 1 else 0.0)), 6), "entropy": round(float(a.entropy), 6),
            "latency_ms": round(latency_ms, 2)}
