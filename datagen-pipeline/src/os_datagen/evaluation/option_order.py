"""Option-order counterfactuals: render the same visible scenario under several option permutations, run a baseline
model, and flag records whose selected answer changes with the order. Position-sensitive rows are a challenge slice
(and a signal about the baseline), not standard training rows."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..training.render import option_ids, render_prompt, truth_sets
from ..utils.jsonl import read_jsonl, write_jsonl
from ..utils.seeds import make_rng
from .runner import Predictor, make_predictor


def permutations(record: dict[str, Any], k: int) -> list[dict[str, Any]]:
    """k re-ordered copies (first = original order) of a choice record; score/boolean records have a fixed order."""
    if record["decision"]["type"] != "choice":
        return [record]
    out = [record]
    for i in range(1, k):
        r = copy.deepcopy(record)
        opts = r["decision"]["question"]["options"]
        make_rng("optorder", record["record_id"], i).shuffle(opts)
        out.append(r)
    return out


def option_order_check(dataset: Path, out: Path, predictor: Predictor | None = None, endpoint: str = "baseline://uniform",
                       model: str = "baseline", k: int = 4, limit: int | None = None) -> dict[str, Any]:
    pred = predictor or make_predictor(endpoint, model, "constrained_generation")
    rows = []
    for n, rec in enumerate(read_jsonl(dataset)):
        if limit and n >= limit:
            break
        sels = []
        for r in permutations(rec, k):
            _, _, probs, _ = pred.predict(render_prompt(r), option_ids(r), r)
            sels.append(max(probs, key=lambda o: probs[o]))
        pref, acc = truth_sets(rec)
        rows.append({"record_id": rec["record_id"], "task_pack": rec["task_pack"], "selections": sels,
                     "unstable": len(set(sels)) > 1, "correct_in_all_orders": all(s in acc for s in sels),
                     "correct_in_some_orders": any(s in acc for s in sels), "model": model})
    write_jsonl(out / "option_order.jsonl", rows)
    summary = {"n": len(rows), "unstable_share": sum(r["unstable"] for r in rows) / max(1, len(rows)), "k": k, "model": model,
               "position_sensitive_record_ids": [r["record_id"] for r in rows if r["unstable"]][:200]}
    (out / "option_order_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
