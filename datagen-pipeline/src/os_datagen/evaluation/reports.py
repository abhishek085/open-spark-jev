from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.jsonl import write_jsonl


def write_eval(out: Path, rows: list[dict[str, Any]], metrics: dict[str, Any], slice_metrics: dict[str, Any],
               reliability: list[dict[str, Any]], selective: list[dict[str, float]], frontier: list[dict[str, float]] | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # PER-ROW predictions are always saved (not only aggregates): record, scores, probabilities, selection, match, latency
    write_jsonl(out / "predictions.jsonl", ({k: v for k, v in r.items() if k not in ("slice",)} | {"slice": r["slice"]} for r in rows))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    (out / "slices.json").write_text(json.dumps(slice_metrics, indent=2, sort_keys=True))
    (out / "reliability.json").write_text(json.dumps(reliability, indent=2))
    (out / "selective_risk.json").write_text(json.dumps(selective, indent=2))
    if frontier:
        (out / "cost_quality_frontier.json").write_text(json.dumps(frontier, indent=2))
    o = metrics["overall"]
    lines = [f"# Evaluation: {metrics['model']} ({metrics['inference_mode']})\n",
             f"- rows: {o['n']}  top-1: {o['top1_accuracy']:.3f}  acceptable: {o['acceptable_accuracy']:.3f}  macro: {o['macro_accuracy']:.3f}",
             f"- NLL {o['nll']:.3f}  Brier {o['brier']:.3f}  ECE {o['ece']:.3f}", f"- probabilities source: {metrics['probabilities_source']}",
             f"- calibration: {metrics['calibration']}",
             "\n> No calibration claim is made unless `calibration.fitted_on` names an independent calibration split.\n"]
    for name, table in slice_metrics.items():
        lines.append(f"\n## by {name}\n\n| slice | n | top-1 | NLL | ECE |\n|---|---|---|---|---|")
        lines += [f"| {g} | {m['n']} | {m['top1_accuracy']:.2f} | {m['nll']:.2f} | {m['ece']:.2f} |" for g, m in table.items()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
