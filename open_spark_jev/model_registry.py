"""Model registry: one saved artifact per experiment, with what changed, how it compares to
Jev, and the fastest way to actually run it. Requested standing practice for this project:
every trained variant gets a MODEL_CARD.md next to its checkpoint plus an entry in
docs/model_lineage.yaml, and docs/MODELS.md is regenerated from that file as the master index.

Every card restates the product contract up front, deliberately repeated per-card rather than
only linked, so a card is understandable on its own if someone only ever opens one:

    Input:  an application state (text/JSON/logs) + one or more typed questions
            (Choice: pick one of N options; Score: place on an ordered rubric;
             Noul: a yes/no claim).
    Output: a structured answer per question -- selected option / rubric level / yes-no --
            each carrying a full probability distribution and a confidence score, never
            free text.

Usage:
    python -m open_spark_jev.model_registry add --id sft-qwen3-1.7b \
        --checkpoint checkpoints/sft-qwen3-1.7b --eval runs/eval_sft.json \
        --mechanism "Phase 1 SFT: soft-target cross-entropy + Brier regularizer, two epochs" \
        --jev-comparison "Supervised bootstrap; TypeSafe hasn't disclosed whether Jev has an \
            SFT stage at all -- this is our default assumption, not a matched claim." \
        --parent null
    python -m open_spark_jev.model_registry render   # regenerate docs/MODELS.md from the lineage file
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from typing import Any

import yaml

LINEAGE_PATH = "docs/model_lineage.yaml"
CONTRACT = (
    "**Input:** an application state (text/JSON/logs) plus one or more typed questions -- "
    "Choice (pick one of N options), Score (place on an ordered rubric), or Noul (a yes/no "
    "claim). **Output:** a structured answer per question -- selected option / rubric level / "
    "yes-no -- each carrying a full probability distribution and a confidence score. Never "
    "free text, never a malformed answer; the answer space is closed by construction."
)


def load_lineage(path: str = LINEAGE_PATH) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_lineage(lineage: dict[str, dict[str, Any]], path: str = LINEAGE_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(lineage, f, sort_keys=False, width=100)


def _fmt_eval_table(eval_path: str | None) -> str:
    if not eval_path or not os.path.exists(eval_path):
        return "_no eval file recorded yet_"
    with open(eval_path) as f:
        d = json.load(f)
    rows = ["| slice | n | accuracy | ECE | Brier | soft Brier vs posterior |",
            "|---|---|---|---|---|---|"]
    for k, v in d.items():
        if not isinstance(v, dict) or "accuracy" not in v:
            continue
        rows.append(
            f"| {k} | {v.get('n', '-')} | {v.get('accuracy', float('nan')):.3f} | "
            f"{v.get('ece', float('nan')):.3f} | {v.get('brier', float('nan')):.3f} | "
            f"{v.get('soft_brier_vs_posterior', float('nan')):.3f} |"
        )
    extra = []
    for k in ("injection_flip_rate", "throughput_decisions_per_s"):
        if k in d:
            extra.append(f"- `{k}`: {d[k]:.4f}" if isinstance(d[k], float) else f"- `{k}`: {d[k]}")
    return "\n".join(rows) + ("\n\n" + "\n".join(extra) if extra else "")


def _run_command(checkpoint: str) -> str:
    return (
        f"cd /home/admin/llm-workspace/Open-Spark-Jev && "
        f".venv/bin/osj decide --model {checkpoint} \\\n"
        f'  --state \'{{"ticket": "Charged twice this month, invoice missing."}}\' --json \\\n'
        f'  -q \'{{"type": "choice", "prompt": "Which queue?", "options": ["billing", "technical", "sales"], "allow_abstain": true}}\' \\\n'
        f'  -q \'{{"type": "noul", "prompt": "The customer reports a duplicate charge."}}\''
    )


def render_card(entry_id: str, entry: dict[str, Any]) -> str:
    parent_str = f"`{entry['parent']}`" if entry.get("parent") else "(base backbone, no parent checkpoint)"
    lines = [
        f"# Model card: {entry_id}",
        "",
        CONTRACT,
        "",
        f"**Date:** {entry.get('date', '?')}  ",
        f"**Checkpoint:** `{entry['checkpoint']}`  ",
        f"**Parent:** {parent_str}",
        "",
        "## What this experiment is",
        entry.get("mechanism", "_not recorded_"),
        "",
        "## How this compares to Jev",
        entry.get("jev_comparison", "_not recorded_"),
        "",
        "## Fastest way to run it",
        "```bash",
        _run_command(entry["checkpoint"]),
        "```",
        "Swap `--state`/`-q` for your own; add `--backend openai --base-url ...` in "
        "`eval/benchmark.py` instead of the CLI to hit a served endpoint (trtllm-serve/vLLM) "
        "rather than load the checkpoint in-process.",
        "",
        "## Results",
        _fmt_eval_table(entry.get("eval")),
        "",
    ]
    if entry.get("notes"):
        lines += ["## Notes", entry["notes"], ""]
    return "\n".join(lines)


def add_entry(args: argparse.Namespace) -> None:
    lineage = load_lineage()
    entry = {
        "date": args.date or str(date.today()),
        "checkpoint": args.checkpoint,
        "parent": None if args.parent in (None, "null", "") else args.parent,
        "eval": args.eval,
        "mechanism": args.mechanism,
        "jev_comparison": args.jev_comparison,
        "notes": args.notes,
    }
    lineage[args.id] = entry
    save_lineage(lineage)
    card_path = os.path.join(args.checkpoint, "MODEL_CARD.md")
    if os.path.isdir(args.checkpoint):
        with open(card_path, "w") as f:
            f.write(render_card(args.id, entry))
        print(f"wrote {card_path}")
    else:
        print(f"warning: {args.checkpoint} does not exist yet, skipped MODEL_CARD.md (lineage entry saved)")
    render_index()


def render_index(path: str = "docs/MODELS.md") -> None:
    lineage = load_lineage()
    lines = [
        "# Model registry",
        "",
        "Every trained variant of open-spark-Jev, in one place: what experiment it represents,",
        "how it compares to Jev, and the fastest way to run it. Generated from",
        f"`{LINEAGE_PATH}` by `python -m open_spark_jev.model_registry render` -- edit that file",
        "and regenerate, don't hand-edit this one.",
        "",
        CONTRACT,
        "",
        "| id | date | mechanism | parent | overall acc | overall ECE | card |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry_id, entry in lineage.items():
        acc = ece = "-"
        if entry.get("eval") and os.path.exists(entry["eval"]):
            with open(entry["eval"]) as f:
                d = json.load(f)
            if "overall" in d:
                acc = f"{d['overall'].get('accuracy', float('nan')):.3f}"
                ece = f"{d['overall'].get('ece', float('nan')):.3f}"
        mech_short = (entry.get("mechanism") or "")[:60]
        card_link = f"[{entry_id}]({entry['checkpoint']}/MODEL_CARD.md)"
        lines.append(f"| {entry_id} | {entry.get('date', '?')} | {mech_short} | {entry.get('parent') or '-'} | {acc} | {ece} | {card_link} |")
    lines += ["", "Full per-domain numbers and the honest before/after on any bug fixes are in "
                    "[docs/BENCHMARKS.md](BENCHMARKS.md); the architecture side (non-training-mechanism "
                    "experiments) is tracked separately in [docs/NOVELTY.md](NOVELTY.md)."]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="record a new model variant and write its MODEL_CARD.md")
    a.add_argument("--id", required=True)
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--eval", default=None, help="path to an eval/benchmark.py JSON output")
    a.add_argument("--mechanism", required=True)
    a.add_argument("--jev-comparison", dest="jev_comparison", required=True)
    a.add_argument("--parent", default=None)
    a.add_argument("--notes", default=None)
    a.add_argument("--date", default=None)
    a.set_defaults(fn=add_entry)

    r = sub.add_parser("render", help="regenerate docs/MODELS.md from docs/model_lineage.yaml")
    r.set_defaults(fn=lambda _a: render_index())

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
