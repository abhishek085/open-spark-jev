"""A11 (docs/NOVELTY.md): adaptive-depth cascade. Composes A7's truncation (depth saturates at
layer 24 of 32 for v7.6-4b) with a confidence gate, so non-autoregressive doesn't have to mean
fixed compute: route easy items to an early exit at layer 24, and only run the remaining 8 layers
for items the gate isn't confident about.

Needs no new training -- depth_probe.py already computes every layer's restricted-softmax
distribution in one forward pass (the logit lens), so this is pure post-hoc analysis on its JSON
output: read the layer-24 distribution's confidence (max prob), fit an exit threshold on the
calibration split (never on the split being evaluated), then measure on held-out splits:

  * accuracy of the cascade (layer-24 answer when confident, layer-32 answer otherwise) vs.
    always running to layer 32
  * fraction of items that exit early, as a proxy for the latency saved (24/32 = 0.75x the full
    forward pass for those items)

  python -m open_spark_jev.experimental.depth_probe --model checkpoints/v7.6-4b \
      --items data/synthetic/prog_v1/calibration.jsonl --limit 200 --out runs/v7/depth_probe_v7.6_calibration.json
  python -m open_spark_jev.experimental.depth_cascade --exit-layer 24 \
      --fit runs/v7/depth_probe_v7.6_calibration.json \
      --eval runs/v7/depth_probe_v7.6_test_locked.json runs/v7/depth_probe_v7.6_challenge.json \
      --out runs/v7/depth_cascade_v7.6.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def softmax(z):
    z = np.asarray(z, float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def per_item_layers(path: str, exit_layer: int, final_layer: int):
    per_item = json.load(open(path))["per_item"]
    out = []
    for r in per_item:
        p_exit = softmax(np.array(r["logits_by_layer"][exit_layer]))
        p_final = softmax(np.array(r["logits_by_layer"][final_layer]))
        out.append(
            {
                "conf_exit": float(p_exit.max()),
                "correct_exit": int(np.argmax(p_exit)) == r["gold_idx"],
                "correct_final": int(np.argmax(p_final)) == r["gold_idx"],
            }
        )
    return out


def fit_threshold(items) -> float:
    """Smallest confidence threshold such that the *blended* cascade accuracy (early-exit items
    scored at layer 24, the rest at layer 32) matches or beats always running to layer 32 -- not
    the exit subset's raw accuracy, which is a different and looser target. Fit on the
    calibration split only.

    NB: an earlier version of this function used --target-exit-accuracy against the exit
    subset's raw accuracy. That is the wrong criterion even before the data problem below: raw
    exit-subset accuracy over-punishes real thresholds because it ignores what the *cascade*
    (not the early-exit branch alone) actually delivers. It was also run on a corrupted
    sample -- depth_probe.py was invoked with --limit on data/synthetic/prog_v1/*.jsonl, whose
    rows are grouped by scenario_family, so the first N rows were entirely one family. Always
    fit (and evaluate) on the full split, or an explicitly shuffled/stratified sample -- never
    a naive head-of-file --limit on these files."""
    full_depth_acc = round(float(np.mean([it["correct_final"] for it in items])), 4)
    best = 1.0
    for thr in np.linspace(
        1.0, 0.5, 101
    ):  # search from strict to loose; first hit is smallest exit-inducing threshold
        r = evaluate_cascade(items, float(thr))
        if (
            r["cascade_accuracy"] >= full_depth_acc
        ):  # both rounded the same way, so the zero-exit case at thr=1.0 always ties
            best = float(thr)
        else:
            break
    return best


def evaluate_cascade(items, threshold: float) -> dict:
    n = len(items)
    exit_mask = [it["conf_exit"] >= threshold for it in items]
    n_exit = sum(exit_mask)
    cascade_correct = [it["correct_exit"] if ex else it["correct_final"] for it, ex in zip(items, exit_mask)]
    full_correct = [it["correct_final"] for it in items]
    exit_only_acc = (
        float(np.mean([it["correct_exit"] for it, ex in zip(items, exit_mask) if ex])) if n_exit else None
    )
    return {
        "n": n,
        "exit_fraction": round(n_exit / n, 4),
        "relative_compute": round((n_exit * 0.75 + (n - n_exit) * 1.0) / n, 4),
        "cascade_accuracy": round(float(np.mean(cascade_correct)), 4),
        "full_depth_accuracy": round(float(np.mean(full_correct)), 4),
        "exit_subset_accuracy": round(exit_only_acc, 4) if exit_only_acc is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exit-layer", type=int, default=24)
    ap.add_argument("--final-layer", type=int, default=32)
    ap.add_argument(
        "--fit", required=True, help="depth_probe.py JSON to fit the exit threshold on (calibration split)"
    )

    ap.add_argument(
        "--eval", nargs="+", required=True, help="depth_probe.py JSON(s) to evaluate the cascade on"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fit_items = per_item_layers(a.fit, a.exit_layer, a.final_layer)
    threshold = fit_threshold(fit_items)

    report = {
        "exit_layer": a.exit_layer,
        "final_layer": a.final_layer,
        "fit_on": a.fit,
        "fit_criterion": "smallest threshold where blended cascade accuracy >= full-depth accuracy, on --fit split",
        "threshold": threshold,
        "evals": {},
    }
    for path in a.eval:
        items = per_item_layers(path, a.exit_layer, a.final_layer)
        report["evals"][path] = evaluate_cascade(items, threshold)

    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
