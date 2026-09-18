"""No-RL control: post-hoc temperature scaling only.

Fits a per-question-type temperature on a validation split of the SFT checkpoint's own logits
(no additional training) and scores it on the test set, exactly like ``train/sft.py``'s
built-in temperature fit but as a standalone, auditable step. This is the cheapest possible
way to chase "calibrated probabilities" and the floor both RLCD mechanisms
(``rlcd_contrastive.yaml``, ``rlcd_direct.yaml``) need to beat to justify the extra training
cost - if temperature scaling alone gets you most of the way to TypeSafe's stated goal, that
is itself an important, reportable finding (see docs/RESEARCH.md R1).

Usage:
  python -m open_spark_jev.eval.calibration_baseline --model checkpoints/sft-qwen3-1.7b \
    --val-data data/synthetic/sim_train.jsonl --test-data data/benchmarks/sim_test.jsonl \
    --out runs/eval_calibration_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import numpy as np

from ..calibration import fit_temperature, softmax, summary
from ..data.corpus import Record, read_jsonl, split_records
from ..model import Calibration, MenuScorer
from ..train.sft import Example, build_examples

log = logging.getLogger("osj.calibration_baseline")


def collect_logits(scorer: MenuScorer, examples: list[Example]) -> dict[str, tuple[list, list]]:
    """question type -> (list of raw label logits, list of true indices), grouped by K to keep
    arrays rectangular (mirrors train/sft.py's evaluate())."""
    import torch

    by_type: dict[str, dict[int, tuple[list, list]]] = defaultdict(dict)
    pad = scorer.tokenizer.pad_token_id or 0
    from ..train.sft import collate

    bs = 16
    for i in range(0, len(examples), bs):
        b = examples[i : i + bs]
        ids, mask, last, lab, lab_mask, tgt, hard = collate(b, pad, scorer.device)
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=scorer.device.startswith("cuda")),
        ):
            z = scorer.train_forward(ids, mask, last, lab, lab_mask)
        for j, e in enumerate(b):
            k = len(e.label_ids)
            zl, yl = by_type[e.qtype].setdefault(k, ([], []))
            zl.append(z[j, :k].float().cpu().numpy().tolist())
            yl.append(e.target_idx)
    return {qt: groups for qt, groups in by_type.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-data", nargs="+", required=True, help="fit temperature here")
    ap.add_argument("--test-data", nargs="+", required=True, help="score here")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out", default="runs/eval_calibration_baseline.json")
    a = ap.parse_args()

    scorer = MenuScorer(a.model, use_state_cache=False)
    val_recs: list[Record] = []
    for p in a.val_data:
        val_recs.extend(read_jsonl(p))
    _, val_recs = split_records(val_recs, a.val_frac, 0)  # split_records returns (rest, val_frac-slice)
    val_ex = build_examples(scorer, val_recs, a.max_len)
    log.info("fitting temperature on %d validation examples", len(val_ex))
    val_logits = collect_logits(scorer, val_ex)

    temps: dict[str, float] = {}
    for qt, groups in val_logits.items():
        ts = [fit_temperature(np.array(zl), np.array(yl)) for zl, yl in groups.values() if len(yl) >= 20]
        temps[qt] = float(np.mean(ts)) if ts else 1.0
    log.info("fitted temperatures: %s", temps)
    scorer.calibration = Calibration(temperature=temps)

    test_recs: list[Record] = []
    for p in a.test_data:
        test_recs.extend(read_jsonl(p))
    test_ex = build_examples(scorer, test_recs, a.max_len)
    test_logits = collect_logits(scorer, test_ex)

    report: dict = {"temperatures": temps, "by_type": {}}
    for qt, groups in test_logits.items():
        probs, ys = [], []
        for zl, yl in groups.values():
            z = np.array(zl) / temps.get(qt, 1.0)
            p = softmax(z)
            probs.extend(p.tolist())
            ys.extend(yl)
        K = max(len(p) for p in probs)
        probs = [p + [0.0] * (K - len(p)) for p in probs]
        report["by_type"][qt] = summary(probs, ys)
    print(json.dumps(report, indent=2))
    import os

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
