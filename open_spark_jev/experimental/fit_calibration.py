"""Refit calibration.json for a checkpoint that wasn't produced by train/sft.py (e.g. a model-soup
or truncation edit), using the same val-split logic and fit_temperature() sft.py uses at the end of
a training run.

model_soup.py and truncate.py both write a checkpoint without calibration -- averaging or deleting
weights invalidates any source checkpoint's fitted temperature, so it must be refit, not inherited.

  python -m open_spark_jev.experimental.fit_calibration --model checkpoints/v7.6-4b \
      --config configs/train/sft_v7_2.yaml
"""

from __future__ import annotations

import argparse
import json

import yaml

from ..data.corpus import read_jsonl, split_records
from ..model import MenuScorer
from ..train.sft import build_examples, evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--config",
        required=True,
        help="training config whose data/seed/val_frac defined this checkpoint's lineage",
    )
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    records = [r for path in cfg["data"] for r in read_jsonl(path)]
    _, val_recs = split_records(records, cfg.get("val_frac", 0.1), cfg.get("seed", 0))

    scorer = MenuScorer(a.model, use_state_cache=False, aux_head=cfg.get("aux_head", False))
    val_ex = build_examples(scorer, val_recs, cfg.get("max_len", 2048))
    pad_id = scorer.tokenizer.pad_token_id if scorer.tokenizer.pad_token_id is not None else 0
    rep = evaluate(scorer, val_ex, cfg.get("batch_size", 8), pad_id)

    scorer.calibration.temperature = {qt: r["temperature"] for qt, r in rep.items()}
    scorer.calibration.save(f"{a.model}/calibration.json")
    print(
        json.dumps(
            {"model": a.model, "n_val": len(val_ex), "temperature": scorer.calibration.temperature}, indent=1
        )
    )


if __name__ == "__main__":
    main()
