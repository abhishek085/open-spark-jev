"""v7.5: replay data trained toward a frozen checkpoint's own distribution, not hard labels.

Borrowed from decider-4b's v2 -> v2.1 fix (Mapika/decider-4b model card, "Changes from v2"): their
stage-2 LoRA trained 6,676 replay rows (sampled from stage-1's own training mixture) on hard labels,
which sharpened logits everywhere and pushed the fitted temperature to 1.935, flattening every served
answer -- their calibration regression. The fix was to train those replay rows toward the FROZEN prior
checkpoint's own predicted distribution instead, with loss KL(p_prior || p_model): mean KL to the prior
dropped 0.112 -> 0.021 nats and the temperature came back to 1.099.

That is close to our own story: v7.2 added prog_v1 (23% of the training mass, concentrated in a few
fixed templates) on top of v6's data and regressed on JevBench hard (ECE 0.265 -> 0.344) even though
v6's own calibration on its own data was already close to the per-splits optimum
(scripts/analysis/calib_v7_study.py). The hypothesis both here and in v7.4's boilerplate diversification
is negative interference: new data pulling the model away from behaviour that was already good. v7.4
attacks the cause (templated data); this attacks the symptom directly, the same way decider-4b did --
anchor a sample of v6's own training rows to what v6 itself would say, via KL, so the model has an
explicit reason not to drift on the data it already handled well while it learns prog_v1's new skills.

Usage:
  python -m open_spark_jev.experimental.replay_kl --model checkpoints/v6-4b \\
      --source data/synthetic/v5_train.jsonl --n 2500 --out data/synthetic/replay_v6.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="frozen checkpoint whose own distribution becomes the target")
    ap.add_argument("--source", required=True, help="jsonl of Records to sample replay rows from")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from ..data.corpus import read_jsonl
    from ..model import MenuScorer

    recs = read_jsonl(a.source)
    random.Random(a.seed).shuffle(recs)
    recs = recs[: a.n]

    scorer = MenuScorer(a.model, use_state_cache=False)
    written = 0
    with open(a.out, "w") as fh:
        for r in recs:
            q = r.question_obj()
            ans = scorer.decide(r.state_obj(), [q], temperature=1.0, return_logits=False)[0]
            dist = {lab: round(float(p), 6) for lab, p in zip(q.labels, ans.probs)}
            row = {
                "id": f"{r.id}-replay-{a.model.rsplit('/', 1)[-1]}",
                "domain": r.domain,
                "source": f"replay:{a.model}",
                "state": r.state,
                "question": r.question,
                "target": {"label": ans.selected, "dist": dist},
                "meta": {**r.meta, "split": "train", "namespace": "replay",
                         "replay_of": a.model, "soft_target": True,
                         "original_gold": r.target.get("label")},
            }
            fh.write(json.dumps(row) + "\n")
            written += 1
    print(f"wrote {written} replay rows -> {a.out} (target = {a.model}'s own T=1 distribution)")


if __name__ == "__main__":
    main()
