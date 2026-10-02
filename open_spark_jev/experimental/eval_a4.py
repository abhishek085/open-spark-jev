"""Standalone eval for A4 (evidential Dirichlet head, train/sft_a4.py).

MenuScorer.decide() -- used by eval/osdg.py, eval/external.py, everything else -- never reads
aux_head; it always scores the restricted LM-head. A checkpoint trained by sft_a4.py is therefore
invisible to the standard eval scripts, the same way option_readout.py and depth_probe.py each
need their own scorer because they read something decide() doesn't. This script computes the
evidential mean p = alpha/S from the trained aux_head, applies the fitted per-type temperature
from calibration.json (via log(p)/T, renormalised -- see a4_probs_logits / train/sft_a4.py's
evaluate_a4), and reports the same accuracy/ECE/Brier/NLL metrics as the v7.x progsplits
SUMMARY.md files so the two are comparable side by side.

  python -m open_spark_jev.experimental.eval_a4 --model checkpoints/v7.7-4b \
      --data-prefix data/synthetic/prog_v1/ --out runs/v7/eval_a4_v7.7.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..calibration import summary
from ..data.corpus import read_jsonl
from ..model import MenuScorer
from ..prompting import render_prefix, render_suffix
from .variants import a4_probs_logits


@torch.no_grad()
def score_split(scorer: MenuScorer, path: str, limit: int = 0) -> dict:
    recs = read_jsonl(path)[: limit or None]
    by_type: dict[str, dict] = {}
    t0 = time.time()
    for i, r in enumerate(recs):
        q, state = r.question_obj(), r.state_obj()
        ids = scorer._encode(render_prefix(state, max_chars=scorer.max_state_chars)) + scorer._encode(
            render_suffix(q)
        )
        t = torch.tensor([ids], device=scorer.device)
        out = scorer.lm(input_ids=t, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1]
        k = len(q.labels)
        z = scorer.aux_head(h.to(scorer.aux_head.weight.dtype))[:, :k].float()
        lab_mask = torch.ones(1, k, dtype=torch.bool, device=scorer.device)
        zl = a4_probs_logits(z, lab_mask)[0]  # log(evidential mean), length-k
        T = scorer.calibration.t(q.type)
        p = F.softmax(zl / T, dim=-1).cpu().numpy()
        d = by_type.setdefault(q.type, {"probs": [], "labels": []})
        d["probs"].append(p.tolist())
        d["labels"].append(q.labels.index(r.target["label"]))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    report = {}
    for qt, d in by_type.items():
        K = max(len(p) for p in d["probs"])
        probs = [p + [0.0] * (K - len(p)) for p in d["probs"]]
        report[qt] = summary(probs, d["labels"])
    n = sum(len(d["labels"]) for d in by_type.values())
    overall_hit = [int(np.argmax(p)) == y for d in by_type.values() for p, y in zip(d["probs"], d["labels"])]
    report["overall"] = {"n": n, "accuracy": round(float(np.mean(overall_hit)), 4)}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-prefix", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scorer = MenuScorer(a.model, use_state_cache=False, aux_head=True)
    state_dict = torch.load(os.path.join(a.model, "aux_head.pt"), map_location=scorer.device)
    scorer.aux_head.load_state_dict(state_dict)
    scorer.aux_head.eval()

    results = {}
    for split in ("calibration", "test_locked", "challenge"):
        path = f"{a.data_prefix}{split}.jsonl"
        if not os.path.exists(path):
            continue
        print(f"=== {split} ===", flush=True)
        results[split] = score_split(scorer, path, a.limit)

    report = {
        "model": a.model,
        "data_prefix": a.data_prefix,
        "splits": results,
        "note": "evidential-mean readout via aux_head, calibrated with the checkpoint's own "
        "calibration.json (fit on log(evidential mean) -- see train/sft_a4.py). "
        "Compare against runs/osdg/*-progsplits/SUMMARY.md for the restricted-LM-head baselines.",
    }
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
