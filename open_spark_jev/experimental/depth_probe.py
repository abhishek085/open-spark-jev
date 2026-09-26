"""A7/A8 (docs/NOVELTY.md): how much depth does a *decision* actually need, and is depth a free ensemble?

One forward pass already computes every layer's hidden state. Applying the (tied) output norm + lm_head to
each layer's last-position hidden state gives a per-layer answer distribution for free -- the "logit lens".
Two independent questions fall out of the same probe:

  A7  depth truncation. If the option-letter answer saturates at layer L < 32, the layers above L are
      latency we pay for nothing. Cutting them is a pure speed win on the axis where decider-4b v2 beats
      us (13 ms vs 75 ms), and unlike quantisation it compounds with NVFP4.
  A8  depth as an ensemble / uncertainty signal. Disagreement between layers is a confidence signal that
      costs *zero* extra compute, unlike permutation ensembling (N forward passes). If late layers disagree,
      the model is unsure -- exactly the signal our shipped single temperature lacks on hard items.

Both matter because v6's calibration is already near-perfect on our own splits (ECE 0.007, see
scripts/analysis/calib_v7_study.py) yet badly overconfident on JevBench hard (ECE 0.265): the missing
ingredient is a *difficulty-aware* confidence, and layer disagreement is the cheapest candidate.

Reads benchmark items only to evaluate; nothing here is used to fit anything that ships (and JevBench
items are never training data -- see scripts/tools/benchmark_overlap.py).

  python -m open_spark_jev.experimental.depth_probe --model checkpoints/v6-4b \
      --items data/benchmarks/v5_test_locked.jsonl --limit 400 --out runs/osdg/depth_probe_v6_own.json
  python -m open_spark_jev.experimental.depth_probe --model checkpoints/v6-4b \
      --jevbench ../external/jevbench/datasets/public/hard.jsonl --out runs/osdg/depth_probe_v6_hard.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..data.corpus import read_jsonl
from ..model import MenuScorer
from ..prompting import render_prefix, render_suffix
from ..schema import Choice, Noul, State


def ece(conf, hit, n=10):
    conf, hit = np.asarray(conf, float), np.asarray(hit, float)
    b = np.minimum((conf * n).astype(int), n - 1)
    return float(sum((b == i).mean() * abs(hit[b == i].mean() - conf[b == i].mean()) for i in range(n) if (b == i).any()))


def jevbench_items(path):
    """(state, question, labels, gold, gold_probs) for each public JevBench item. Evaluation only."""
    out = []
    for line in open(path):
        it = json.loads(line)
        q = it["question"]
        labels = [str(x) for x in it["labels"]]   # score-tier items label levels as ints
        crit = q.get("criteria") or {}
        if q.get("type") == "noul":
            prompt = q.get("instructions") or ""
            if crit:
                prompt += "\nTrue means: " + str(crit.get("true", "")) + "\nFalse means: " + str(crit.get("false", ""))
            question = Noul(prompt=prompt)
            labels = question.labels                      # our Noul renders as yes/no
            gold = "yes" if it["expected"] in ("yes", "true", True) else "no"
            gp = it["provenance"].get("gold_probs") or {}
            gold_probs = {"yes": gp.get("yes", gp.get("true", 0.0)), "no": gp.get("no", gp.get("false", 0.0))} if gp else None
        else:
            # choice items define criteria per option id; score items give an ordered list of level meanings
            defs = crit.items() if isinstance(crit, dict) else zip(labels, crit)
            prompt = (q.get("instructions") or "") + ("\nOption definitions:\n" + "\n".join(f"- {k}: {v}" for k, v in defs) if crit else "")
            question = Choice(prompt=prompt, options=labels)
            gold = str(it["expected"])
            gold_probs = it["provenance"].get("gold_probs")
        out.append({"id": it["id"], "family": it.get("family", "na"), "state": State(content=it["state"]),
                    "question": question, "labels": question.labels, "gold": gold, "gold_probs": gold_probs})
    return out


def own_items(path, limit):
    recs = read_jsonl(path)
    if limit:
        recs = recs[:limit]
    out = []
    for r in recs:
        q = r.question_obj()
        out.append({"id": r.id, "family": r.meta.get("scenario_family", "na"), "state": r.state_obj(), "question": q,
                    "labels": q.labels, "gold": r.target["label"], "gold_probs": None})
    return out


@torch.no_grad()
def per_layer_label_logits(scorer: MenuScorer, item) -> np.ndarray:
    """[n_layers+1, n_labels] option-letter logits read off every layer's last-position hidden state."""
    ids = scorer._encode(render_prefix(item["state"], max_chars=scorer.max_state_chars)) + scorer._encode(render_suffix(item["question"]))
    t = torch.tensor([ids], device=scorer.device)
    out = scorer.lm(input_ids=t, output_hidden_states=True)
    norm, head = scorer.lm.model.norm, scorer.lm.lm_head
    lab = torch.tensor(scorer.labels.ids_for(len(item["labels"])), device=scorer.device)
    rows = []
    for h in out.hidden_states:                      # embeddings + one per layer
        z = head(norm(h[:, -1]))[0]                  # the pretrained readout, applied early
        rows.append(z[lab].float().cpu().numpy())
    return np.stack(rows), len(ids)


def softmax(z):
    z = np.asarray(z, float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items")
    ap.add_argument("--jevbench")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = jevbench_items(a.jevbench) if a.jevbench else own_items(a.items, a.limit)
    if a.jevbench and a.limit:
        items = items[: a.limit]
    scorer = MenuScorer(a.model, use_state_cache=False)
    n_layers = scorer.lm.config.num_hidden_layers
    layer_kind = scorer.lm.config.layer_types

    per_item = []
    t0 = time.time()
    for i, it in enumerate(items):
        Z, ntok = per_layer_label_logits(scorer, it)
        gi = it["labels"].index(it["gold"])
        per_item.append({"id": it["id"], "family": it["family"], "n_tokens": ntok, "gold_idx": gi,
                         "n_labels": len(it["labels"]), "logits_by_layer": [[round(float(v), 4) for v in row] for row in Z],
                         "gold_probs": ([it["gold_probs"].get(l, 0.0) for l in it["labels"]] if it["gold_probs"] else None)})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    # ---- A7: accuracy and calibration as a function of depth -------------------------------
    by_layer = []
    for L in range(n_layers + 1):
        probs = [softmax(np.array(r["logits_by_layer"][L])) for r in per_item]
        hit = [float(int(np.argmax(p)) == r["gold_idx"]) for p, r in zip(probs, per_item)]
        conf = [float(p.max()) for p in probs]
        by_layer.append({"layer": L, "kind": ("embed" if L == 0 else layer_kind[L - 1]),
                         "accuracy": round(float(np.mean(hit)), 4), "mean_conf": round(float(np.mean(conf)), 4),
                         "ece": round(ece(conf, hit), 4),
                         "agree_with_final": round(float(np.mean([int(np.argmax(softmax(np.array(r["logits_by_layer"][L]))) ==
                                                                  np.argmax(softmax(np.array(r["logits_by_layer"][n_layers])))) for r in per_item])), 4)})
    final_acc = by_layer[n_layers]["accuracy"]
    saturate = next((d["layer"] for d in by_layer if d["accuracy"] >= final_acc - 0.005), n_layers)

    # ---- A8: last-K-layer ensemble, and layer disagreement as an uncertainty signal ---------
    ens = []
    for K in (1, 2, 3, 4, 6, 8):
        probs = [np.mean([softmax(np.array(r["logits_by_layer"][L])) for L in range(n_layers + 1 - K, n_layers + 1)], axis=0) for r in per_item]
        hit = [float(int(np.argmax(p)) == r["gold_idx"]) for p, r in zip(probs, per_item)]
        conf = [float(p.max()) for p in probs]
        ens.append({"last_k_layers": K, "accuracy": round(float(np.mean(hit)), 4),
                    "mean_conf": round(float(np.mean(conf)), 4), "ece": round(ece(conf, hit), 4)})

    # does the final answer's correctness depend on how much the late layers agreed?
    tail = range(n_layers - 7, n_layers + 1)
    rows = []
    for r in per_item:
        ds = [softmax(np.array(r["logits_by_layer"][L])) for L in tail]
        top = [int(np.argmax(d)) for d in ds]
        rows.append({"flips": len(set(top)) - 1, "correct": int(top[-1] == r["gold_idx"]),
                     "conf": float(ds[-1].max()), "mean_tvd": float(np.mean([0.5 * np.abs(ds[i] - ds[-1]).sum() for i in range(len(ds) - 1)]))})
    buckets = {}
    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 99)):
        sel = [x for x in rows if lo <= x["flips"] <= hi]
        if sel:
            buckets[f"flips_{lo}-{hi}" if lo != hi else f"flips_{lo}"] = {
                "n": len(sel), "accuracy": round(float(np.mean([x["correct"] for x in sel])), 4),
                "mean_conf": round(float(np.mean([x["conf"] for x in sel])), 4),
                "mean_tvd_to_final": round(float(np.mean([x["mean_tvd"] for x in sel])), 4)}

    report = {"model": a.model, "source": a.jevbench or a.items, "n_items": len(per_item), "n_layers": n_layers,
              "final_accuracy": final_acc, "depth_saturates_at_layer": saturate,
              "layers_prunable": n_layers - saturate, "by_layer": by_layer,
              "last_k_layer_ensemble": ens, "late_layer_disagreement": buckets,
              "seconds": round(time.time() - t0, 1)}
    json.dump({"summary": report, "per_item": per_item}, open(a.out, "w"))
    print(json.dumps(report, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
