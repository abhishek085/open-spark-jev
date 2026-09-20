"""spark-s1 version-ladder evaluation on the os-datagen splits (calibration / test_locked / challenge).

One command scores a model (frozen base or trained checkpoint) and reports, per split:
  - accuracy / NLL / Brier / ECE at T=1 (raw) and after a per-question-type temperature fitted on the
    CALIBRATION split only (v1), applied unchanged to test_locked and challenge;
  - option-order robustness: choice questions are re-scored under P random option permutations, we report
    accuracy per permutation, the share of rows whose selected option changes, and slot bias;
  - selective prediction (v1): a confidence threshold picked on calibration, coverage and selective accuracy on test.
Per-row predictions are always written (runs/osdg/<name>/rows.jsonl).

  python -m open_spark_jev.eval.osdg --model models/Qwen3-1.7B --name v0-qwen3-1.7b
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

from ..calibration import brier, ece, fit_temperature, nll, softmax
from ..data.corpus import read_jsonl

SPLITS = {"calibration": "data/benchmarks/osdg_calibration.jsonl", "test_locked": "data/benchmarks/osdg_test_locked.jsonl",
          "challenge": "data/benchmarks/osdg_challenge.jsonl"}


def permute(rec, seed):
    if rec.question["type"] != "choice" or seed == 0:
        return rec
    r = copy.deepcopy(rec)
    opts = list(r.question["options"])
    random.Random(f"{seed}-{rec.id}").shuffle(opts)
    r.question["options"] = opts
    return r


def metrics(probs, y):
    if not len(y):
        return {}
    K = max(len(p) for p in probs)
    P = [list(p) + [0.0] * (K - len(p)) for p in probs]
    return {"n": len(y), "acc": round(float(np.mean([int(np.argmax(p) == t) for p, t in zip(P, y)])), 4), "nll": round(nll(P, y), 4),
            "brier": round(brier(P, y), 4), "ece": round(ece(P, y), 4)}


def score_all(scorer, recs, perms):
    rows = []
    for p in range(perms):
        for r in recs:
            rp = permute(r, p)
            q = rp.question_obj()
            a = scorer.decide(rp.state_obj(), [q], temperature=1.0, return_logits=True)[0]
            z = a.raw_logits if a.raw_logits is not None else [float(np.log(max(x, 1e-12))) for x in a.probs]
            rows.append({"id": r.id, "perm": p, "pack": r.domain, "qtype": q.type, "labels": q.labels, "gold": r.target["label"],
                         "gold_idx": q.labels.index(r.target["label"]), "acceptable": r.meta.get("acceptable"), "logits": [round(float(v), 4) for v in z],
                         "difficulty": r.meta.get("difficulty"), "family": r.meta.get("scenario_family")})
    return rows


def evaluate(scorer, model_label, name, perms=3, limit=0, target_sel=0.95):
    data = {s: read_jsonl(f)[: limit or None] for s, f in SPLITS.items()}
    rows = {s: score_all(scorer, recs, perms) for s, recs in data.items()}
    out_dir = os.path.join("runs", "osdg", name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as fh:
        for s, rs in rows.items():
            for r in rs:
                fh.write(json.dumps({"split": s, **r}) + "\n")

    T = {}
    for qt in ("choice", "score", "noul"):
        cal = [r for r in rows["calibration"] if r["qtype"] == qt]
        T[qt] = fit_temperature([r["logits"] + [-1e4] * (max(len(x["logits"]) for x in cal) - len(r["logits"])) for r in cal], [r["gold_idx"] for r in cal]) if len(cal) >= 10 else 1.0
    summ = {"model": model_label, "temperature_fit_on_calibration": {k: round(v, 3) for k, v in T.items()}, "splits": {}}

    def probs_of(r, calibrated):
        return softmax(np.array(r["logits"]) / (T[r["qtype"]] if calibrated else 1.0)).tolist()

    # abstention threshold on calibration (v1): smallest confidence threshold reaching the target selective accuracy
    cal0 = [r for r in rows["calibration"] if r["perm"] == 0]
    confs = sorted({max(probs_of(r, True)) for r in cal0})
    thr = None
    for c in confs:
        kept = [r for r in cal0 if max(probs_of(r, True)) >= c]
        if len(kept) >= 20 and np.mean([int(np.argmax(probs_of(r, True)) == r["gold_idx"]) for r in kept]) >= target_sel:
            thr = c
            break
    summ["abstain_threshold_from_calibration"] = thr
    for s, rs in rows.items():
        blk = {}
        for cal in (False, True):
            per_perm = [metrics([probs_of(r, cal) for r in rs if r["perm"] == p], [r["gold_idx"] for r in rs if r["perm"] == p]) for p in range(perms)]
            blk["calibrated" if cal else "raw"] = {"original_order": per_perm[0], "mean_over_perms_acc": round(float(np.mean([m["acc"] for m in per_perm])), 4)}
        ch = defaultdict(dict)
        for r in rs:
            if r["qtype"] == "choice":
                ch[r["id"]][r["perm"]] = r["labels"][int(np.argmax(r["logits"]))]
        blk["order_sensitivity"] = {"choice_rows": len(ch), "selected_option_changes_under_permutation": round(float(np.mean([len(set(v.values())) > 1 for v in ch.values()])), 4) if ch else None}
        slot = Counter(int(np.argmax(r["logits"])) for r in rs if r["qtype"] == "choice")
        n = sum(slot.values()) or 1
        blk["slot_share"] = {int(k): round(v / n, 3) for k, v in sorted(slot.items())}
        acc_ok = [int(r["labels"][int(np.argmax(r["logits"]))] in (r["acceptable"] or [r["gold"]])) for r in rs if r["perm"] == 0]
        blk["acceptable_option_acc"] = round(float(np.mean(acc_ok)), 4)
        r0 = [r for r in rs if r["perm"] == 0]
        if thr is not None:
            kept = [r for r in r0 if max(probs_of(r, True)) >= thr]
            blk["selective"] = {"threshold": round(float(thr), 4), "coverage": round(len(kept) / len(r0), 4),
                                "selective_acc": round(float(np.mean([int(np.argmax(probs_of(r, True)) == r["gold_idx"]) for r in kept])), 4) if kept else None}
        packs = defaultdict(list)
        for r in r0:
            packs[r["pack"]].append(int(np.argmax(r["logits"]) == r["gold_idx"]))
        blk["by_pack_acc"] = {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(packs.items())}
        summ["splits"][s] = blk
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    lines = [f"## {name}", "", f"T (fit on calibration): {summ['temperature_fit_on_calibration']}; abstain threshold: {thr}", "",
             "| split | acc | acc (mean of perms) | option flips | NLL raw->cal | ECE raw->cal | Brier raw->cal | selective (cov / acc) |", "|---|---|---|---|---|---|---|---|"]
    for s, b in summ["splits"].items():
        r, c = b["raw"]["original_order"], b["calibrated"]["original_order"]
        sel = b.get("selective") or {}
        lines.append(f"| {s} | {r['acc']:.3f} | {b['raw']['mean_over_perms_acc']:.3f} | {b['order_sensitivity']['selected_option_changes_under_permutation']} | {r['nll']:.2f}->{c['nll']:.2f} | "
                     f"{r['ece']:.3f}->{c['ece']:.3f} | {r['brier']:.3f}->{c['brier']:.3f} | {sel.get('coverage', '-')} / {sel.get('selective_acc', '-')} |")
    open(os.path.join(out_dir, "SUMMARY.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))



def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target-selective-acc", type=float, default=0.95)
    a = ap.parse_args()

    from ..experimental.variant_scorer import VariantScorer, is_variant
    from ..model import MenuScorer

    scorer = VariantScorer(a.model) if is_variant(a.model) else MenuScorer(a.model, use_state_cache=False)
    evaluate(scorer, a.model, a.name, a.perms, a.limit, a.target_selective_acc)


if __name__ == "__main__":
    main()
