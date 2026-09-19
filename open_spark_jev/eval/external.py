"""Score a checkpoint on each third-party Jev evaluation source SEPARATELY.

Sources live in data/benchmarks/external/<source>.jsonl (built by scripts/external/build_external.py,
provenance in data/external/<source>/PROVENANCE.md). Nothing is pooled across sources: every source
gets its own metrics block and its own row in the summary table, so results can be quoted as
"on <source> we got ...".

Per source: accuracy, macro-F1, Brier, NLL, ECE (own calibration.json, no re-fitting on the source),
plus per-slice breakdown. If the source recorded Jev's own outputs (meta.jev) we also report Jev's
accuracy/ECE/Brier on the same rows and our agreement with Jev. ext-jev-directory additionally
reports the eval-level pass rate (an eval passes only if all its questions match).

  python -m open_spark_jev.eval.external --model checkpoints/rlcd-direct-qwen3-1.7b --name rlcd-direct
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from ..calibration import brier_noul, ece_noul, summary
from ..data.corpus import read_jsonl
from .benchmark import _pad, run


def _jev_probs(rec, labels):
    j = rec.meta.get("jev")
    if not j:
        return None
    if "p_yes" in j:
        return [j["p_yes"], 1 - j["p_yes"]]
    d = j["probabilities"]
    v = [float(d.get(lab, 0.0)) for lab in labels]
    s = sum(v)
    return [x / s for x in v] if s > 0 else None


def score_source(records, answers) -> dict:
    labels = [r.label_index() for r in records]
    probs = [a.probs for a in answers]
    out = {"n": len(records), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary(_pad(probs), labels).items()}}
    noul = [(a.probability, 1 if r.target["label"] == "yes" else 0) for r, a in zip(records, answers) if r.question["type"] == "noul"]
    if noul:
        p, t = zip(*noul)
        out["noul"] = {"n": len(noul), "brier": round(brier_noul(list(p), list(t)), 4), "ece": round(ece_noul(list(p), list(t)), 4)}
    slices = defaultdict(list)
    for i, r in enumerate(records):
        slices[r.domain].append(i)
    out["slices"] = {}
    for dom, idx in sorted(slices.items()):
        s = summary(_pad([probs[i] for i in idx]), [labels[i] for i in idx])
        out["slices"][dom] = {"n": len(idx), "accuracy": round(s["accuracy"], 4), "ece": round(s["ece"], 4)}
    # eval-level pass rate (directory)
    if all("eval_id" in r.meta for r in records):
        ev = defaultdict(list)
        for r, a in zip(records, answers):
            ev[r.meta["eval_id"]].append(a.selected == r.question_obj().labels[r.label_index()])
        out["eval_pass_rate"] = {"evals": len(ev), "passed": sum(all(v) for v in ev.values()), "rate": round(sum(all(v) for v in ev.values()) / len(ev), 4)}
    # comparison with recorded Jev output
    rows = [(i, _jev_probs(r, r.question_obj().labels)) for i, r in enumerate(records) if r.meta.get("jev")]
    rows = [(i, jp) for i, jp in rows if jp is not None]
    if rows:
        idx = [i for i, _ in rows]
        jp = [p for _, p in rows]
        jl = [labels[i] for i in idx]
        js = summary(_pad(jp), jl)
        mine = summary(_pad([probs[i] for i in idx]), jl)
        agree = float(np.mean([int(np.argmax(jp[k]) == np.argmax(probs[i])) for k, i in enumerate(idx)]))
        out["vs_recorded_jev"] = {"n": len(idx), "jev_accuracy": round(js["accuracy"], 4), "jev_ece": round(js["ece"], 4), "jev_brier": round(js["brier"], 4),
                                  "ours_accuracy_same_rows": round(mine["accuracy"], 4), "ours_ece_same_rows": round(mine["ece"], 4), "ours_brier_same_rows": round(mine["brier"], 4),
                                  "argmax_agreement_with_jev": round(agree, 4)}
    return out


def table(name: str, res: dict) -> str:
    lines = [f"## {name}", "", "Sources are reported separately and never pooled. Provenance for each: `data/external/<source>/PROVENANCE.md`.", "",
             "| source | n | acc | ECE | Brier | Jev acc (recorded) | ours acc (same rows) | agreement w/ Jev |", "|---|---|---|---|---|---|---|---|"]
    for sid, r in res.items():
        j = r.get("vs_recorded_jev")
        lines.append(f"| {sid} | {r['n']} | {r['accuracy']:.3f} | {r['ece']:.3f} | {r['brier']:.3f} | "
                     + (f"{j['jev_accuracy']:.3f} | {j['ours_accuracy_same_rows']:.3f} | {j['argmax_agreement_with_jev']:.3f} |" if j else "- | - | - |"))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", default="data/benchmarks/external")
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from ..model import MenuScorer

    scorer = MenuScorer(a.model, use_state_cache=False)
    files = sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
    res = {}
    for f in files:
        sid = os.path.basename(f)[:-6]
        if a.sources and sid not in a.sources:
            continue
        recs = read_jsonl(f)[: a.limit or None]
        answers, secs = run(scorer, recs, batch=8)
        res[sid] = score_source(recs, answers)
        res[sid]["seconds"] = round(secs, 1)
        print(f"{sid}: n={res[sid]['n']} acc={res[sid]['accuracy']:.3f} ece={res[sid]['ece']:.3f}", flush=True)
    out_dir = os.path.join("runs", "external", a.name)
    os.makedirs(out_dir, exist_ok=True)
    for sid, r in res.items():
        with open(os.path.join(out_dir, f"{sid}.json"), "w") as fh:
            json.dump({"source": sid, "model": a.model, **r}, fh, indent=2)
    with open(os.path.join(out_dir, "SUMMARY.md"), "w") as fh:
        fh.write(table(a.name, res))


if __name__ == "__main__":
    main()
