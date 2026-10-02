#!/usr/bin/env python
"""CPU-only audit of training-data diversity and split leakage (docs/NOVELTY.md, data-first direction).

For each training file, per family: rows, unique decisions (state+question text, ignoring option order), and "skeletons" --
the state+prompt text with digits masked and whitespace collapsed -- as a cheap proxy for how many genuinely different
templates the model sees. Also exact-content overlap between train and each eval split (leakage), since a fixed train set
that memorises to loss ~1e-4 makes any leaked duplicate look like generalisation.

  python scripts/analysis/data_audit.py --out runs/v7/data_audit.json
"""

import argparse, collections, hashlib, json, re


def text_of(r):
    st = r.get("state", {})
    s = st.get("content") if isinstance(st, dict) else st
    q = r.get("question", {})
    return (s if isinstance(s, str) else json.dumps(s, sort_keys=True)), (
        q.get("prompt") or q.get("instructions") or ""
    )


def unique_key(r):
    s, p = text_of(r)
    q = r.get("question", {})
    opts = sorted(
        map(str, q.get("options", []) or q.get("labels", []) or [])
    )  # option ORDER does not make a new decision
    return hashlib.md5(json.dumps([s, p, opts]).encode()).hexdigest()


def skeleton(r):
    s, p = text_of(r)
    t = re.sub(r"\d+([.,]\d+)*", "#", s + " || " + p)
    return hashlib.md5(re.sub(r"\s+", " ", t).encode()).hexdigest()


def family(r):
    m = r.get("meta", {})
    return m.get("scenario_family") or m.get("family") or r.get("domain", "?")


def grams(r, n=5):
    s, p = text_of(r)
    w = re.sub(r"\d+([.,]\d+)*", "#", (s + " " + p).lower()).split()
    return {" ".join(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}


def boilerplate(rows, cap=400, thresh_frac=0.05):
    """Mean share of a row's 5-word phrases that recur in >=5% of its family's rows (template boilerplate)."""
    import random

    by = collections.defaultdict(list)
    for r in rows:
        by[family(r)].append(r)
    per, allv = {}, []
    for f, rs in by.items():
        rs = random.Random(0).sample(rs, min(cap, len(rs)))
        G = [grams(r) for r in rs]
        df = collections.Counter(g for gs in G for g in gs)
        th = max(3, int(thresh_frac * len(rs)))
        vals = [sum(df[g] >= th for g in gs) / max(len(gs), 1) for gs in G]
        per[f] = round(sum(vals) / len(vals), 3)
        allv += vals
    return per, round(sum(allv) / len(allv), 3)


def coverage(train_rows, eval_rows, cap=600):
    """Share of the eval rows' 5-word phrases (per family) already present anywhere in the train rows of that family."""
    import random

    tb = collections.defaultdict(set)
    for r in train_rows:
        tb[family(r)] |= grams(r)
    by = collections.defaultdict(list)
    for r in eval_rows:
        by[family(r)].append(r)
    out = {}
    for f, rs in by.items():
        rs = random.Random(0).sample(rs, min(cap, len(rs)))
        vals = [len(grams(r) & tb.get(f, set())) / max(len(grams(r)), 1) for r in rs]
        out[f] = round(sum(vals) / len(vals), 3)
    return out


def load(path):
    return [json.loads(l) for l in open(path)]


def summarize(name, rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[family(r)].append(r)
    fams = {}
    for f, rs in by.items():
        fams[f] = {
            "rows": len(rs),
            "unique_decisions": len({unique_key(r) for r in rs}),
            "skeletons": len({skeleton(r) for r in rs}),
        }
    tot = {
        "rows": len(rows),
        "unique_decisions": len({unique_key(r) for r in rows}),
        "skeletons": len({skeleton(r) for r in rows}),
        "families": len(by),
    }
    return tot, fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = "data/synthetic/"
    sets = {"v5_train": D + "v5_train.jsonl", "prog_v1_train": D + "prog_v1/train.jsonl"}
    evals = {
        "prog_v1_calibration": D + "prog_v1/calibration.jsonl",
        "prog_v1_test_locked": D + "prog_v1/test_locked.jsonl",
        "prog_v1_challenge": D + "prog_v1/challenge.jsonl",
        "v5_calibration": "data/benchmarks/v5_calibration.jsonl",
        "v5_test_locked": "data/benchmarks/v5_test_locked.jsonl",
        "v5_challenge": "data/benchmarks/v5_challenge.jsonl",
    }
    report = {"train": {}, "leakage_exact_unique_decision_overlap": {}}
    train_keys = {}
    for n, p in sets.items():
        rows = load(p)
        tot, fams = summarize(n, rows)
        report["train"][n] = {"total": tot, "per_family": fams}
        train_keys[n] = {unique_key(r) for r in rows}
        print(
            f"{n}: rows={tot['rows']} unique_decisions={tot['unique_decisions']} skeletons={tot['skeletons']} families={tot['families']}"
        )
    for n, p in sets.items():
        per, mean = boilerplate(load(p))
        report["train"][n]["boilerplate_5gram_share"] = {
            "mean": mean,
            "per_family_sample": dict(list(per.items())[:8]),
        }
        print(
            f"{n}: mean boilerplate 5-gram share = {mean}  (per family, first 8: {dict(list(per.items())[:8])})"
        )
    report["train_phrase_coverage_of_eval"] = {}
    for tn, tp in (("prog_v1_train", sets["prog_v1_train"]), ("v5_train", sets["v5_train"])):
        trows = load(tp)
        for en, ep in evals.items():
            if (tn == "prog_v1_train") != en.startswith("prog_v1"):
                continue
            cov = coverage(trows, load(ep))
            report["train_phrase_coverage_of_eval"][f"{tn} -> {en}"] = cov
            print(
                f"coverage {tn} -> {en}: mean {round(sum(cov.values()) / len(cov), 3)}  {dict(list(cov.items())[:5])}"
            )
    for en, ep in evals.items():
        ek = {unique_key(r) for r in load(ep)}
        for tn, tk in train_keys.items():
            ov = len(ek & tk)
            report["leakage_exact_unique_decision_overlap"][f"{tn} vs {en}"] = {
                "eval_unique": len(ek),
                "overlap": ov,
            }
            if ov:
                print(f"  LEAK? {tn} vs {en}: {ov}/{len(ek)} eval decisions appear verbatim in train")
    json.dump(report, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
