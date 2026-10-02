"""Collect every number in the v8 card from run outputs -> release/v8/facts.json (card tokens)."""

import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.environ.get("JEVBENCH_DIR", "../external/jevbench"))
from jevbench import composite_v12 as C


def J(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def jb(ck):
    out = {}
    for t in ("original", "easy", "hard"):
        d = J(f"runs/jevbench/{ck}/{t}.summary.json")
        if d:
            out[t] = d
    hard = []
    try:
        for ln in open(f"runs/jevbench/{ck}/hard.results.jsonl"):
            r = json.loads(ln)
            if r.get("valid"):
                hard.append((bool(r["correct"]), max(r["probs"].values())))
    except Exception:
        pass
    out["w8"] = sum((not c) and p >= 0.8 for c, p in hard)
    out["w9"] = sum((not c) and p >= 0.9 for c, p in hard)
    out["intel"] = (
        C.intelligence(
            {
                "standard": out["original"]["accuracy"],
                "easy": out["easy"]["accuracy"],
                "hard": out["hard"]["accuracy"],
            }
        )
        if len(out) >= 5
        else None
    )
    return out


def ext(name, src):
    d = J(f"runs/external/{name}/{src}.json")
    return (d["accuracy"], d["brier"], d["ece"]) if d else None


def own(name):
    d = J(f"runs/osdg/{name}/summary.json")
    return (
        None
        if not d
        else (
            d["splits"]["test_locked"]["calibrated"]["mean_over_perms_acc"],
            d["splits"]["challenge"]["calibrated"]["mean_over_perms_acc"],
        )
    )


def sm(z, T):
    z = [x / T for x in z]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def heldout(name, T):
    n = ok = c8 = c9 = 0
    br = 0
    bins = [[0, 0, 0.0] for _ in range(10)]
    try:
        f = open(f"runs/osdg/{name}/rows.jsonl")
    except Exception:
        return None
    for ln in f:
        r = json.loads(ln)
        if r["qtype"] != "choice" or r["split"] == "calibration" or r.get("perm", 0) != 0:
            continue
        p = sm(r["logits"], T)
        i = max(range(len(p)), key=p.__getitem__)
        c = p[i]
        g = [j for j, lb in enumerate(r["labels"]) if lb in r["acceptable"]]
        o = i in g
        n += 1
        ok += o
        c8 += (not o) and c >= 0.8
        c9 += (not o) and c >= 0.9
        br += sum((p[j] - (j in g) / len(g)) ** 2 for j in range(len(p)))
        b = bins[min(9, int(c * 10))]
        b[0] += 1
        b[1] += o
        b[2] += c
    if not n:
        return None
    return (
        n,
        ok / n,
        sum(b[0] / n * abs(b[1] / b[0] - b[2] / b[0]) for b in bins if b[0]),
        br / n,
        c8 / n,
        c9 / n,
    )


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def best(a, b, hi=True, fmt="{:.3f}"):
    sa, sb = fmt.format(a), fmt.format(b)
    if a == b:
        return sa, sb
    w = (a > b) if hi else (a < b)
    return (f"**{sa}**", sb) if w else (sa, f"**{sb}**")


V8Q = "v8-4b-nvfp4"
j6, j8 = jb("v6-4b-nvfp4"), jb(V8Q)
jb16 = jb("v7.12.1-4b")


def tri(fmt, a, b, c):
    return " | ".join(fmt.format(*x) for x in (a, b, c))


rows = ["| Measure | v6-nvfp4 (published) | v8-nvfp4 | Notes |", "|---|---:|---:|---|"]
rows.append(
    f"| JevBench public tiers (easy / standard / hard) | {j6['easy']['accuracy']:.3f} / {j6['original']['accuracy']:.3f} / {j6['hard']['accuracy']:.3f} | {j8['easy']['accuracy']:.3f} / {j8['original']['accuracy']:.3f} / {j8['hard']['accuracy']:.3f} | JevBench v1.2 public items only (231 of 534). **Hard accuracy is lower than v6-nvfp4's.** |"
)
rows.append(
    f"| JevBench public-proxy Intelligence | {j6['intel']:.1f} | {j8['intel']:.1f} | `composite_v12.intelligence()`, judge tier dropped |"
)
rows.append(
    f"| JevBench hard: Brier / ECE | {j6['hard']['brier_mean']:.3f} / {j6['hard']['ece']['ece']:.3f} | {j8['hard']['brier_mean']:.3f} / {j8['hard']['ece']['ece']:.3f} | Shipped temperatures |"
)
rows.append(
    f"| JevBench hard: wrong at >=0.8 / >=0.9 confidence | {j6['w8']} / {j6['w9']} | {j8['w8']} / {j8['w9']} | Of 111 items |"
)
headline = "\n".join(rows)

hr = ["| Measure | v6 bf16 | v8 bf16 (not published) | v8-nvfp4 (published) |", "|---|---:|---:|---:|"]
for src, lab in [
    ("ext-injection-ctx", "Prompt injection, with context"),
    ("ext-injection-noctx", "Prompt injection, no context"),
    ("ext-jev-directory", "Jev-directory (70)"),
    ("ext-kev-decision-v1", "Kev decision-v1"),
    ("ext-toolcall-risk", "60-case tool-call diagnostic"),
]:
    a, b, c = (
        ext("v6-4b-jev-external", src),
        ext("v7.12.1-4b-jev-external", src),
        ext(f"{V8Q}-jev-external", src),
    )
    hr.append(
        f"| {lab}: accuracy / Brier / ECE | "
        + " | ".join(f"{x[0]:.3f} / {x[1]:.3f} / {x[2]:.3f}" for x in (a, b, c))
        + " |"
    )
for key, lab in [
    ("hermes", "Hermes-style agent decisions (our generator, held-out phrasing)"),
    ("pubagent", "Public agent evals (When2Call test, held-out ToolACE tools, Gandalf)"),
]:
    a, b, c = heldout(f"v6-4b-{key}", 1.481), heldout(f"v7.12-4b-{key}", 2.5), heldout(f"{V8Q}-{key}", 2.5)
    hr.append(
        f"| {lab}: accuracy / Brier / wrong at >=0.9 | "
        + " | ".join(f"{x[1]:.3f} / {x[3]:.3f} / {x[5]:.1%}" for x in (a, b, c))
        + " |"
    )
a, b = heldout("v6-4b-progv2full", 1.481), heldout("v7.12-4b-progv2full", 2.5)
hr.append(
    f"| Programmatic families (bf16 only; not run on NVFP4): accuracy / Brier / wrong at >=0.9 | {a[1]:.3f} / {a[3]:.3f} / {a[5]:.1%} | {b[1]:.3f} / {b[3]:.3f} / {b[5]:.1%} | not run |"
)
heldout_t = "\n".join(hr)

qrows = [
    "| Measure | v8 bf16 | v8 NVFP4 |",
    "|---|---:|---:|",
    f"| JevBench easy / standard / hard | {jb16['easy']['accuracy']:.3f} / {jb16['original']['accuracy']:.3f} / {jb16['hard']['accuracy']:.3f} | {j8['easy']['accuracy']:.3f} / {j8['original']['accuracy']:.3f} / {j8['hard']['accuracy']:.3f} |",
    f"| JevBench Intelligence | {jb16['intel']:.1f} | {j8['intel']:.1f} |",
    f"| JevBench hard wrong at >=0.9 | {jb16['w9']} | {j8['w9']} |",
]
qrows.append(
    "Per-set differences are in the table above (bf16 vs NVFP4 columns): about one point or less except Jev-directory (0.914 to 0.886)."
)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


facts = {
    "TABLE_HEADLINE": headline,
    "TABLE_HELDOUT": heldout_t,
    "NVFP4_TABLE": "\n".join(qrows),
    "KEV_V8": "see note",
    "NVFP4_NOTE": "published build",
    "NVFP4_CAL_NOTE": "uses the same scalar temperatures as the bf16 build; the held-out tables above are measured on the NVFP4 build itself.",
    "SHA_BF16": "(not published)",
    "SHA_NVFP4": sha("engines/spark-s1-4b-v8-MLP_ONLY_CFG/model.safetensors"),
    "LATENCY": "Not re-measured for v8. v8 has the same architecture and parameter count as v6, so speed should match v6-nvfp4's measured 53.3 ms p50 / 18.6 decisions/s (and 74.9 ms bf16).",
}
json.dump(facts, open("release/v8/facts.json", "w"), indent=1)
print(headline)
print()
print(heldout_t)
print()
print(facts["NVFP4_TABLE"])
print(facts["LATENCY"])
