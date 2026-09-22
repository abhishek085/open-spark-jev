"""Assemble Jev-style training / calibration / test data (v5): harness and decision packs only.

Sources: the v1 os-datagen release and any os-datagen LLM runs (general-purpose packs are dropped), the code-only tool-call-risk run, and the code-only
Jev-use-case run. Code-only packs are capped per pack so no single pack dominates. Rows within Jaccard 0.7 of any 60-set command are dropped.

  python scripts/build_v5_data.py --llm datagen-pipeline/artifacts/mixture_v2_r1 datagen-pipeline/artifacts/harness_r1 \
      --toolrisk datagen-pipeline/artifacts/toolrisk_r1 --usecases datagen-pipeline/artifacts/jevuse_r1
Writes data/synthetic/v5_train.jsonl (+ option-permuted copies), v5_rl_pool.jsonl and data/benchmarks/v5_{calibration,test_locked,challenge}.jsonl.
"""
import argparse
import copy
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "datagen-pipeline/src")
from os_datagen.training.export import export_training  # noqa: E402

GENERAL = ("foundation_semantic_entailment", "foundation_document_type", "foundation_rule_application", "foundation_temporal_reasoning", "foundation_extraction_validation")
ap = argparse.ArgumentParser()
ap.add_argument("--llm", nargs="*", default=[])
ap.add_argument("--toolrisk", nargs="*", default=[])
ap.add_argument("--usecases", nargs="*", default=[])
ap.add_argument("--perms", type=int, default=2)
ap.add_argument("--rl-holdout", type=float, default=0.15)
ap.add_argument("--toolrisk-train-cap", type=int, default=1500)
ap.add_argument("--usecase-pack-cap", type=int, default=450, help="max train rows per code-only use-case pack")
ap.add_argument("--eval-pack-cap", type=int, default=120, help="max calibration/test/challenge rows per code-only use-case pack")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prefix", default="v5")
a = ap.parse_args()
rng = random.Random(a.seed)
SCR = Path("/tmp/claude-1000/-home-admin/aece0fe9-f2ff-4298-bafa-f902cf88f34b/scratchpad/v5build")
SCR.mkdir(parents=True, exist_ok=True)
FILE = {"train": "accepted_train.jsonl", "calibration": "accepted_calibration.jsonl", "locked_test": "accepted_test_locked.jsonl", "challenge": "accepted_challenge.jsonl"}
SPLITS = list(FILE)


def load(path):
    return [json.loads(x) for x in open(path) if x.strip()]


def export(run, split):
    out = SCR / f"{Path(run).name}_{split}.jsonl"
    export_training(Path(run) / FILE[split], out, fmt="jev", allow_splits={split})
    return load(out)


def toks(t):
    return set(re.sub(r"[^a-z0-9]+", " ", str(t).lower()).split())


test60 = [toks(json.loads(x)["state"]["content"]) for x in open("data/benchmarks/external/ext-toolcall-risk.jsonl")]


def leaks(rec):
    c = toks(rec["state"]["content"])
    return any(len(c & t) / max(1, len(c | t)) >= 0.7 for t in test60)


def jev_only(rows):
    return [r for r in rows if not str(r["domain"]).startswith(GENERAL)]


def cap_per_pack(rows, cap):
    by = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(r)
    out = []
    for v in by.values():
        rng.shuffle(v)
        out += v[:cap]
    return out


old = {"train": load("data/synthetic/osdg_train.jsonl"), "calibration": load("data/benchmarks/osdg_calibration.jsonl"),
       "locked_test": load("data/benchmarks/osdg_test_locked.jsonl"), "challenge": load("data/benchmarks/osdg_challenge.jsonl")}
old = {k: jev_only(v) for k, v in old.items()}
new = {k: [] for k in SPLITS}
for run in a.llm:
    for k in SPLITS:
        new[k] += jev_only(export(run, k))
tcr = {k: [] for k in SPLITS}
dropped = 0
for run in a.toolrisk:
    for k in SPLITS:
        rows = export(run, k)
        keep = [r for r in rows if not leaks(r)]
        dropped += len(rows) - len(keep)
        tcr[k] += keep
rng.shuffle(tcr["train"])
tcr["train"] = tcr["train"][: a.toolrisk_train_cap]
uc = {k: [] for k in SPLITS}
for run in a.usecases:
    for k in SPLITS:
        uc[k] += export(run, k)
uc = {k: cap_per_pack(v, a.usecase_pack_cap if k == "train" else a.eval_pack_cap) for k, v in uc.items()}

rng.shuffle(new["train"])
n_rl = int(len(new["train"]) * a.rl_holdout)
rl_pool, new_train = new["train"][:n_rl], new["train"][n_rl:]


def permute(rows, k):
    out = []
    for r in rows:
        out.append(r)
        if r["question"]["type"] != "choice":
            continue
        for i in range(1, k + 1):
            c = copy.deepcopy(r)
            opts = list(c["question"]["options"])
            random.Random(f"aug-{i}-{r['id']}").shuffle(opts)
            if opts != r["question"]["options"]:
                c["question"]["options"] = opts
                c["id"] = f"{r['id']}-p{i}"
                out.append(c)
    return out


train = old["train"] + new_train + tcr["train"] + uc["train"]
rng.shuffle(train)
train_aug = permute(train, a.perms)
rng.shuffle(train_aug)


def write(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


P = a.prefix
write(f"data/synthetic/{P}_train.jsonl", train_aug)
write(f"data/synthetic/{P}_rl_pool.jsonl", rl_pool)
for k, name in (("calibration", "calibration"), ("locked_test", "test_locked"), ("challenge", "challenge")):
    write(f"data/benchmarks/{P}_{name}.jsonl", old[k] + new[k] + tcr[k] + uc[k])
print("train rows", len(train), "-> with permutations", len(train_aug), "| rl pool", len(rl_pool), "| 60-set near-duplicates dropped:", dropped)
print("train by pack:", dict(Counter(r["domain"].replace("harness_", "").replace("foundation_", "") for r in train).most_common(60)))
for k in ("calibration", "locked_test", "challenge"):
    print(k, len(old[k]) + len(new[k]) + len(tcr[k]) + len(uc[k]), "(old", len(old[k]), "llm", len(new[k]), "toolrisk", len(tcr[k]), "usecases", len(uc[k]), ")")
