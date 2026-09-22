"""Assemble the v4 training / calibration / test data from: the v1 os-datagen release, one or more new os-datagen mixture runs, and the code-only
tool-call-risk run. Nothing from the 60-row diagnostic set is used: tool-call rows within Jaccard 0.7 of any of its commands are dropped.

  python scripts/build_v4_data.py --mixture datagen-pipeline/artifacts/mixture_v2_r1 --toolrisk datagen-pipeline/artifacts/toolrisk_r1
Writes data/synthetic/v4_train.jsonl (+ option-permuted copies), data/synthetic/v4_rl_pool.jsonl (rows held out of SFT for RLCD) and
data/benchmarks/v4_{calibration,test_locked,challenge}.jsonl.
"""
import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "datagen-pipeline/src")
from os_datagen.training.export import export_training  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mixture", nargs="*", default=[])
ap.add_argument("--toolrisk", nargs="*", default=[])
ap.add_argument("--perms", type=int, default=2, help="extra random option orders per choice row")
ap.add_argument("--rl-holdout", type=float, default=0.15, help="share of NEW mixture train rows kept out of SFT for RLCD")
ap.add_argument("--toolrisk-train-cap", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
SCR = Path("/tmp/claude-1000/-home-admin/aece0fe9-f2ff-4298-bafa-f902cf88f34b/scratchpad/v4build")
SCR.mkdir(parents=True, exist_ok=True)
FILE = {"train": "accepted_train.jsonl", "calibration": "accepted_calibration.jsonl", "locked_test": "accepted_test_locked.jsonl", "challenge": "accepted_challenge.jsonl"}


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


old = {"train": load("data/synthetic/osdg_train.jsonl"), "calibration": load("data/benchmarks/osdg_calibration.jsonl"),
       "locked_test": load("data/benchmarks/osdg_test_locked.jsonl"), "challenge": load("data/benchmarks/osdg_challenge.jsonl")}
new = {k: [] for k in old}
for run in a.mixture:
    for k in new:
        new[k] += export(run, k)
tcr = {k: [] for k in old}
dropped = 0
for run in a.toolrisk:
    for k in tcr:
        rows = export(run, k)
        keep = [r for r in rows if not leaks(r)]
        dropped += len(rows) - len(keep)
        tcr[k] += keep
rng.shuffle(tcr["train"])
tcr["train"] = tcr["train"][: a.toolrisk_train_cap]

# hold out part of the NEW mixture train rows (whole records) for RLCD so the policy sees rows SFT never trained on
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


train = old["train"] + new_train + tcr["train"]
rng.shuffle(train)
train_aug = permute(train, a.perms)
rng.shuffle(train_aug)


def write(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


write("data/synthetic/v4_train.jsonl", train_aug)
write("data/synthetic/v4_rl_pool.jsonl", rl_pool)
for k, name in (("calibration", "calibration"), ("locked_test", "test_locked"), ("challenge", "challenge")):
    write(f"data/benchmarks/v4_{name}.jsonl", old[k] + new[k] + tcr[k])
from collections import Counter  # noqa: E402

print("train rows", len(train), "-> with permutations", len(train_aug), "| rl pool", len(rl_pool), "| tool-call rows dropped as near-duplicates of the 60-set:", dropped)
print("train by pack:", dict(Counter(r["domain"] for r in train).most_common(20)))
for k in ("calibration", "locked_test", "challenge"):
    print(k, len(old[k]) + len(new[k]) + len(tcr[k]), "(old", len(old[k]), "new", len(new[k]), "toolrisk", len(tcr[k]), ")")
