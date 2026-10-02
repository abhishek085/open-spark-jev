"""RLDM-6: evolution-strategy search over soup mixture weights (A13's model soup, but searched instead of hand-picked).

A13 / v7.8 showed linear interpolation between checkpoints can beat both parents, and that *which* parents and *how much*
matters (v7.8's 50/50 with v6 bought JevBench-hard at a real own-domain cost). Instead of guessing 50/50, treat the
mixture weights w = softmax(theta) over N sibling checkpoints as a policy searched by a (mu/mu, lambda)-ES, reward =
calibrated-Brier on held-out *calibration* splits (fitness). No gradient, no per-example fitting, no reward saturation:
each candidate is just "does this blend do better", so the failure modes that killed live RL here do not apply.

Fitness data are calibration splits only (prog_v1 + v5, stratified/random sample) -- never test_locked/challenge, never
JevBench, never external sets; those stay pure evaluation for the final champion. Fitness = min over a temperature grid of
mean Brier, so it rewards accuracy and calibration jointly (a proper score) without depending on a stored calibration.json.

  python -m open_spark_jev.experimental.es_soup --out checkpoints/v7.10-4b --log runs/v7/es_soup_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
from safetensors import safe_open

from ..calibration import ece as ece_fn
from ..data.corpus import read_jsonl
from ..model import MenuScorer
from ..train.sft import build_examples, collate

CKPTS = [
    "checkpoints/v6-4b",
    "checkpoints/v7.2-4b",
    "checkpoints/v7.3-4b",
    "checkpoints/v7.4-4b",
    "checkpoints/v7.5-4b",
]


def fitness_examples(scorer, n_each, seed):
    rng = random.Random(seed)
    prog = read_jsonl("data/synthetic/prog_v1/calibration.jsonl")
    fams = sorted({r.meta.get("scenario_family", "?") for r in prog})
    per = max(1, n_each // len(fams))
    pick = []
    for f in fams:
        fr = [r for r in prog if r.meta.get("scenario_family", "?") == f]
        pick += rng.sample(fr, min(per, len(fr)))
    v5 = read_jsonl("data/benchmarks/v5_calibration.jsonl")
    pick += rng.sample(v5, min(n_each, len(v5)))
    ex = build_examples(scorer, pick, 2048)
    ex.sort(key=lambda e: len(e.input_ids))
    return ex


def token_batches(examples, budget):
    """examples are sorted by length ascending, so each new example is the longest so far in its batch."""
    batch = []
    for e in examples:
        if batch and (len(batch) + 1) * len(e.input_ids) > budget:
            yield batch
            batch = []
        batch.append(e)
    if batch:
        yield batch


@torch.no_grad()
def eval_fitness(scorer, examples, budget, pad_id):
    scorer.lm.eval()
    Z, Y = [], []
    for b in token_batches(examples, budget):
        ids, mask, last, lab, lab_mask, tgt, hard = collate(b, pad_id, scorer.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = scorer.train_forward(ids, mask, last, lab, lab_mask)
        for j, e in enumerate(b):
            Z.append(z[j, : len(e.label_ids)].float().cpu().numpy())
            Y.append(e.target_idx)
    best = None
    for T in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        P = []
        for z in Z:
            q = np.exp((z - z.max()) / T)
            P.append(q / q.sum())
        brier = float(np.mean([((p - np.eye(len(p))[y]) ** 2).sum() for p, y in zip(P, Y)]))
        if best is None or brier < best[0]:
            hit = [float(np.argmax(p) == y) for p, y in zip(P, Y)]
            K = max(len(p) for p in P)
            best = (
                brier,
                T,
                float(np.mean(hit)),
                float(ece_fn([list(p) + [0.0] * (K - len(p)) for p in P], Y)),
            )
    return {"brier": best[0], "T": best[1], "acc": best[2], "ece": best[3]}


class Soup:
    def __init__(self, scorer, paths):
        self.scorer = scorer
        # Lazy, memory-mapped reads: nothing but the one live model stays resident. On this box the GPU shares the
        # 121 GB system pool, and holding all 5 checkpoints (~42 GB) resident plus big-vocab logits is what got the
        # first two full runs OOM-killed (2026-09-29 17:12 and 20:57). Page cache is reclaimable; resident tensors are not.
        self.paths = [p + "/model.safetensors" for p in paths]
        with safe_open(self.paths[0], "pt", device="cpu") as f0:
            self.sds = [{k: None for k in f0.keys()}]  # key set only; handles are NOT kept open (see set())
        self.target = scorer.lm.state_dict()
        # saved checkpoints name params 'model.language_model.*'; the loaded module names them 'model.*'
        self.map = {k: k.replace("model.language_model.", "model.", 1) for k in self.sds[0]}
        missing = [k for k, v in self.map.items() if v not in self.target]
        assert not missing, (
            f"{len(missing)}/{len(self.map)} saved keys do not map onto the model, e.g. {missing[:3]}"
        )
        assert scorer.lm.config.get_text_config().tie_word_embeddings or "lm_head.weight" in self.sds[0], (
            "lm_head untied and not saved: would stay unblended"
        )
        self.keys = list(self.map)

    @torch.no_grad()
    def set(self, w):
        # Handles are opened per blend and closed at the end: a long-lived safe_open keeps every touched page mapped, and
        # mapped pages stay counted as in-use (free memory fell 118 -> 48 GB over a smoke run and never recovered).
        import gc

        files = [
            safe_open(p, "pt", device="cpu") if float(wi) != 0.0 else None for p, wi in zip(self.paths, w)
        ]
        try:
            for k in self.keys:
                acc = None
                for wi, f in zip(w, files):
                    if f is None:
                        continue  # zero weight: one-hot baselines read a single checkpoint
                    t = f.get_tensor(k).to(self.target[self.map[k]].device).float() * float(wi)
                    acc = t if acc is None else acc + t
                tk = self.map[k]
                self.target[tk].copy_(acc.to(self.target[tk].dtype))
        finally:
            del files
            gc.collect()


def softmax(t):
    t = np.asarray(t, float) - np.max(t)
    e = np.exp(t)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--decay", type=float, default=0.85)
    ap.add_argument("--n-each", type=int, default=200)
    ap.add_argument(
        "--tok-budget",
        type=int,
        default=8192,
        help="max tokens per forward batch (bounds the tokens x vocab logits tensor)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-save", action="store_true", help="skip writing the champion checkpoint (smoke tests)"
    )
    a = ap.parse_args()

    try:  # unprivileged processes may raise their own score: under memory pressure the kernel kills THIS job first
        open("/proc/self/oom_score_adj", "w").write("1000")
    except OSError:
        pass
    avail = int([ln for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")][0].split()[1]) / 1048576
    print(f"MemAvailable {avail:.0f} GB", flush=True)
    if avail < 30:
        import subprocess

        print(
            subprocess.run(
                "ps -eo rss,pid,cmd --sort=-rss | head -6 | cut -c1-140",
                shell=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        raise SystemExit(
            f"refusing to start: only {avail:.0f} GB available (need ~30: model 8.4 + activations + margin). Another large tenant (e.g. the Hermes backend) is resident."
        )
    scorer = MenuScorer(CKPTS[2], use_state_cache=False)
    pad_id = scorer.tokenizer.pad_token_id or 0
    ex = fitness_examples(scorer, a.n_each, a.seed)
    print("fitness examples:", len(ex), flush=True)
    soup = Soup(scorer, CKPTS)
    import os

    cache = {}
    if os.path.exists(
        a.log
    ):  # resume: the search is seeded, so replaying the log reproduces the same candidates
        for line in open(a.log):
            r0 = json.loads(line)
            cache[tuple(r0["w"])] = {k: r0[k] for k in ("brier", "T", "acc", "ece")}
        print(f"resuming: {len(cache)} evaluations already logged", flush=True)
    log = open(a.log, "a")
    N = len(CKPTS)
    n_eval = [len(cache)]

    def score(w, tag):
        key = tuple(round(float(x), 4) for x in w)
        if key in cache:
            return cache[key]
        t0 = time.time()
        soup.set(w)
        r = eval_fitness(scorer, ex, a.tok_budget, pad_id)
        n_eval[0] += 1
        rec = {
            "n": n_eval[0],
            "tag": tag,
            "sec": round(time.time() - t0, 1),
            "w": [round(float(x), 4) for x in w],
            **{k: round(v, 4) for k, v in r.items()},
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
        cache[key] = r
        return r

    # named baselines inside the simplex: each parent, uniform, v7.6 (=.5 v7.2 + .5 v7.3), v7.8 (=.5 v6 + .25 v7.2 + .25 v7.3)
    names = ["v6", "v7.2", "v7.3", "v7.4", "v7.5"]
    for i, n in enumerate(names):
        score(np.eye(N)[i], f"base:{n}")
    score(np.ones(N) / N, "base:uniform")
    score(np.array([0, 0.5, 0.5, 0, 0]), "base:v7.6-equiv")
    score(np.array([0.5, 0.25, 0.25, 0, 0]), "base:v7.8-equiv")

    rng = np.random.default_rng(a.seed)
    theta, sigma = np.zeros(N), a.sigma
    best = (1e9, None)
    for g in range(a.gens):
        cands = [theta + sigma * rng.standard_normal(N) for _ in range(a.pop)]
        res = [(score(softmax(c), f"gen{g}")["brier"], c) for c in cands]
        res.sort(key=lambda x: x[0])
        theta = np.mean([c for _, c in res[: a.elite]], axis=0)
        if res[0][0] < best[0]:
            best = (res[0][0], res[0][1])
        sigma *= a.decay
        print(f"gen {g}: best-of-gen brier {res[0][0]:.4f}  overall best {best[0]:.4f}", flush=True)

    w = softmax(best[1])
    print("champion weights", dict(zip(names, np.round(w, 3))), flush=True)
    if a.no_save:
        print("--no-save: not writing a checkpoint", flush=True)
        return
    soup.set(w)
    scorer.lm.save_pretrained(a.out, safe_serialization=True)
    scorer.tokenizer.save_pretrained(a.out)
    import os
    import shutil

    for f in ("chat_template.jinja",):
        if os.path.exists(CKPTS[2] + "/" + f):
            shutil.copy(CKPTS[2] + "/" + f, a.out)
    json.dump(
        {
            "method": "ES over soup weights",
            "sources": CKPTS,
            "weights": dict(zip(names, [float(x) for x in w])),
            "fitness_brier": best[0],
            "note": "calibration.json NOT written: refit with fit_calibration.py",
        },
        open(a.out + "/es_soup.json", "w"),
        indent=1,
    )
    print("wrote", a.out)


if __name__ == "__main__":
    main()
