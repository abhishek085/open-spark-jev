"""A6 (docs/NOVELTY.md): does scoring answer *combinations* jointly beat independent softmaxes?

The existing simulators pose one question per state, so correlated multi-question decisions
cannot be tested with them. This module adds a two-question naive-Bayes simulator whose latent
is the JOINT class (queue, urgency) with a non-factorising prior (abuse tends to be urgent,
sales tends to be low urgency), giving an exact joint posterior per state.

Two arms are trained identically (same states, same SFT recipe, LoRA):
  independent  two records per state (queue Choice, urgency Choice); joint = outer product
  joint        one record per state: a 12-way Choice over "queue & urgency" combinations

and scored on the SAME held-out states against the exact joint posterior: joint soft-Brier,
joint argmax agreement, and marginal soft-Brier (to check marginals are not sacrificed).

  python -m open_spark_jev.experimental.a6_joint gen  --n-train 4000 --n-test 1500
  python -m open_spark_jev.experimental.a6_joint eval --independent CKPT --joint CKPT
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random

import numpy as np

from ..data.corpus import Record, write_jsonl
from ..data.simulators import NaiveBayesSpec

QUEUES = ["billing", "technical", "account", "abuse"]
URGENCY = ["low", "normal", "high"]
COMBOS = [f"{q} & {u}" for q, u in itertools.product(QUEUES, URGENCY)]
OUT = "data/synthetic/a6"

# P(queue, urgency): deliberately non-factorising.
JOINT_PRIOR = {
    "billing": [0.60, 0.35, 0.05], "technical": [0.05, 0.25, 0.70],
    "account": [0.20, 0.70, 0.10], "abuse": [0.02, 0.08, 0.90],
}
QUEUE_PRIOR = [0.30, 0.30, 0.25, 0.15]


def _spec() -> NaiveBayesSpec:
    prior = [QUEUE_PRIOR[i] * JOINT_PRIOR[q][j] for i, q in enumerate(QUEUES) for j in range(3)]
    topic_vals = ["invoice wrong", "charged twice", "app down", "API errors", "cannot log in", "locked out", "harassment", "spam flood"]
    topic_by_q = {"billing": [.20, .20, .12, .12, .10, .10, .08, .08], "technical": [.10, .10, .20, .20, .10, .10, .10, .10],
                  "account": [.08, .08, .10, .10, .20, .20, .12, .12], "abuse": [.06, .06, .10, .10, .12, .12, .22, .22]}
    tone_vals = ["calm", "annoyed", "panicked"]
    tone_by_u = {"low": [.5, .35, .15], "normal": [.4, .4, .2], "high": [.25, .4, .35]}
    tier_vals = ["free", "pro", "enterprise"]
    tier_by_u = {"low": [.45, .35, .2], "normal": [.4, .38, .22], "high": [.3, .35, .35]}
    hist_vals = ["new", "regular", "flagged"]
    hist_by_q = {"billing": [.3, .65, .05], "technical": [.3, .65, .05], "account": [.4, .55, .05], "abuse": [.2, .3, .5]}

    alert_vals = ["none", "quota_alert", "sec_alert"]
    alert_by_c = {c: ([.2, .75, .05] if c in ("technical & high", "billing & low") else [.2, .05, .75] if c in ("abuse & high", "account & high") else [.8, .1, .1]) for c in COMBOS}

    def table(by_q=None, by_u=None):
        return {c: (by_q[c.split(" & ")[0]] if by_q else by_u[c.split(" & ")[1]]) for c in COMBOS}

    feats = {
        "topic": (topic_vals, table(by_q=topic_by_q)),
        "tone": (tone_vals, table(by_u=tone_by_u)),
        "tier": (tier_vals, table(by_u=tier_by_u)),
        "history": (hist_vals, table(by_q=hist_by_q)),
        "alert": (alert_vals, alert_by_c),
    }

    def render(f, rng):
        body = f"Customer ({f['tier']} plan, {f['history']} account) writes in a {f['tone']} tone: \"{f['topic']}\". Monitoring: {f['alert']}. Ticket #{rng.randint(10**4, 10**5)}."
        return {"content": body, "schema_hint": "customer support ticket", "domain": "pair_support"}

    return NaiveBayesSpec(name="pair_support", labels=COMBOS, prior=prior, features=feats, question={}, render=render)


def _marginals(post: list[float]) -> tuple[list[float], list[float]]:
    m = np.array(post).reshape(len(QUEUES), len(URGENCY))
    return m.sum(1).tolist(), m.sum(0).tolist()


def generate(n: int, seed: int) -> dict[str, list[Record]]:
    spec, rng = _spec(), random.Random(f"a6-{seed}")
    joint, ind = [], []
    for i in range(n):
        feats, post, lab = spec.sample(rng, "sample")
        state = spec.render(feats, rng)
        pq, pu = _marginals(post)
        qlab, ulab = lab.split(" & ")
        meta = {"joint_posterior": post}
        base = {"domain": "pair_support", "source": "a6-sim", "state": state}
        joint.append(Record(id=f"a6j-{seed}-{i}", question={"type": "choice", "prompt": "Which queue should handle this ticket, and how urgent is it?", "options": COMBOS, "allow_abstain": False},
                            target={"label": lab, "dist": dict(zip(COMBOS, post))}, meta=meta, **base))
        ind.append(Record(id=f"a6q-{seed}-{i}", question={"type": "choice", "prompt": "Which support queue should handle this ticket?", "options": QUEUES, "allow_abstain": False},
                          target={"label": qlab, "dist": dict(zip(QUEUES, pq))}, meta=meta, **base))
        ind.append(Record(id=f"a6u-{seed}-{i}", question={"type": "choice", "prompt": "How urgent is this ticket?", "options": URGENCY, "allow_abstain": False},
                          target={"label": ulab, "dist": dict(zip(URGENCY, pu))}, meta=meta, **base))
    return {"joint": joint, "independent": ind}


def cmd_gen(a) -> None:
    os.makedirs(OUT, exist_ok=True)
    for split, n, seed in (("train", a.n_train, 0), ("test", a.n_test, 1)):
        for arm, recs in generate(n, seed).items():
            write_jsonl(f"{OUT}/{arm}_{split}.jsonl", recs)
    print("wrote", os.listdir(OUT))


def cmd_eval(a) -> None:
    import torch

    from ..data.corpus import read_jsonl
    from ..model import MenuScorer
    from ..schema import State, parse_question

    def soft_brier(p, t):
        return float(np.sum((np.array(p) - np.array(t)) ** 2))

    def ece(conf_correct, bins=10):
        conf, corr = np.array([c for c, _ in conf_correct]), np.array([x for _, x in conf_correct], float)
        e = 0.0
        for lo in np.linspace(0, 0.9, bins):
            s = (conf >= lo) & (conf < lo + 0.1)
            if s.any():
                e += s.mean() * abs(corr[s].mean() - conf[s].mean())
        return float(e)

    res = {}
    ind_test = read_jsonl(f"{OUT}/independent_test.jsonl")
    joint_test = read_jsonl(f"{OUT}/joint_test.jsonl")
    for arm, ckpt in (("independent", a.independent), ("joint", a.joint)):
        scorer = MenuScorer(ckpt, use_state_cache=False)
        rows = {"jb": [], "jacc": [], "qb": [], "ub": [], "ece": []}
        with torch.no_grad():
            for k, jr in enumerate(joint_test[: a.limit]):
                post = jr.meta["joint_posterior"]
                st, tgt_q, tgt_u = State(**jr.state), *_marginals(post)
                if arm == "independent":
                    qa, ua = ind_test[2 * k], ind_test[2 * k + 1]
                    pq = np.array(scorer.decide(st, [parse_question(qa.question)])[0].probs)
                    pu = np.array(scorer.decide(st, [parse_question(ua.question)])[0].probs)
                    pj = np.outer(pq, pu).ravel()
                else:
                    pj = np.array(scorer.decide(st, [parse_question(jr.question)])[0].probs)
                    m = pj.reshape(len(QUEUES), len(URGENCY))
                    pq, pu = m.sum(1), m.sum(0)
                rows["jb"].append(soft_brier(pj, post))
                rows["qb"].append(soft_brier(pq, tgt_q))
                rows["ub"].append(soft_brier(pu, tgt_u))
                rows["jacc"].append(int(pj.argmax() == int(np.argmax(post))))
                rows["ece"].append((float(pj.max()), int(pj.argmax() == COMBOS.index(jr.target["label"]))))
        res[arm] = {"joint_soft_brier": float(np.mean(rows["jb"])), "joint_argmax_agreement": float(np.mean(rows["jacc"])),
                    "queue_marginal_soft_brier": float(np.mean(rows["qb"])), "urgency_marginal_soft_brier": float(np.mean(rows["ub"])),
                    "joint_ece_vs_hard_label": ece(rows["ece"]), "n": len(rows["jb"])}
        del scorer
        torch.cuda.empty_cache()
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--n-train", type=int, default=4000)
    g.add_argument("--n-test", type=int, default=1500)
    g.set_defaults(fn=cmd_gen)
    e = sub.add_parser("eval")
    e.add_argument("--independent", required=True)
    e.add_argument("--joint", required=True)
    e.add_argument("--limit", type=int, default=1500)
    e.add_argument("--out", default="runs/variants/a6/result.json")
    e.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
