"""RLDM-3: learn the A11 cascade gate with an exact expected-reward update instead of hand-fitting one threshold.

Decision per item: a in {exit at layer 24, continue to layer 32}. Reward = correctness of the answer that would be
served, minus a compute price lambda * (fraction of the 32-layer pass spent: 0.75 for exit, 1.0 for continue).
Because there are only two actions and both outcomes are already cached (depth_probe.py's per-layer logit lens),
the expected reward J = p_exit*R_exit + (1-p_exit)*R_cont is computed exactly for every item -- no sampling, the same
"exact policy gradient over a small menu" design as RLCD-direct (docs/RESEARCH.md), and no saturation loop since the
backbone is frozen and the policy is ~100 parameters. Honest framing: with two actions and full-information rewards this is
a cost-sensitive contextual bandit; the novelty is (a) the reward couples accuracy to compute directly and (b) the state
includes A8's layer-disagreement signal, which the single-threshold gate cannot use.

Fit on the FULL calibration split only (never a head-of-file --limit: prog_v1 files are grouped by family), evaluate on
splits never used for fitting. Compared against A11's threshold gate on the same Pareto axes.

  python -m open_spark_jev.experimental.gate_rl --fit runs/v7/depth_probe_v7.6_calibration_full.json \
      --eval runs/v7/depth_probe_v7.6_test_locked_full.json runs/v7/depth_probe_v7.6_challenge_full.json \
             runs/v7/depth_probe_v7.6_hard.json runs/v7/depth_probe_v7.6_ext-*.json --out runs/v7/gate_rl_v7.6.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

EXIT_L, FINAL_L, EXIT_COST = 24, 32, 0.75


def softmax(z):
    z = np.asarray(z, float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def load(path):
    per = json.load(open(path))["per_item"]
    X, c_exit, c_final = [], [], []
    for r in per:
        L = r["logits_by_layer"]
        p24, p32 = softmax(L[EXIT_L]), softmax(L[FINAL_L])
        srt = np.sort(p24)[::-1]
        ent = -(p24 * np.log(p24 + 1e-12)).sum() / max(np.log(len(p24)), 1e-9)
        tops = [int(np.argmax(softmax(L[k]))) for k in range(EXIT_L - 7, EXIT_L + 1)]
        flips = len(set(tops)) - 1
        tvd = 0.5 * np.abs(softmax(L[EXIT_L - 4]) - p24).sum()
        X.append([srt[0], srt[0] - (srt[1] if len(srt) > 1 else 0.0), ent, np.log(len(p24)), flips, tvd])
        c_exit.append(float(np.argmax(p24) == r["gold_idx"]))
        c_final.append(float(np.argmax(p32) == r["gold_idx"]))
    return np.array(X, np.float32), np.array(c_exit, np.float32), np.array(c_final, np.float32)


def fit_policy(X, c_exit, c_final, lam, hidden=16, steps=600, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd)
    r_exit = torch.tensor(c_exit) - lam * EXIT_COST
    r_cont = torch.tensor(c_final) - lam * 1.0
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=0.02, weight_decay=1e-3)
    for _ in range(steps):
        p = torch.sigmoid(net(Xt)).squeeze(-1)
        J = (p * r_exit + (1 - p) * r_cont).mean()  # exact expected reward, no sampling
        opt.zero_grad()
        (-J).backward()
        opt.step()
    return net, mu, sd


def act(net, mu, sd, X):
    with torch.no_grad():
        return (torch.sigmoid(net(torch.tensor((X - mu) / sd))).squeeze(-1) > 0.5).numpy()


def score(exit_mask, c_exit, c_final):
    n = len(c_exit)
    acc = np.where(exit_mask, c_exit, c_final).mean()
    return {
        "exit_fraction": round(float(exit_mask.mean()), 4),
        "relative_compute": round(float((exit_mask.sum() * EXIT_COST + (n - exit_mask.sum())) / n), 4),
        "cascade_accuracy": round(float(acc), 4),
        "full_depth_accuracy": round(float(c_final.mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Xf, cef, cff = load(a.fit)
    evals = {p: load(p) for p in a.eval}
    thr_grid = np.linspace(0.5, 1.0, 26)
    report = {
        "fit_on": a.fit,
        "n_fit": len(Xf),
        "features": ["conf24", "margin24", "entropy24", "log_n_labels", "flips17-24", "tvd20-24"],
        "sets": {},
    }
    for path, (X, ce, cf) in evals.items():
        entry = {"n": len(X), "threshold_gate": {}, "gate_rl": {}}
        for t in thr_grid:
            entry["threshold_gate"][f"{t:.2f}"] = score(X[:, 0] >= t, ce, cf)
        report["sets"][path] = entry
    for lam in a.lambdas:
        net, mu, sd = fit_policy(Xf, cef, cff, lam)
        fit_s = score(act(net, mu, sd, Xf), cef, cff)
        print(f"lambda={lam}: fit-set {fit_s}", flush=True)
        for path, (X, ce, cf) in evals.items():
            report["sets"][path]["gate_rl"][str(lam)] = score(act(net, mu, sd, X), ce, cf)

    # zero-loss operating points: max exit_fraction with cascade acc >= full acc - 0.002
    summary = {}
    for path, e in report["sets"].items():

        def best(d):
            ok = [
                (v["exit_fraction"], k, v)
                for k, v in d.items()
                if v["cascade_accuracy"] >= v["full_depth_accuracy"] - 0.002
            ]
            return max(ok, key=lambda x: x[0]) if ok else None

        bt, br = best(e["threshold_gate"]), best(e["gate_rl"])
        summary[path] = {
            "threshold_gate_best_zero_loss": (bt[1], bt[2]) if bt else None,
            "gate_rl_best_zero_loss": (br[1], br[2]) if br else None,
        }
    report["zero_loss_summary"] = summary
    json.dump(report, open(a.out, "w"), indent=1)
    for path, s in summary.items():
        tg, rl = s["threshold_gate_best_zero_loss"], s["gate_rl_best_zero_loss"]
        print(
            path.split("v7.6_")[-1],
            "| thr-gate:",
            (tg[0], tg[1]["exit_fraction"], tg[1]["cascade_accuracy"]) if tg else None,
            "| RL gate:",
            (rl[0], rl[1]["exit_fraction"], rl[1]["cascade_accuracy"]) if rl else None,
        )
    print("wrote", a.out)


if __name__ == "__main__":
    main()
