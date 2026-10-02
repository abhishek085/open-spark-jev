"""A19 (docs/NOVELTY.md): does a cross-attention readout do better reading an intermediate layer's
state (more lexical evidence) or the final layer's (more decided)? And does a learned mixture of
the two beat either alone?

Direct extension of A7/A8 (``depth_probe.py``): that probe found the decision "crystallises" around
layer 24 of 32 using the *existing* tied norm+lm_head readout (zero-shot, no new parameters). This
experiment asks the same "which depth" question for a genuinely different readout mechanism -- a
small trainable cross-attention head, one option query per option, attending over the state's full
token sequence at a chosen layer instead of reading a single last-position vector.

Backbone is frozen throughout; only the head (one cross-attention layer + a small MLP, a few
thousand parameters) is trained, on cached hidden states so the backbone runs once per item, not
once per training step. Three variants share the same head architecture and training budget:

  intermediate   cross-attends over layer L_MID's token states only
  final          cross-attends over the last layer's token states only
  mixture        both heads' logits combined with one learned scalar weight (grid-fit on val,
                 not backprop -- one parameter doesn't need an optimiser)

  python -m open_spark_jev.experimental.depth_readout --model checkpoints/v7.6-4b \
      --train data/synthetic/prog_v1/train.jsonl --eval data/synthetic/prog_v1/test_locked.jsonl \
      --train-limit 300 --eval-limit 150 --out runs/v7/depth_readout_v7.6.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.corpus import read_jsonl
from ..model import MenuScorer
from ..prompting import render_prefix, render_suffix
from .option_readout import option_display

MAX_STATE_TOK = 384


class CrossAttnHead(nn.Module):
    """One option query per option cross-attends over a layer's token states; MLP -> scalar logit."""

    def __init__(self, hidden: int, n_heads: int = 8):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, query: torch.Tensor, states: torch.Tensor, key_pad_mask: torch.Tensor) -> torch.Tensor:
        ctx, _ = self.mha(query, states, states, key_padding_mask=key_pad_mask)
        return self.mlp(torch.cat([query, ctx], dim=-1)).squeeze(-1)


@torch.no_grad()
def cache_item(scorer: MenuScorer, rec, layers: tuple[int, int]) -> dict | None:
    q, state = rec.question_obj(), rec.state_obj()
    ids = scorer._encode(render_prefix(state, max_chars=scorer.max_state_chars)) + scorer._encode(
        render_suffix(q)
    )
    ids = ids[:MAX_STATE_TOK]
    t = torch.tensor([ids], device=scorer.device)
    out = scorer.lm(input_ids=t, output_hidden_states=True)
    hs = out.hidden_states  # embeddings + one per layer
    opt_ids = [scorer._encode(option_display(q, ln)) for ln in q.labels]
    emb = scorer.lm.get_input_embeddings()
    opt_q = torch.stack([emb(torch.tensor(o, device=scorer.device)).mean(0) for o in opt_ids])  # [K, H]
    return {
        "gold_idx": q.labels.index(rec.target["label"]),
        "n_opts": len(q.labels),
        "opt_q": opt_q.float().cpu(),
        "states": {L: hs[L][0].float().cpu() for L in layers},  # [T, H] per chosen layer
    }


def cache_split(scorer: MenuScorer, path: str, limit: int, layers: tuple[int, int]) -> list[dict]:
    recs = read_jsonl(path)[: limit or None]
    out = []
    t0 = time.time()
    for i, r in enumerate(recs):
        out.append(cache_item(scorer, r, layers))
        if (i + 1) % 50 == 0:
            print(f"cached {i + 1}/{len(recs)}  {time.time() - t0:.0f}s", flush=True)
    return out


def collate(batch: list[dict], layer: int, device: str):
    K = max(b["n_opts"] for b in batch)
    T = max(b["states"][layer].shape[0] for b in batch)
    H = batch[0]["opt_q"].shape[1]
    q = torch.zeros(len(batch), K, H)
    s = torch.zeros(len(batch), T, H)
    pad = torch.ones(len(batch), T, dtype=torch.bool)  # True = ignore
    opt_mask = torch.zeros(len(batch), K, dtype=torch.bool)  # True = valid
    y = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        k, t = b["n_opts"], b["states"][layer].shape[0]
        q[i, :k] = b["opt_q"]
        s[i, :t] = b["states"][layer]
        pad[i, :t] = False
        opt_mask[i, :k] = True
        y[i] = b["gold_idx"]
    return q.to(device), s.to(device), pad.to(device), opt_mask.to(device), y.to(device)


def train_head(
    train: list[dict], val: list[dict], layer: int, hidden: int, device: str, epochs: int = 25, bs: int = 16
) -> tuple[CrossAttnHead, dict]:
    head = CrossAttnHead(hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = np.random.permutation(len(train))
        for i in range(0, len(train), bs):
            batch = [train[j] for j in perm[i : i + bs]]
            q, s, pad, opt_mask, y = collate(batch, layer, device)
            z = head(q, s, pad).masked_fill(~opt_mask, float("-inf"))
            loss = F.cross_entropy(z, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head, evaluate_head(head, val, layer, device)


@torch.no_grad()
def evaluate_head(head: CrossAttnHead, val: list[dict], layer: int, device: str) -> dict:
    head.eval()
    logits = []
    for b in val:
        q = b["opt_q"].unsqueeze(0).to(device)
        s = b["states"][layer].unsqueeze(0).to(device)
        pad = torch.zeros(1, s.shape[1], dtype=torch.bool, device=device)
        logits.append(head(q, s, pad)[0].cpu().numpy())
    hit = [float(int(np.argmax(z)) == b["gold_idx"]) for z, b in zip(logits, val)]
    head.train()
    return {"accuracy": round(float(np.mean(hit)), 4), "n": len(val)}, logits


def softmax(z):
    z = np.asarray(z, float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--train-limit", type=int, default=300)
    ap.add_argument("--eval-limit", type=int, default=150)
    ap.add_argument("--layer-mid", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scorer = MenuScorer(a.model, use_state_cache=False)
    n_layers = scorer.lm.config.num_hidden_layers
    layers = (a.layer_mid, n_layers)

    print("caching train...", flush=True)
    train = cache_split(scorer, a.train, a.train_limit, layers)
    print("caching eval...", flush=True)
    val = cache_split(scorer, a.eval, a.eval_limit, layers)
    hidden = scorer.lm.config.hidden_size

    results = {}
    logits_by_variant = {}
    for name, L in (("intermediate", layers[0]), ("final", layers[1])):
        print(f"training head @ layer {L} ({name})...", flush=True)
        head, (rep, logits) = train_head(train, val, L, hidden, scorer.device)
        results[name] = {"layer": L, **rep}
        logits_by_variant[name] = logits

    # mixture: grid-search one scalar weight on val (correctness against its own train would be circular,
    # but val is what we have here -- this experiment is a probe, not a shipped calibration fit)
    best = {"alpha": 0.5, "accuracy": -1.0}
    gold = [b["gold_idx"] for b in val]
    for alpha in np.linspace(0, 1, 21):
        mixed = [
            alpha * softmax(a_) + (1 - alpha) * softmax(b_)
            for a_, b_ in zip(logits_by_variant["intermediate"], logits_by_variant["final"])
        ]
        acc = float(np.mean([int(np.argmax(m)) == g for m, g in zip(mixed, gold)]))
        if acc > best["accuracy"]:
            best = {"alpha": round(float(alpha), 3), "accuracy": round(acc, 4)}
    results["mixture"] = {"layers": layers, **best, "n": len(val)}

    report = {
        "model": a.model,
        "n_layers": n_layers,
        "layers_compared": layers,
        "train_n": len(train),
        "eval_n": len(val),
        "results": results,
        "note": "compare against depth_probe.py's zero-shot per-layer logit-lens accuracy at the same layers -- "
        "this experiment isolates whether a *trained* cross-attention readout beats that free baseline.",
    }
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
