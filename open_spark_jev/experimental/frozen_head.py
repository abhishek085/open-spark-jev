"""spark-s1-v2: frozen backbone + small trained head on the answer-position hidden state.

The backbone is never updated. We cache the last-prompt-token hidden state once, train an MLP that maps it to
per-slot logits (masked to the question's option count), and score it through the shared osdg evaluator.

  python -m open_spark_jev.experimental.frozen_head --base models/Qwen3-1.7B --name v2-1.7b
"""

from __future__ import annotations

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.corpus import read_jsonl
from ..eval.osdg import evaluate
from ..model import MenuScorer
from ..prompting import render_prefix, render_suffix
from ..schema import Answer


class Head(nn.Module):
    def __init__(self, d, hidden=1024, k=26, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, k))

    def forward(self, h):
        return self.net(h)


@torch.no_grad()
def feature(sc: MenuScorer, rec):
    ids = sc._encode(render_prefix(rec.state_obj(), max_chars=sc.max_state_chars)) + sc._encode(render_suffix(rec.question_obj()))
    x = torch.tensor([ids], device=sc.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = sc.lm.model(input_ids=x).last_hidden_state[0, -1]
    return h.float().cpu()


class HeadScorer:
    def __init__(self, sc, head):
        self.sc, self.head = sc, head.to(sc.device).eval()

    @torch.no_grad()
    def decide(self, state, questions, temperature=None, return_logits=False):
        outs = []
        for q in questions:
            rec = type("R", (), {"state_obj": lambda s, st=state: st, "question_obj": lambda s, qq=q: qq})()
            z = self.head(feature(self.sc, rec).to(self.sc.device))[: len(q.labels)]
            outs.append(Answer.from_probs(q, F.softmax(z, -1).tolist(), raw_logits=z.tolist() if return_logits else None))
        return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/synthetic/osdg_train_aug.jsonl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    sc = MenuScorer(a.base, use_state_cache=False)
    for p in sc.lm.parameters():
        p.requires_grad_(False)
    recs = read_jsonl(a.data)
    # split by source record id so permuted copies of one row never straddle train/val
    key = lambda r: r.id.rsplit("-p", 1)[0] if r.id[-3:-1] == "-p" else r.id  # noqa: E731
    base_ids = sorted({key(r) for r in recs})
    val_ids = set(random.Random(a.seed).sample(base_ids, len(base_ids) // 10))
    X = torch.stack([feature(sc, r) for r in recs])
    y = torch.tensor([r.label_index() for r in recs])
    K = torch.tensor([len(r.question_obj().labels) for r in recs])
    isval = torch.tensor([key(r) in val_ids for r in recs])
    head = Head(X.shape[1]).to(sc.device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    Xtr, ytr, Ktr = X[~isval].to(sc.device), y[~isval].to(sc.device), K[~isval].to(sc.device)
    Xv, yv, Kv = X[isval].to(sc.device), y[isval].to(sc.device), K[isval].to(sc.device)
    mask = lambda k: torch.arange(26, device=sc.device)[None] < k[:, None]  # noqa: E731
    best, best_state = 1e9, None
    for _ep in range(a.epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=sc.device)
        for i in range(0, len(perm), 64):
            b = perm[i : i + 64]
            z = head(Xtr[b]).masked_fill(~mask(Ktr[b]), -1e4)
            loss = F.cross_entropy(z, ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            zv = head(Xv).masked_fill(~mask(Kv), -1e4)
            vl = F.cross_entropy(zv, yv).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    print(f"head trained: {len(Xtr)} train / {len(Xv)} val rows, best val NLL {best:.3f}", flush=True)
    out = os.path.join("checkpoints", a.name)
    os.makedirs(out, exist_ok=True)
    torch.save({"head": head.state_dict(), "base": a.base}, os.path.join(out, "head.pt"))
    evaluate(HeadScorer(sc, head), f"frozen:{a.base}", a.name + "-osdg", perms=3)


if __name__ == "__main__":
    main()
