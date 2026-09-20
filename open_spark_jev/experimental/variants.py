"""Matched-budget architecture variants A0/A2/A3/A4 (docs/NOVELTY.md).

Every variant shares the same LoRA-adapted Qwen3 backbone, data, loss family, seed and
budget; only the *readout* differs, so differences are attributable to the architecture:

  a0  restricted LM-head readout at the last position (the baseline mechanism)
  a2  prefix-LM attention: bidirectional over the state span, causal after it
  a3  slot-query head: 26 learned option-slot queries cross-attend over all hidden states
  a4  evidential (Dirichlet) head on the last hidden state, evidential-Brier loss

Usage:
  python -m open_spark_jev.experimental.variants --variant a3 --out runs/variants/a3
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..calibration import fit_temperature, summary
from ..data.corpus import read_jsonl, split_records
from ..model import MenuScorer, gather_label_logits
from ..prompting import render_prefix
from ..schema import MAX_OPTIONS
from ..train.sft import build_examples, collate, menu_loss

log = logging.getLogger("osj.variants")
QTYPES = ("choice", "score", "noul")


class SlotQueryHead(nn.Module):
    """A3. One learned query per option slot (+ a question-type embedding) cross-attends over
    the backbone's hidden states; each attended vector is scored by a small MLP -> one logit."""

    def __init__(self, hidden: int, d: int = 512, heads: int = 8):
        super().__init__()
        self.kv = nn.Linear(hidden, d)
        self.slot = nn.Parameter(torch.randn(MAX_OPTIONS, d) * 0.02)
        self.qtype = nn.Parameter(torch.randn(len(QTYPES), d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor, qtype_idx: torch.Tensor, k: int) -> torch.Tensor:
        kv = self.kv(h.float())
        q = self.slot[None, :k] + self.qtype[qtype_idx][:, None]
        a, _ = self.attn(q, kv, kv, key_padding_mask=~pad_mask)
        return self.mlp(self.norm(a + q)).squeeze(-1)  # [B, K]


class VariantModel(nn.Module):
    def __init__(self, base: MenuScorer, variant: str):
        super().__init__()
        self.base, self.variant = base, variant
        hid = base.lm.config.hidden_size
        self.head: nn.Module | None = None
        if variant == "a3":
            self.head = SlotQueryHead(hid).to(base.device)
        elif variant == "a4":
            self.head = nn.Linear(hid, MAX_OPTIONS).to(base.device)

    def _prefix_mask(self, mask: torch.Tensor, prefix_len: torch.Tensor) -> torch.Tensor:
        """Bool [B,1,L,L]: causal everywhere, plus full bidirectional attention inside the
        state prefix (A2). Padding keys are masked out."""
        B, L = mask.shape
        i = torch.arange(L, device=mask.device)
        causal = i[:, None] >= i[None, :]
        in_prefix = (i[None, :, None] < prefix_len[:, None, None]) & (i[None, None, :] < prefix_len[:, None, None])
        allowed = (causal[None] | in_prefix) & mask.bool()[:, None, :]
        return allowed[:, None]

    def logits(self, ids, mask, last, lab, lab_mask, prefix_len, qtype_idx) -> torch.Tensor:
        lm = self.base.lm
        rows = torch.arange(ids.shape[0], device=ids.device)
        if self.variant == "a0":
            out = lm(input_ids=ids, attention_mask=mask)
            return gather_label_logits(out.logits[rows, last].float(), lab, lab_mask)
        if self.variant == "a2":
            out = lm(input_ids=ids, attention_mask=self._prefix_mask(mask, prefix_len))
            return gather_label_logits(out.logits[rows, last].float(), lab, lab_mask)
        # inner transformer (no LM head, so no [B,L,151k] logits); LoRA layers are injected in place so they stay active
        inner = (lm.get_base_model() if hasattr(lm, "get_base_model") else lm).model
        h = inner(input_ids=ids, attention_mask=mask).last_hidden_state
        k = lab.shape[1]
        if self.variant == "a3":
            z = self.head(h, mask.bool(), qtype_idx, k)
        else:  # a4: evidence logits from last-token hidden state
            z = self.head(h[rows, last].float())[:, :k]
        return z.masked_fill(~lab_mask, float("-inf"))


def evidential_loss(z: torch.Tensor, tgt: torch.Tensor, lab_mask: torch.Tensor, kl_w: float = 0.01) -> torch.Tensor:
    """Sensoy et al. evidential-Brier: E||y-p||^2 under Dir(alpha) + KL(misleading evidence||uniform)."""
    ev = F.softplus(z.masked_fill(~lab_mask, -30.0)).masked_fill(~lab_mask, 0.0)
    alpha = ev + 1.0
    S = alpha.masked_fill(~lab_mask, 0.0).sum(-1, keepdim=True)
    p = alpha / S
    err = ((tgt - p) ** 2).masked_fill(~lab_mask, 0.0).sum(-1)
    var = (p * (1 - p) / (S + 1)).masked_fill(~lab_mask, 0.0).sum(-1)
    at = (tgt + (1 - tgt) * alpha).masked_fill(~lab_mask, 1.0)
    k = lab_mask.sum(-1).float()
    S_t = at.sum(-1)
    kl = (torch.lgamma(S_t) - torch.lgamma(k) - torch.lgamma(at).masked_fill(~lab_mask, 0.0).sum(-1)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(S_t)[:, None])).masked_fill(~lab_mask, 0.0).sum(-1))
    return (err + var + kl_w * kl).mean()


def a4_probs_logits(z: torch.Tensor, lab_mask: torch.Tensor) -> torch.Tensor:
    """Log of the evidential mean p=alpha/S, usable as logits for temperature scaling."""
    alpha = F.softplus(z.masked_fill(~lab_mask, -30.0)).masked_fill(~lab_mask, 0.0) + 1.0
    p = alpha / alpha.masked_fill(~lab_mask, 0.0).sum(-1, keepdim=True)
    return torch.log(p.clamp_min(1e-9)).masked_fill(~lab_mask, float("-inf"))


def prefix_lengths(scorer: MenuScorer, examples) -> list[int]:
    out = []
    for e in examples:
        st = e.record.state_obj()
        pre = scorer.tokenizer.encode(render_prefix(st), add_special_tokens=False)
        n = 0
        for a, b in zip(pre, e.input_ids):
            if a != b:
                break
            n += 1
        out.append(n)
    return out


def make_batch(b, pad_id, device, plen):
    ids, mask, last, lab, lab_mask, tgt, hard = collate(b, pad_id, device)
    pl = torch.tensor([plen[id(e)] for e in b], device=device)
    qi = torch.tensor([QTYPES.index(e.qtype) for e in b], device=device)
    return ids, mask, last, lab, lab_mask, tgt, hard, pl, qi


@torch.no_grad()
def collect_logits(model: VariantModel, examples, plen, bs, pad_id):
    model.eval()
    out = []
    for i in range(0, len(examples), bs):
        b = examples[i : i + bs]
        ids, mask, last, lab, lab_mask, tgt, hard, pl, qi = make_batch(b, pad_id, model.base.device, plen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.logits(ids, mask, last, lab, lab_mask, pl, qi)
        if model.variant == "a4":
            z = a4_probs_logits(z.float(), lab_mask)
        for j, e in enumerate(b):
            out.append(z[j, : len(e.label_ids)].float().cpu().numpy())
    return out


def report(val_logits, val_ex, test_logits, test_ex) -> dict:
    """Fit one temperature per qtype on val, apply to test; overall + per-domain metrics."""
    T = {}
    for qt in QTYPES:
        groups: dict[int, tuple[list, list]] = {}
        for z, e in zip(val_logits, val_ex):
            if e.qtype == qt:
                g = groups.setdefault(len(z), ([], []))
                g[0].append(z)
                g[1].append(e.target_idx)
        ts = [fit_temperature(np.array(zl), np.array(yl)) for zl, yl in groups.values() if len(yl) >= 20]
        T[qt] = float(np.mean(ts)) if ts else 1.0
    rows = []
    for z, e in zip(test_logits, test_ex):
        p = np.exp((z - z.max()) / T[e.qtype])
        p /= p.sum()
        rows.append((e, p))

    def agg(sel):
        if not sel:
            return None
        K = max(len(p) for _, p in sel)
        probs = [list(p) + [0.0] * (K - len(p)) for _, p in sel]
        ys = [e.target_idx for e, _ in sel]
        r = summary(probs, ys)
        soft = [(np.asarray(p, dtype=np.float64), np.asarray(e.target_dist, dtype=np.float64)) for e, p in sel if e.target_dist is not None]
        if soft:  # per-row (K differs across questions), then mean
            r["soft_brier"] = float(np.mean([np.sum((p_ - t_) ** 2) for p_, t_ in soft]))
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}

    rep = {"temperature": T, "overall": agg(rows)}
    for d in sorted({e.domain for e, _ in rows}):
        rep[d] = agg([(e, p) for e, p in rows if e.domain == d])
    return rep


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["a0", "a2", "a3", "a4"])
    ap.add_argument("--init", default="models/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--extra-data", nargs="*", default=[], help="extra training jsonl files pooled with sim+teacher before sampling (e.g. M17 public text)")
    ap.add_argument("--always-data", nargs="*", default=[], help="jsonl files included in full; the rest of --n-train is sampled from sim+teacher(+extra)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run for plumbing checks")
    ap.add_argument("--save-dir", default=None, help="where to save the servable variant (default checkpoints/variants/<variant>; not saved with --smoke)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    base = MenuScorer(a.init, use_state_cache=False, attn_implementation="sdpa")
    pad_id = base.tokenizer.pad_token_id or 0
    from peft import LoraConfig, get_peft_model

    base.lm = get_peft_model(base.lm, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    base.lm.gradient_checkpointing_enable()
    base.lm.enable_input_require_grads()
    base.lm.config.use_cache = False
    model = VariantModel(base, a.variant)

    recs = read_jsonl("data/synthetic/sim_train.jsonl") + read_jsonl("data/synthetic/teacher_train_split.jsonl")
    for _extra in a.extra_data:
        recs += read_jsonl(_extra)
    random.Random(a.seed).shuffle(recs)
    always = [r for _f in a.always_data for r in read_jsonl(_f)]
    recs = recs[: 300 if a.smoke else max(0, a.n_train - len(always))] + always
    random.Random(a.seed + 1).shuffle(recs)
    train_r, val_r = split_records(recs, 0.1, a.seed)
    train_ex = build_examples(base, train_r, a.max_len)
    val_ex = build_examples(base, val_r, a.max_len)
    tests = {"sim_test": build_examples(base, read_jsonl("data/benchmarks/sim_test.jsonl")[: 200 if a.smoke else None], a.max_len),
             "teacher_test": build_examples(base, read_jsonl("data/benchmarks/teacher_test.jsonl")[: 100 if a.smoke else None], a.max_len)}
    import glob as _glob
    for _f in sorted(_glob.glob("data/benchmarks/external/*.jsonl")):
        _recs = read_jsonl(_f)[: 40 if a.smoke else None]
        tests[os.path.basename(_f)[:-6]] = build_examples(base, _recs, a.max_len)
        log.info("external %s: %d of %d records fit max_len", os.path.basename(_f)[:-6], len(tests[os.path.basename(_f)[:-6]]), len(_recs))
    plen = {}
    for ex in (train_ex, val_ex, *tests.values()):
        for e, n in zip(ex, prefix_lengths(base, ex)):
            plen[id(e)] = n
    log.info("variant %s: train %d val %d tests %s", a.variant, len(train_ex), len(val_ex), {k: len(v) for k, v in tests.items()})

    lora = [p for n, p in model.base.lm.named_parameters() if p.requires_grad]
    groups = [{"params": lora, "lr": a.lr}]
    if model.head is not None:
        groups.append({"params": list(model.head.parameters()), "lr": a.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.95))
    steps_total = math.ceil(len(train_ex) / a.bs / a.accum) * a.epochs
    warm = max(1, int(0.05 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))
    step, t0 = 0, time.time()
    for _ep in range(a.epochs):
        model.train()
        train_ex.sort(key=lambda e: len(e.input_ids) // 64 + random.random())
        batches = [train_ex[i : i + a.bs] for i in range(0, len(train_ex), a.bs)]
        random.shuffle(batches)
        for bi, b in enumerate(batches):
            ids, mask, last, lab, lab_mask, tgt, hard, pl, qi = make_batch(b, pad_id, base.device, plen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.logits(ids, mask, last, lab, lab_mask, pl, qi)
                loss = (evidential_loss(z.float(), tgt, lab_mask) if a.variant == "a4" else menu_loss(z.float(), tgt, lab_mask, 0.5)) / a.accum
            loss.backward()
            if (bi + 1) % a.accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    log.info("step %d/%d loss %.4f %.0fs", step, steps_total, loss.item() * a.accum, time.time() - t0)
    train_seconds = time.time() - t0
    val_logits = collect_logits(model, val_ex, plen, a.bs, pad_id)
    result = {"variant": a.variant, "train_seconds": round(train_seconds), "n_train": len(train_ex),
              "trainable_params": sum(p.numel() for p in lora) + (sum(p.numel() for p in model.head.parameters()) if model.head else 0)}
    for name, ex in tests.items():
        result[name] = report(val_logits, val_ex, collect_logits(model, ex, plen, a.bs, pad_id), ex)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if not a.smoke or a.save_dir:
        save_variant(model, a, result["sim_test"]["temperature"], result)
    print(json.dumps({k: (v["overall"] if isinstance(v, dict) and "overall" in v else v) for k, v in result.items()}, indent=2))


def save_variant(model: VariantModel, a, temperature: dict, result: dict) -> None:
    """Persist LoRA adapter + head + calibration so the variant can be served and re-scored
    (experimental/variant_scorer.py). Weights are otherwise thrown away after a run."""
    d = a.save_dir or os.path.join("checkpoints", "variants", a.variant)
    os.makedirs(d, exist_ok=True)
    model.base.lm.save_pretrained(os.path.join(d, "lora"))
    if model.head is not None:
        torch.save(model.head.state_dict(), os.path.join(d, "head.pt"))
    with open(os.path.join(d, "variant.json"), "w") as f:
        json.dump({"variant": a.variant, "init": a.init, "temperature": temperature, "n_train": result["n_train"],
                   "train_seconds": result["train_seconds"], "date": time.strftime("%Y-%m-%d")}, f, indent=2)
    log.info("saved servable variant to %s", d)


if __name__ == "__main__":
    main()
