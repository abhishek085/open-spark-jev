"""Train control-jev-es: a ModernBERT encoder + a from-scratch linear scorer head, fully fine-tuned.

Same loss as open_spark_jev.train.sft's menu_loss (architecture-agnostic: it only needs restricted
per-option logits and a target distribution), same per-question-type temperature fit as sft.py, applied
to the option-marker-at-[MASK] readout described in experimental/control_es.py instead of a decoder's
letter-slot readout.

Usage:
    python -m open_spark_jev.train.sft_control_es --config configs/train/sft_control_jev_es.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from ..calibration import fit_temperature, summary
from ..data.corpus import Record, read_jsonl, split_records
from ..experimental.control_es import ScoreHead, render_es_prompt
from .sft import menu_loss

log = logging.getLogger("osj.sft_control_es")


@dataclass
class Example:
    input_ids: list[int]
    mask_positions: list[int]  # one per label, in order
    target_idx: int
    target_dist: list[float] | None
    qtype: str


def build_examples(tokenizer, records: list[Record], max_len: int) -> list[Example]:
    mask_id = tokenizer.mask_token_id
    out = []
    for r in records:
        q = r.question_obj()
        text = render_es_prompt(r.state_obj(), q, tokenizer.mask_token)
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)
        positions = [i for i, t in enumerate(ids) if t == mask_id]
        if len(positions) != len(q.labels):
            continue  # truncated before all option markers -- drop, don't silently mis-align
        out.append(Example(input_ids=ids, mask_positions=positions, target_idx=r.label_index(),
                            target_dist=r.target_dist(), qtype=q.type))
    return out


def collate(batch: list[Example], pad_id: int, device: str):
    L = max(len(e.input_ids) for e in batch)
    K = max(len(e.mask_positions) for e in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    pos = torch.zeros((len(batch), K), dtype=torch.long)
    lab_mask = torch.zeros((len(batch), K), dtype=torch.bool)
    tgt = torch.zeros((len(batch), K), dtype=torch.float)
    for i, e in enumerate(batch):
        ids[i, : len(e.input_ids)] = torch.tensor(e.input_ids)
        attn[i, : len(e.input_ids)] = 1
        pos[i, : len(e.mask_positions)] = torch.tensor(e.mask_positions)
        lab_mask[i, : len(e.mask_positions)] = True
        if e.target_dist is not None:
            tgt[i, : len(e.target_dist)] = torch.tensor(e.target_dist)
        else:
            tgt[i, e.target_idx] = 1.0
    return (t.to(device) for t in (ids, attn, pos, lab_mask, tgt))


def score(backbone, head: ScoreHead, ids, attn, pos, lab_mask) -> torch.Tensor:
    hidden = backbone(input_ids=ids, attention_mask=attn).last_hidden_state  # [B, L, H]
    B, K = pos.shape
    gathered = torch.gather(hidden, 1, pos.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))  # [B, K, H]
    z = head(gathered)  # [B, K]
    return z.masked_fill(~lab_mask, float("-inf"))


@torch.no_grad()
def evaluate(backbone, head, tokenizer, examples: list[Example], batch_size: int, pad_id: int, device: str) -> dict:
    backbone.eval()
    by_type: dict[str, dict] = {}
    for i in range(0, len(examples), batch_size):
        b = examples[i : i + batch_size]
        ids, attn, pos, lab_mask, tgt = collate(b, pad_id, device)
        z = score(backbone, head, ids, attn, pos, lab_mask)
        for j, e in enumerate(b):
            d = by_type.setdefault(e.qtype, {"logits": [], "labels": []})
            d["logits"].append(z[j, : len(e.mask_positions)].float().cpu().numpy().tolist())
            d["labels"].append(e.target_idx)
    report = {}
    for qt, d in by_type.items():
        groups: dict[int, tuple[list, list]] = {}
        for zz, y in zip(d["logits"], d["labels"]):
            g = groups.setdefault(len(zz), ([], []))
            g[0].append(zz)
            g[1].append(y)
        T = float(np.mean([fit_temperature(np.array(zl), np.array(yl)) for zl, yl in groups.values() if len(yl) >= 20] or [1.0]))
        probs, ys = [], []
        for zl, yl in groups.values():
            zl = np.array(zl) / T
            p = np.exp(zl - zl.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            probs.extend(p.tolist())
            ys.extend(yl)
        K = max(len(p) for p in probs)
        probs = [p + [0.0] * (K - len(p)) for p in probs]
        report[qt] = {"temperature": T, **summary(probs, ys)}
    return report


def main(cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    backbone = AutoModel.from_pretrained(cfg["model"], dtype=torch.float32).to(device)
    head = ScoreHead(backbone.config.hidden_size).to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    records: list[Record] = []
    for p in cfg["data"]:
        records.extend(read_jsonl(p))
    log.info("loaded %d records", len(records))
    train_recs, val_recs = split_records(records, cfg.get("val_frac", 0.1), cfg.get("seed", 0))
    train_ex = build_examples(tokenizer, train_recs, cfg.get("max_len", 512))
    val_ex = build_examples(tokenizer, val_recs, cfg.get("max_len", 512))
    log.info("train %d  val %d examples", len(train_ex), len(val_ex))

    params = list(backbone.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 3e-5), weight_decay=cfg.get("weight_decay", 0.01), betas=(0.9, 0.95))
    bs, accum = cfg.get("batch_size", 32), cfg.get("grad_accum", 1)
    epochs = cfg.get("epochs", 3)
    lambda_brier = cfg.get("lambda_brier", 0.5)
    steps_total = (len(train_ex) // bs // accum) * epochs
    warm = max(1, int(0.05 * steps_total))
    import math

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps_total))))
    )

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        backbone.train()
        idx = list(range(len(train_ex)))
        random.Random(cfg.get("seed", 0) + ep).shuffle(idx)
        opt.zero_grad()
        for bi in range(0, len(idx), bs):
            batch = [train_ex[j] for j in idx[bi : bi + bs]]
            ids, attn, pos, lab_mask, tgt = collate(batch, pad_id, device)
            z = score(backbone, head, ids, attn, pos, lab_mask)
            loss = menu_loss(z, tgt, lab_mask, lambda_brier) / accum
            loss.backward()
            if (bi // bs + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % cfg.get("log_every", 10) == 0:
                    log.info("ep %d step %d/%d loss %.4f lr %.2e %.1fs", ep, step, steps_total,
                              float(loss.detach()) * accum, sched.get_last_lr()[0], time.time() - t0)
        report = evaluate(backbone, head, tokenizer, val_ex, bs, pad_id, device)
        log.info("epoch %d val: %s", ep, {k: {kk: vv for kk, vv in v.items() if kk != "confusion"} for k, v in report.items()})

    backbone.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(head.state_dict(), os.path.join(out_dir, "score_head.pt"))
    temps = {qt: r["temperature"] for qt, r in report.items()}
    import json

    json.dump({"temperature": temps}, open(os.path.join(out_dir, "calibration.json"), "w"), indent=2)
    json.dump({"kind": "control-jev-es", "base_model": cfg["model"], "readout": "option-marker-at-mask"},
               open(os.path.join(out_dir, "control_es.json"), "w"), indent=2)
    log.info("saved to %s with temperatures %s", out_dir, temps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    main(a.config)
