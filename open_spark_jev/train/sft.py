"""Phase 1: supervised multi-task menu-head training.

Loss per example (restricted label logits z in R^K):
    L = CE(softmax(z), hard label)                      if only a hard label exists
      = KL(target_dist || softmax(z))                   if a soft target (posterior / teacher) exists
      + lambda_brier * sum_k (softmax(z)_k - t_k)^2     optional Brier regulariser (calibration)

All question types share the same loss; the model learns the label-slot convention once and
transfers it across domains. After training, a temperature per question type is fitted on
the validation split and saved to ``calibration.json``.

Runs as a plain single-GPU PyTorch loop (bf16 autocast, grad accumulation, optional LoRA
via PEFT). A 1.7B backbone trains fully (no LoRA) in ~40 GB on the Spark's unified memory.

Usage:
    python -m open_spark_jev.train.sft --config configs/train/sft.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..calibration import fit_temperature, summary
from ..data.corpus import Record, read_jsonl, split_records
from ..model import MenuScorer
from ..prompting import render_prompt

log = logging.getLogger("osj.sft")


@dataclass
class Example:
    input_ids: list[int]
    label_ids: list[int]
    target_idx: int
    target_dist: list[float] | None
    qtype: str
    domain: str
    record: Record | None = None  # the source Record, so downstream code (e.g. train/rlcd.py's
    # exact policy gradient, which needs question_obj()/domain for reward shaping) never has to
    # re-derive which Example came from which Record by re-running this function's filters --
    # doing that by position previously mismatched silently whenever a record was dropped here
    # for length but not marked suspect (a filter build_examples applies that callers did not
    # replicate). Keep this field in sync with `records` below; do not filter examples without it.


def build_examples(scorer: MenuScorer, records: list[Record], max_len: int) -> list[Example]:
    out = []
    for r in records:
        if r.meta.get("suspect"):
            continue
        q = r.question_obj()
        ids = scorer.tokenizer.encode(render_prompt(r.state_obj(), q), add_special_tokens=False)
        if len(ids) > max_len:
            continue
        out.append(
            Example(
                input_ids=ids,
                label_ids=scorer.labels.ids_for(len(q.labels)),
                target_idx=r.label_index(),
                target_dist=r.target_dist(),
                qtype=q.type,
                domain=r.domain,
                record=r,
            )
        )
    return out


def collate(batch: list[Example], pad_id: int, device: str):
    L = max(len(e.input_ids) for e in batch)
    K = max(len(e.label_ids) for e in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    last = torch.zeros(len(batch), dtype=torch.long)
    lab = torch.zeros((len(batch), K), dtype=torch.long)
    lab_mask = torch.zeros((len(batch), K), dtype=torch.bool)
    tgt = torch.zeros((len(batch), K), dtype=torch.float)
    hard = torch.zeros(len(batch), dtype=torch.long)
    for i, e in enumerate(batch):
        ids[i, : len(e.input_ids)] = torch.tensor(e.input_ids)
        mask[i, : len(e.input_ids)] = 1
        last[i] = len(e.input_ids) - 1
        lab[i, : len(e.label_ids)] = torch.tensor(e.label_ids)
        lab_mask[i, : len(e.label_ids)] = True
        hard[i] = e.target_idx
        if e.target_dist is not None:
            tgt[i, : len(e.target_dist)] = torch.tensor(e.target_dist)
        else:
            tgt[i, e.target_idx] = 1.0
    return (t.to(device) for t in (ids, mask, last, lab, lab_mask, tgt, hard))


def menu_loss(z: torch.Tensor, tgt: torch.Tensor, lab_mask: torch.Tensor, lambda_brier: float) -> torch.Tensor:
    logp = F.log_softmax(z, dim=-1)
    logp = logp.masked_fill(~lab_mask, 0.0)
    ce = -(tgt * logp).sum(-1)  # = CE for one-hot, cross-entropy part of KL for soft targets
    loss = ce.mean()
    if lambda_brier > 0:
        p = logp.exp().masked_fill(~lab_mask, 0.0)
        loss = loss + lambda_brier * ((p - tgt) ** 2).sum(-1).mean()
    return loss


@torch.no_grad()
def evaluate(scorer: MenuScorer, examples: list[Example], batch_size: int, pad_id: int) -> dict:
    scorer.lm.eval()
    by_type: dict[str, dict] = {}
    for i in range(0, len(examples), batch_size):
        b = examples[i : i + batch_size]
        ids, mask, last, lab, lab_mask, tgt, hard = collate(b, pad_id, scorer.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scorer.device.startswith("cuda")):
            z = scorer.train_forward(ids, mask, last, lab, lab_mask)
        for j, e in enumerate(b):
            d = by_type.setdefault(e.qtype, {"logits": [], "labels": []})
            d["logits"].append(z[j, : len(e.label_ids)].float().cpu().numpy().tolist())
            d["labels"].append(e.target_idx)
    report = {}
    for qt, d in by_type.items():
        # ragged K: evaluate per K group
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
        # summary() needs equal K; pad with zeros (argmax / brier unaffected by zero pads).
        K = max(len(p) for p in probs)
        probs = [p + [0.0] * (K - len(p)) for p in probs]
        report[qt] = {"temperature": T, **summary(probs, ys)}
    return report


def main(cfg_path: str, overrides: dict | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg.update(overrides)
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))

    scorer = MenuScorer(cfg["model"], use_state_cache=False, aux_head=cfg.get("aux_head", False))
    pad_id = scorer.tokenizer.pad_token_id if scorer.tokenizer.pad_token_id is not None else 0

    if cfg.get("lora"):
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"].get("dropout", 0.05),
                        target_modules=cfg["lora"].get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        scorer.lm = get_peft_model(scorer.lm, lc)
        scorer.lm.print_trainable_parameters()
    if cfg.get("gradient_checkpointing", True):
        scorer.lm.gradient_checkpointing_enable()
        scorer.lm.config.use_cache = False

    records: list[Record] = []
    for p in cfg["data"]:
        records.extend(read_jsonl(p))
    log.info("loaded %d records", len(records))
    train_recs, val_recs = split_records(records, cfg.get("val_frac", 0.1), cfg.get("seed", 0))
    train_ex = build_examples(scorer, train_recs, cfg.get("max_len", 2048))
    val_ex = build_examples(scorer, val_recs, cfg.get("max_len", 2048))
    log.info("train %d  val %d examples", len(train_ex), len(val_ex))

    params = [p for p in scorer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 2e-5), weight_decay=cfg.get("weight_decay", 0.01), betas=(0.9, 0.95))
    bs, accum = cfg.get("batch_size", 8), cfg.get("grad_accum", 4)
    epochs = cfg.get("epochs", 2)
    steps_total = math.ceil(len(train_ex) / bs / accum) * epochs
    warm = int(0.05 * steps_total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps_total))))
    )
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    lambda_brier = cfg.get("lambda_brier", 0.5)

    step, t0 = 0, time.time()
    for ep in range(epochs):
        scorer.lm.train()
        random.shuffle(train_ex)
        # length-bucketing keeps padding low
        train_ex.sort(key=lambda e: len(e.input_ids) // 64 + random.random())
        batches = [train_ex[i : i + bs] for i in range(0, len(train_ex), bs)]
        random.shuffle(batches)
        for bi, b in enumerate(batches):
            ids, mask, last, lab, lab_mask, tgt, hard = collate(b, pad_id, scorer.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scorer.device.startswith("cuda")):
                z = scorer.train_forward(ids, mask, last, lab, lab_mask)
                loss = menu_loss(z, tgt, lab_mask, lambda_brier) / accum
            loss.backward()
            if (bi + 1) % accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.get("log_every", 10) == 0:
                    log.info("ep %d step %d/%d loss %.4f lr %.2e %.1fs", ep, step, steps_total, loss.item() * accum, sched.get_last_lr()[0], time.time() - t0)
        rep = evaluate(scorer, val_ex, bs, pad_id)
        log.info("epoch %d val: %s", ep, json.dumps(rep, indent=None))
        with open(os.path.join(out_dir, f"val_epoch{ep}.json"), "w") as f:
            json.dump(rep, f, indent=2)

    # merge LoRA, store temperatures, save
    if cfg.get("lora"):
        scorer.lm = scorer.lm.merge_and_unload()
    scorer.calibration.temperature = {qt: r["temperature"] for qt, r in rep.items()}
    scorer.save_pretrained(out_dir)
    log.info("saved to %s with temperatures %s", out_dir, scorer.calibration.temperature)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (yaml-parsed)")
    a = ap.parse_args()
    ov = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    main(a.config, ov)
