"""A4 (docs/NOVELTY.md): train the v7.x data/LoRA recipe with an evidential (Dirichlet) readout
head instead of the restricted LM-head, ported from the original 1.7B/old-data A0-A3 comparison
harness (experimental/variants.py) onto the current 4B pipeline so it is directly comparable to
v7.2/v7.3/v7.6's numbers.

Identical to train/sft.py except: the model is built with aux_head=True (a small
Linear(hidden, MAX_OPTIONS) reading the last hidden state instead of the tied LM head, already
supported by MenuScorer/train_forward -- see model.py), and the loss is
experimental.variants.evidential_loss (Sensoy et al. evidential-Brier) instead of menu_loss.

Note MenuScorer.decide() (used by eval/osdg.py, eval/external.py, etc.) does NOT read aux_head --
it always uses the restricted LM-head. A checkpoint trained here needs experimental/eval_a4.py,
not the standard eval scripts, to actually score the trained head.

  python -m open_spark_jev.train.sft_a4 --config configs/train/sft_v7_7.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time

import torch
import yaml

from ..data.corpus import Record, read_jsonl, split_records
from ..experimental.variants import a4_probs_logits, evidential_loss
from ..model import MenuScorer
from ..train.sft import build_examples, collate

log = logging.getLogger("osj.sft_a4")


def main(cfg_path: str, overrides: dict | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg.update(overrides)
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))

    scorer = MenuScorer(cfg["model"], use_state_cache=False, aux_head=True)

    pad_id = scorer.tokenizer.pad_token_id if scorer.tokenizer.pad_token_id is not None else 0

    if cfg.get("lora"):
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"].get("dropout", 0.05),
            target_modules=cfg["lora"].get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
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
    opt = torch.optim.AdamW(
        params, lr=cfg.get("lr", 2e-5), weight_decay=cfg.get("weight_decay", 0.01), betas=(0.9, 0.95)
    )
    bs, accum = cfg.get("batch_size", 8), cfg.get("grad_accum", 4)
    epochs = cfg.get("epochs", 2)
    steps_total = math.ceil(len(train_ex) / bs / accum) * epochs
    warm = max(1, int(0.05 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps_total))))
        ),
    )
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    step, t0 = 0, time.time()
    for ep in range(epochs):
        scorer.lm.train()
        random.shuffle(train_ex)
        train_ex.sort(key=lambda e: len(e.input_ids) // 64 + random.random())
        batches = [train_ex[i : i + bs] for i in range(0, len(train_ex), bs)]
        random.shuffle(batches)
        for bi, b in enumerate(batches):
            ids, mask, last, lab, lab_mask, tgt, hard = collate(b, pad_id, scorer.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scorer.device.startswith("cuda")):
                z = scorer.train_forward(ids, mask, last, lab, lab_mask)
                loss = evidential_loss(z.float(), tgt, lab_mask) / accum
            loss.backward()
            if (bi + 1) % accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.get("log_every", 10) == 0:
                    log.info(
                        "ep %d step %d/%d loss %.4f lr %.2e %.1fs",
                        ep,
                        step,
                        steps_total,
                        loss.item() * accum,
                        sched.get_last_lr()[0],
                        time.time() - t0,
                    )

        # validation: route raw evidence logits through the evidential mean, then reuse the
        # standard fit_temperature/summary machinery on log(p) -- see a4_probs_logits docstring
        scorer.lm.eval()
        rep = evaluate_a4(scorer, val_ex, bs, pad_id)
        log.info("epoch %d val: %s", ep, json.dumps(rep, indent=None))
        with open(os.path.join(out_dir, f"val_epoch{ep}.json"), "w") as f:
            json.dump(rep, f, indent=2)

    if cfg.get("lora"):
        scorer.lm = scorer.lm.merge_and_unload()
    scorer.calibration.temperature = {qt: r["temperature"] for qt, r in rep.items()}
    scorer.save_pretrained(out_dir)
    log.info("saved to %s with temperatures %s", out_dir, scorer.calibration.temperature)


@torch.no_grad()
def evaluate_a4(scorer: MenuScorer, examples, batch_size: int, pad_id: int) -> dict:
    """Same shape as train/sft.py's evaluate(), but reads a4_probs_logits (evidential mean, in
    log space) instead of raw restricted-LM-head logits before fitting/reporting temperature."""
    import numpy as np

    from ..calibration import fit_temperature, summary

    scorer.lm.eval()
    by_type: dict[str, dict] = {}
    for i in range(0, len(examples), batch_size):
        b = examples[i : i + batch_size]
        ids, mask, last, lab, lab_mask, tgt, hard = collate(b, pad_id, scorer.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scorer.device.startswith("cuda")):
            z = scorer.train_forward(ids, mask, last, lab, lab_mask)
            zl = a4_probs_logits(z.float(), lab_mask)
        for j, e in enumerate(b):
            d = by_type.setdefault(e.qtype, {"logits": [], "labels": []})
            d["logits"].append(zl[j, : len(e.label_ids)].float().cpu().numpy().tolist())
            d["labels"].append(e.target_idx)
    report = {}
    for qt, d in by_type.items():
        groups: dict[int, tuple[list, list]] = {}
        for zz, y in zip(d["logits"], d["labels"]):
            g = groups.setdefault(len(zz), ([], []))
            g[0].append(zz)
            g[1].append(y)
        T = float(
            np.mean(
                [fit_temperature(np.array(zl), np.array(yl)) for zl, yl in groups.values() if len(yl) >= 20]
                or [1.0]
            )
        )
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides (yaml-parsed)")
    a = ap.parse_args()
    ov = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)}
    main(a.config, ov)
