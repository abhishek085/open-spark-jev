"""RLDM-9: offline advantage-weighted regression (Peng et al. 2019) fine-tune on top of a
finished SFT checkpoint, using the weighted dataset built by experimental/awr_data.py.

No live rollout, no reward-model calls, no policy-gradient step -- every previous RL attempt in
this repo (RLCD-direct, RLCD-contrastive, GRPO) trained against live rollouts of the policy being
updated, and the one clean root cause the project's own diagnostics found for why RL kept failing
in production was reward saturation: once the SFT policy already fits its own training rows
(near-zero train loss), there's no gradient left for a live reward to exploit (docs/BENCHMARKS.md,
"v7 attempt 1", reward J=1.0 / KL=0.0 after 30 steps on already-memorized rows). AWR sidesteps
this entirely -- the "advantage" here isn't computed against the model currently being trained,
it's computed once, offline, from several *sibling* checkpoints' already-logged predictions on the
calibration split, so there's no analogous saturation loop to fall into.

Loss: weight * CE(current_policy(state), gold) -- ordinary cross-entropy, per-example reweighted
by awr_data.py's advantage-derived weight. This keeps "what counts as good" identical to the
project's existing Brier-based reward family; only the learning mechanism changes.

  python -m open_spark_jev.train.awr --config configs/train/awr_v1.yaml
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

import torch
import torch.nn.functional as F
import yaml

from ..data.corpus import Record
from ..model import MenuScorer
from ..train.sft import collate as sft_collate
from ..train.sft import evaluate

log = logging.getLogger("osj.awr")


@dataclass
class WeightedExample:
    input_ids: list[int]
    label_ids: list[int]
    target_idx: int
    target_dist: list[float] | None
    qtype: str
    domain: str
    weight: float
    record: Record | None = None


def load_weighted(scorer: MenuScorer, path: str, max_len: int) -> list[WeightedExample]:
    from ..prompting import render_prompt

    out = []
    for line in open(path):
        o = json.loads(line)
        r = Record(**o["record"])
        if r.meta.get("suspect"):
            continue
        q = r.question_obj()
        ids = scorer.tokenizer.encode(render_prompt(r.state_obj(), q), add_special_tokens=False)
        if len(ids) > max_len:
            continue
        out.append(
            WeightedExample(
                input_ids=ids,
                label_ids=scorer.labels.ids_for(len(q.labels)),
                target_idx=r.label_index(),
                target_dist=r.target_dist(),
                qtype=q.type,
                domain=r.domain,
                weight=float(o["weight"]),
                record=r,
            )
        )
    return out


def collate(batch: list[WeightedExample], pad_id: int, device: str):
    ids, mask, last, lab, lab_mask, tgt, hard = sft_collate(batch, pad_id, device)
    w = torch.tensor([e.weight for e in batch], device=device, dtype=torch.float32)
    return ids, mask, last, lab, lab_mask, tgt, hard, w


def awr_loss(
    z: torch.Tensor, tgt: torch.Tensor, lab_mask: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    logp = F.log_softmax(z, dim=-1)
    logp = logp.masked_fill(~lab_mask, 0.0)
    ce = -(tgt * logp).sum(-1)  # per-example CE (or KL-to-soft-target, same as menu_loss's CE term)
    return (weight * ce).mean()


def main(cfg_path: str, overrides: dict | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg.update(overrides)
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))

    scorer = MenuScorer(cfg["model"], use_state_cache=False)
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

    all_ex = load_weighted(scorer, cfg["data"], cfg.get("max_len", 2048))
    random.Random(cfg.get("seed", 0)).shuffle(all_ex)
    n_val = max(1, int(len(all_ex) * cfg.get("val_frac", 0.05)))
    val_ex, train_ex = all_ex[:n_val], all_ex[n_val:]
    log.info(
        "train %d  val %d weighted examples (weight range %.3f-%.3f)",
        len(train_ex),
        len(val_ex),
        min(e.weight for e in all_ex),
        max(e.weight for e in all_ex),
    )

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
            ids, mask, last, lab, lab_mask, tgt, hard, w = collate(b, pad_id, scorer.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scorer.device.startswith("cuda")):
                z = scorer.train_forward(ids, mask, last, lab, lab_mask)
                loss = awr_loss(z.float(), tgt, lab_mask, w) / accum
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

        scorer.lm.eval()
        rep = evaluate(scorer, val_ex, bs, pad_id)
        log.info("epoch %d val: %s", ep, json.dumps(rep, indent=None))
        with open(os.path.join(out_dir, f"val_epoch{ep}.json"), "w") as f:
            json.dump(rep, f, indent=2)

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
