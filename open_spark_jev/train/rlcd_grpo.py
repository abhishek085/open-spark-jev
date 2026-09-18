"""Ablation baseline: sampled single-token GRPO with TRL.

The policy emits ONE token after "Answer:" (the label letter). GRPO samples G completions per
prompt, scores each with the reward functions below, and applies group-relative advantages.
This is the "standard RL" comparison point for the exact menu policy gradient in rlcd.py.

Expected finding (docs/RESEARCH.md, Exp. 3): 0/1 correctness rewards on sampled actions
drive the policy toward argmax collapse (ECE rises even as accuracy rises) unless the reward
also includes a proper scoring rule evaluated on the *distribution*, which sampled-action RL
cannot see. That is the argument for the exact formulation.

Requires ``trl>=0.24``. Uses TRL directly (not Unsloth): Unsloth's RL patch monkey-patches
GRPOTrainer at import time and is not needed for a 1.7B model on a 121 GB Spark.
"""

from __future__ import annotations

import argparse
import logging

import yaml

from ..data.corpus import read_jsonl
from ..prompting import LABEL_CHARS, render_prompt

log = logging.getLogger("osj.grpo")


def main(cfg_path: str) -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    recs = []
    for p in cfg["data"]:
        recs.extend(r for r in read_jsonl(p) if not r.meta.get("suspect"))
    rows = []
    for r in recs:
        q = r.question_obj()
        rows.append({"prompt": render_prompt(r.state_obj(), q), "target": LABEL_CHARS[r.label_index()],
                     "labels": q.labels, "safe": cfg.get("safe_labels", {}).get(r.domain, "")})
    ds = Dataset.from_list(rows)

    def correctness(completions, target, **kw):
        return [1.0 if c.strip()[:1] == t else 0.0 for c, t in zip(completions, target)]

    def conservative(completions, target, labels, safe, **kw):
        out = []
        for c, t, labs, s in zip(completions, target, labels, safe):
            if not s or s not in labs:
                out.append(0.0)
                continue
            si = LABEL_CHARS[labs.index(s)]
            got = c.strip()[:1]
            out.append(-1.5 if (t == si and got != si) else (-0.5 if (got == si and t != si) else 0.0))
        return out

    import inspect

    wanted = dict(
        output_dir=cfg["output_dir"],
        learning_rate=cfg.get("lr", 5e-6),
        per_device_train_batch_size=cfg.get("batch_size", 8),
        gradient_accumulation_steps=cfg.get("grad_accum", 2),
        num_generations=cfg.get("group_size", 8),
        max_completion_length=1,
        max_prompt_length=cfg.get("max_len", 2048),
        beta=cfg.get("kl_beta", 0.05),
        temperature=1.0,
        num_train_epochs=cfg.get("epochs", 1),
        bf16=True,
        logging_steps=10,
        report_to=[],
    )
    # TRL renames/removes GRPOConfig fields between releases (e.g. max_prompt_length); keep only
    # the ones this version accepts so the script stays runnable across TRL >= 0.24.
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    args = GRPOConfig(**{k: v for k, v in wanted.items() if k in accepted})
    trainer = GRPOTrainer(model=cfg["policy_init"], reward_funcs=[correctness, conservative], args=args, train_dataset=ds)
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
