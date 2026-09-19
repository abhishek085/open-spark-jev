"""A5 (docs/NOVELTY.md): domain-routed LoRA adapters behind a tiny router.

Four LoRA adapters (one per domain group) share a frozen Qwen3 backbone. Each adapter trains
only on its group's examples (so the total training budget matches the single-adapter A0 run),
and a linear router over the frozen base's mean-pooled hidden state picks the adapter at
inference - no domain tag needed. Reports routing accuracy, routed metrics, and oracle-routed
metrics (true group) so router error and adapter quality can be separated.

Usage: python -m open_spark_jev.experimental.a5_routed --out runs/variants/a5
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
import torch.nn.functional as F

from ..data.corpus import read_jsonl, split_records
from ..model import MenuScorer
from ..train.sft import build_examples, collate, menu_loss
from .variants import VariantModel, make_batch, report

log = logging.getLogger("osj.a5")

GROUPS = {
    0: ["routing", "email_routing", "urgency_triage", "ecommerce_support_intent", "calendar_conflict",
        "form_completion", "moderation", "feature_flag"],
    1: ["security", "risk", "content_trust", "secret_detection", "api_trace"],
    2: ["incident", "ops_incident_routing", "test_failure_triage", "build_log_classification",
        "data_quality_action", "schema_change_impact", "sql_safety", "patch_acceptance"],
    3: ["retrieval_decision", "doc_type_classification", "extraction_correctness", "rag_chunk_relevance",
        "rag_evidence_sufficiency", "query_rewrite_needed", "search_result_quality", "task_decomposition",
        "claim_support", "game"],
}
DOMAIN_GROUP = {d: g for g, ds in GROUPS.items() for d in ds}


def group_of(e) -> int:
    return DOMAIN_GROUP.get(e.domain, 3)


@torch.no_grad()
def pooled_features(base: MenuScorer, examples, bs, pad_id) -> torch.Tensor:
    feats = []
    base.lm.eval()
    for i in range(0, len(examples), bs):
        b = examples[i : i + bs]
        ids, mask, *_ = collate(b, pad_id, base.device)
        with torch.autocast("cuda", dtype=torch.bfloat16), base.lm.disable_adapter():
            h = base.lm(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1]
        m = mask.unsqueeze(-1).float()
        feats.append(((h.float() * m).sum(1) / m.sum(1)).cpu())
    return torch.cat(feats)


@torch.no_grad()
def routed_logits(model: VariantModel, examples, groups, bs, pad_id):
    """Logits with each example scored by its assigned group's adapter (batched per group)."""
    model.eval()
    out = [None] * len(examples)
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        model.base.lm.set_adapter(f"g{g}")
        for s in range(0, len(idx), bs):
            sub = idx[s : s + bs]
            b = [examples[i] for i in sub]
            ids, mask, last, lab, lab_mask, tgt, hard, pl, qi = make_batch(b, pad_id, model.base.device, {id(e): 0 for e in b})
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.logits(ids, mask, last, lab, lab_mask, pl, qi)
            for j, i in enumerate(sub):
                out[i] = z[j, : len(examples[i].label_ids)].float().cpu().numpy()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="models/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    base = MenuScorer(a.init, use_state_cache=False, attn_implementation="sdpa")
    pad_id = base.tokenizer.pad_token_id or 0
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    base.lm = get_peft_model(base.lm, cfg, adapter_name="g0")
    for g in (1, 2, 3):
        base.lm.add_adapter(f"g{g}", cfg)
    base.lm.gradient_checkpointing_enable()
    base.lm.enable_input_require_grads()
    base.lm.config.use_cache = False
    model = VariantModel(base, "a0")

    recs = read_jsonl("data/synthetic/sim_train.jsonl") + read_jsonl("data/synthetic/teacher_train_split.jsonl")
    random.Random(a.seed).shuffle(recs)
    recs = recs[: 300 if a.smoke else a.n_train]
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
    all_params = [p for g in range(4) for n, p in base.lm.named_parameters() if f".g{g}." in n]
    for p in all_params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(all_params, lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    n_batches = sum(math.ceil(sum(1 for e in train_ex if group_of(e) == g) / a.bs) for g in range(4))
    steps_total = math.ceil(n_batches / a.accum)
    warm = max(1, int(0.05 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))

    batches = []
    for g in range(4):
        ex = [e for e in train_ex if group_of(e) == g]
        ex.sort(key=lambda e: len(e.input_ids) // 64 + random.random())
        batches += [(g, ex[i : i + a.bs]) for i in range(0, len(ex), a.bs)]
    random.shuffle(batches)
    log.info("a5: %d train examples, group sizes %s, %d batches", len(train_ex), [sum(1 for e in train_ex if group_of(e) == g) for g in range(4)], len(batches))

    t0, step = time.time(), 0
    model.train()
    for bi, (g, b) in enumerate(batches):
        base.lm.set_adapter(f"g{g}")
        ids, mask, last, lab, lab_mask, tgt, hard, pl, qi = make_batch(b, pad_id, base.device, {id(e): 0 for e in b})
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model.logits(ids, mask, last, lab, lab_mask, pl, qi)
            loss = menu_loss(z.float(), tgt, lab_mask, 0.5) / a.accum
        loss.backward()
        if (bi + 1) % a.accum == 0 or bi == len(batches) - 1:
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                log.info("step %d/%d loss %.4f %.0fs", step, steps_total, loss.item() * a.accum, time.time() - t0)
    train_seconds = time.time() - t0

    log.info("training router on frozen-base pooled features")
    Xtr = pooled_features(base, train_ex, a.bs, pad_id)
    ytr = torch.tensor([group_of(e) for e in train_ex])
    router = torch.nn.Linear(Xtr.shape[1], 4)
    ropt = torch.optim.Adam(router.parameters(), lr=1e-3)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    for _ in range(300):
        ropt.zero_grad()
        F.cross_entropy(router((Xtr - mu) / sd), ytr).backward()
        ropt.step()

    def route(ex):
        X = pooled_features(base, ex, a.bs, pad_id)
        return router((X - mu) / sd).argmax(-1).tolist()

    val_groups = route(val_ex)
    val_logits = routed_logits(model, val_ex, val_groups, a.bs, pad_id)
    result = {"variant": "a5", "train_seconds": round(train_seconds), "n_train": len(train_ex),
              "trainable_params": sum(p.numel() for p in all_params) + sum(p.numel() for p in router.parameters())}
    for name, ex in tests.items():
        rg = route(ex)
        og = [group_of(e) for e in ex]
        result[name] = report(val_logits, val_ex, routed_logits(model, ex, rg, a.bs, pad_id), ex)
        result[name]["routing_accuracy"] = round(float(np.mean([r == o for r, o in zip(rg, og)])), 4)
        result[name]["oracle_routed_overall"] = report(val_logits, val_ex, routed_logits(model, ex, og, a.bs, pad_id), ex)["overall"]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: (v["overall"] if isinstance(v, dict) and "overall" in v else v) for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
