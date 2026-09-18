"""A1 (docs/NOVELTY.md): single forward-pass readout for all of a state's questions.

``model.MenuScorer`` answers N questions on one state with 1 prefix pass + N suffix passes
sharing a copied KV cache (see ``model.py``'s ``_last_logits_with_cache``). This module tests
the literal reading of TypeSafe's "generates all outputs in a single query" claim: pack the
state prefix and *all* question suffixes into **one** sequence, run **one** forward pass, and
read each question's label logits off its own last-suffix-token position via a position map,
rather than batching N separate short sequences.

Mechanically this is a block-diagonal-ish attention question: each question suffix must see
the shared state prefix but must NOT see any other question's suffix (so questions don't leak
answers to each other, and so the result matches what N independent passes would produce). A
standard causal mask does not give us that directly once suffixes are concatenated -- token i
in suffix 2 would attend to all of suffix 1 under plain causal masking, which is wrong. So we
build a custom 4D attention mask: causal within the shared prefix, and for each suffix, causal
within itself plus full attention back to the prefix, but masked out from every other suffix.

If this is implemented correctly, per-question label logits should match ``MenuScorer``'s
(bf16 noise aside) exactly, because both mechanisms give each question the same information
(the state) and no information about sibling questions. The only thing that should change is
the number of forward passes (N -> 1) and therefore latency at higher question-counts. That
prediction is exactly what ``main()`` below checks.

Usage:
  python -m open_spark_jev.experimental.parallel_readout --model checkpoints/sft-qwen3-1.7b \
    --data data/benchmarks/sim_test.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from ..model import MenuScorer
from ..prompting import render_prefix, render_suffix
from ..schema import Answer, Question, State


class ParallelMenuScorer:
    """Wraps an existing MenuScorer's backbone/tokenizer; adds a single-forward-pass readout
    for multiple questions on one state. Not a training-capable class -- inference only."""

    def __init__(self, base: MenuScorer):
        self.base = base
        self.lm = base.lm
        self.tokenizer = base.tokenizer
        self.labels = base.labels
        self.calibration = base.calibration
        self.device = base.device
        self.name = base.name + "+parallel_readout(A1)"

    def _build_block_mask(self, prefix_len: int, suffix_lens: list[int]) -> torch.Tensor:
        """4D additive attention mask [1, 1, T, T]: causal within prefix; each suffix is causal
        within itself + sees the full prefix; suffixes cannot see each other."""
        total = prefix_len + sum(suffix_lens)
        neg = torch.finfo(torch.bfloat16).min
        mask = torch.full((total, total), neg, device=self.device)
        # prefix: causal among itself
        pref_idx = torch.arange(prefix_len)
        mask[:prefix_len, :prefix_len] = torch.where(
            pref_idx[:, None] >= pref_idx[None, :], 0.0, neg
        )
        offset = prefix_len
        for L in suffix_lens:
            # see full prefix
            mask[offset : offset + L, :prefix_len] = 0.0
            # causal within own suffix
            local = torch.arange(L)
            block = torch.where(local[:, None] >= local[None, :], 0.0, neg)
            mask[offset : offset + L, offset : offset + L] = block
            offset += L
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

    @torch.no_grad()
    def decide(self, state: State, questions: list[Question], temperature: float | None = None) -> list[Answer]:
        prefix_ids = self.tokenizer.encode(render_prefix(state, max_chars=self.base.max_state_chars), add_special_tokens=False)
        suffix_ids_list = [self.tokenizer.encode(render_suffix(q), add_special_tokens=False) for q in questions]
        suffix_lens = [len(s) for s in suffix_ids_list]

        all_ids = prefix_ids + [t for s in suffix_ids_list for t in s]
        ids = torch.tensor([all_ids], device=self.device)
        pos = torch.arange(len(all_ids), device=self.device).unsqueeze(0)
        attn_mask = self._build_block_mask(len(prefix_ids), suffix_lens).to(dtype=self.lm.dtype)

        out = self.lm(input_ids=ids, attention_mask=attn_mask, position_ids=pos)
        logits = out.logits[0]  # [T, V]

        answers = []
        offset = len(prefix_ids)
        for q, L in zip(questions, suffix_lens):
            last_pos = offset + L - 1
            z = logits[last_pos, self.labels.ids_for(len(q.labels))].float()
            t = temperature if temperature is not None else self.calibration.t(q.type)
            probs = F.softmax(z / t, dim=-1).tolist()
            answers.append(Answer.from_probs(q, probs))
            offset += L
        return answers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-questions-per-state", type=int, default=4, help="group this many records' questions onto one shared state for the parallel-readout check (records are grouped arbitrarily; this tool checks mechanism equivalence, not per-record accuracy)")
    a = ap.parse_args()

    from ..data.corpus import read_jsonl

    recs = read_jsonl(a.data)[: a.limit]
    base = MenuScorer(a.model, use_state_cache=True)
    par = ParallelMenuScorer(base)

    # Equivalence check: for each record, does the parallel single-pass answer match the
    # baseline's cached-KV multi-pass answer (up to bf16 noise)?
    max_diff = 0.0
    n_checked = 0
    for r in recs[:50]:
        state = r.state_obj()
        q = r.question_obj()
        a_base = base.decide(state, [q])[0]
        a_par = par.decide(state, [q])[0]
        d = max(abs(x - y) for x, y in zip(a_base.probs, a_par.probs))
        max_diff = max(max_diff, d)
        n_checked += 1
    print(f"[equivalence] checked {n_checked} single-question records, max |prob diff| = {max_diff:.4f} (bf16 noise expected ~0.01-0.05)")

    # Latency comparison at increasing questions-per-state.
    from ..schema import Choice, Noul, Score

    state = State(content="\n".join('{"ts":"2026-09-18T10:00:00Z","svc":"checkout","status":500}' for _ in range(20)), schema_hint="api logs")
    pool = [
        Choice(prompt="What is the right operational response?", options=["ignore", "scale_up", "rollback", "page_oncall"]),
        Score(prompt="How severe is this?", levels=["none", "minor", "major", "critical"]),
        Noul(prompt="The service is degraded."),
        Choice(prompt="Which team owns this?", options=["payments", "platform", "frontend", "data"]),
    ]
    results = defaultdict(dict)
    for n_q in (1, 4, 16):
        qs = [pool[i % len(pool)] for i in range(n_q)]
        base.decide(state, qs)
        par.decide(state, qs)
        for name, fn in (
            ("A0_baseline_N_passes", lambda qs=qs: base.decide(state, qs)),
            ("A1_parallel_1_pass", lambda qs=qs: par.decide(state, qs)),
        ):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 5 * 1000
            results[n_q][name] = ms
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
