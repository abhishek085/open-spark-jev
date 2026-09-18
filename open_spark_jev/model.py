"""Qwen3 menu scorer (Hugging Face backend).

``MenuScorer`` wraps a causal LM and exposes one operation:

    decide(state, questions) -> list[Answer]

Mechanics
---------
* The state prefix (system prompt + fenced state) is tokenised and run **once**; its KV cache
  is kept.
* Each question suffix is appended to a copy of that cache and run in a single batched
  forward pass. Suffixes are right-padded; we read the logits at each suffix's last real
  token (the position right after ``"Answer:"``).
* The full-vocab logits at that position are gathered at the label token ids
  (``" A"``, ``" B"``, ...) - only as many as the question has labels - divided by the
  calibrated temperature, and softmaxed.

There is **no extra parameter** in the default configuration: the "menu head" is the
backbone's own LM head restricted to a handful of rows. That is what lets a TensorRT-LLM
engine built from the plain HF checkpoint serve identical decisions (see ``serve/gateway.py``,
which reproduces this with ``/v1/completions`` + ``logprobs``).

Optional ``aux_head``: a learned linear head on the last hidden state that produces
``MAX_OPTIONS`` slot logits. Useful as an ablation ("does a dedicated head calibrate better
than the LM head?") but it is *not* servable by trtllm-serve; keep it off for deployment.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .prompting import LabelSpace, render_prefix, render_suffix
from .schema import MAX_OPTIONS, Answer, Question, State

log = logging.getLogger(__name__)


@dataclass
class Calibration:
    """Per-question-type temperatures learned on a held-out split (see train/sft.py)."""

    temperature: dict[str, float] = field(default_factory=lambda: {"choice": 1.0, "score": 1.0, "noul": 1.0})

    @classmethod
    def load(cls, path: str) -> Calibration:
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature}, f, indent=2)

    def t(self, qtype: str) -> float:
        return float(self.temperature.get(qtype, 1.0))


def gather_label_logits(
    last_logits: torch.Tensor,  # [B, V]
    label_ids: torch.Tensor,  # [B, K] padded with any valid id where mask==0
    label_mask: torch.Tensor,  # [B, K] bool
) -> torch.Tensor:
    """Restrict full-vocab logits to each row's label ids. Masked slots -> -inf."""
    g = torch.gather(last_logits, 1, label_ids)
    return g.masked_fill(~label_mask, float("-inf"))


class MenuScorer(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
        calibration: Calibration | None = None,
        use_state_cache: bool = True,
        aux_head: bool = False,
        attn_implementation: str | None = None,
        max_state_chars: int = 12_000,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        kw = {}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        self.lm = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=dtype, **kw).to(self.device)
        self.lm.eval()
        self.labels = LabelSpace.build(self.tokenizer)
        self.calibration = calibration or Calibration()
        self.use_state_cache = use_state_cache
        self.max_state_chars = max_state_chars
        self.aux_head: nn.Linear | None = None
        if aux_head:
            hidden = self.lm.config.hidden_size
            self.aux_head = nn.Linear(hidden, MAX_OPTIONS).to(self.device, dtype)
        cal_path = os.path.join(model_name_or_path, "calibration.json")
        if calibration is None and os.path.exists(cal_path):
            self.calibration = Calibration.load(cal_path)
            log.info("loaded calibration %s", self.calibration.temperature)

    # ------------------------------------------------------------------ tokenisation

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _pad_right(self, seqs: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        last = torch.zeros(len(seqs), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
            last[i] = len(s) - 1
        return ids.to(self.device), mask.to(self.device), last.to(self.device)

    # ------------------------------------------------------------------ core forward

    @torch.no_grad()
    def _last_logits_with_cache(self, prefix_ids: list[int], suffixes: list[list[int]]) -> torch.Tensor:
        """Run the prefix once, then all suffixes batched on top of its cache. Returns [B, V]."""
        from transformers import DynamicCache

        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        P = len(prefix_ids)
        B = len(suffixes)
        prefix = torch.tensor([prefix_ids], device=self.device)
        out = self.lm(input_ids=prefix, use_cache=True)
        cache = out.past_key_values
        # Expand the single-row cache to B rows.
        if hasattr(cache, "batch_repeat_interleave"):
            cache.batch_repeat_interleave(B)
        else:  # pragma: no cover - older transformers
            cache = DynamicCache.from_legacy_cache(
                tuple(tuple(t.repeat(B, 1, 1, 1) for t in layer) for layer in cache.to_legacy_cache())
            )
        suf_ids, suf_mask, last = self._pad_right(suffixes, pad)
        attn = torch.cat([torch.ones((B, P), dtype=torch.long, device=self.device), suf_mask], dim=1)
        pos = (P + torch.arange(suf_ids.shape[1], device=self.device)).unsqueeze(0).expand(B, -1)
        out = self.lm(
            input_ids=suf_ids,
            attention_mask=attn,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        logits = out.logits  # [B, S, V]
        return logits[torch.arange(B, device=self.device), last]

    @torch.no_grad()
    def _last_logits_full(self, prefix_ids: list[int], suffixes: list[list[int]]) -> torch.Tensor:
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        seqs = [prefix_ids + s for s in suffixes]
        ids, mask, last = self._pad_right(seqs, pad)
        out = self.lm(input_ids=ids, attention_mask=mask)
        return out.logits[torch.arange(len(seqs), device=self.device), last]

    def label_logits(self, state: State, questions: Sequence[Question]) -> tuple[list[torch.Tensor], int]:
        """Uncalibrated label logits per question (list of 1-D tensors) + prefix token count."""
        prefix_ids = self._encode(render_prefix(state, max_chars=self.max_state_chars))
        suffixes = [self._encode(render_suffix(q)) for q in questions]
        if self.use_state_cache:
            try:
                last = self._last_logits_with_cache(prefix_ids, suffixes)
            except Exception as e:  # fall back rather than fail a decision
                log.warning("state-cache path failed (%s); falling back to full forward", e)
                last = self._last_logits_full(prefix_ids, suffixes)
        else:
            last = self._last_logits_full(prefix_ids, suffixes)
        outs = []
        for i, q in enumerate(questions):
            ids = torch.tensor(self.labels.ids_for(len(q.labels)), device=self.device)
            outs.append(last[i, ids].float())
        return outs, len(prefix_ids)

    # ------------------------------------------------------------------ public API

    @torch.no_grad()
    def decide(
        self,
        state: State,
        questions: Sequence[Question],
        temperature: float | None = None,
        return_logits: bool = False,
    ) -> list[Answer]:
        logits, _ = self.label_logits(state, questions)
        answers = []
        for q, z in zip(questions, logits):
            t = temperature if temperature is not None else self.calibration.t(q.type)
            probs = F.softmax(z / t, dim=-1).tolist()
            answers.append(Answer.from_probs(q, probs, raw_logits=z.tolist() if return_logits else None))
        return answers

    @torch.no_grad()
    def decide_timed(self, state: State, questions: Sequence[Question], **kw) -> tuple[list[Answer], float, int]:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, n_prefix = self.label_logits(state, questions)
        answers = []
        for q, z in zip(questions, logits):
            t = kw.get("temperature") or self.calibration.t(q.type)
            answers.append(Answer.from_probs(q, F.softmax(z / t, dim=-1).tolist()))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return answers, (time.perf_counter() - t0) * 1000.0, n_prefix

    # ------------------------------------------------------------------ training helpers

    def train_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_index: torch.Tensor,
        label_ids: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable restricted label logits [B, K] for a padded batch of full prompts."""
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=self.aux_head is not None)
        B = input_ids.shape[0]
        rows = torch.arange(B, device=input_ids.device)
        if self.aux_head is not None:
            h = out.hidden_states[-1][rows, last_index]
            z = self.aux_head(h)[:, : label_ids.shape[1]]
            return z.float().masked_fill(~label_mask, float("-inf"))
        last = out.logits[rows, last_index]
        return gather_label_logits(last.float(), label_ids, label_mask)

    def save_pretrained(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.lm.save_pretrained(out_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(out_dir)
        self.calibration.save(os.path.join(out_dir, "calibration.json"))
        if self.aux_head is not None:
            torch.save(self.aux_head.state_dict(), os.path.join(out_dir, "aux_head.pt"))
