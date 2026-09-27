"""control-jev-es: a ModernBERT (encoder-only, non-autoregressive) decision scorer.

Borrowed from Laya (convaiinnovations/laya, Apache 2.0): a bidirectional encoder reads the state and
every question's options in one pass, and each option is scored at its own dedicated ``[MASK]`` token
instead of at a letter slot the way MenuScorer reads a decoder. That readout is naturally suited to an
encoder (bidirectional context helps score a masked position; a decoder can't see past it) and needs no
LM head: option text is never generated, so there's no letter-token restriction the way MenuScorer's
menu head has, and no fixed 26-option ceiling.

Same ``decide(state, questions) -> list[Answer]`` interface as ``MenuScorer``, so gateway.py can serve
it behind the identical ``/v1/systemone`` wire format -- see ``is_control_es`` / the branch in
``serve/gateway.py``'s ``get_backend``. A checkpoint directory is control-jev-es if it has
``control_es.json`` (mirrors ``variant_scorer.py``'s ``variant.json`` convention).

The head here is a single linear layer on the backbone's last hidden state at each mask position
(``score = W @ h_mask``), fully fine-tuned together with the backbone -- Laya's own head additionally
runs the option-marker positions through two extra from-scratch transformer layers before scoring;
that is a natural v2 refinement once this simpler head's numbers are in.
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from torch import nn

from ..model import Calibration
from ..schema import Answer, Question, State


def is_control_es(path: str) -> bool:
    return os.path.exists(os.path.join(path, "control_es.json"))


def render_es_prompt(state: State, q: Question, mask_token: str) -> str:
    content = state.content if isinstance(state.content, str) else json.dumps(state.content, ensure_ascii=False)
    lines = [f"Context: {content}", f"Question: {q.prompt}", "Options:"]
    for label in q.labels:
        lines.append(f"- {label}: {mask_token}")
    return "\n".join(lines)


class ScoreHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, mask_hidden: torch.Tensor) -> torch.Tensor:  # [..., H] -> [...]
        return self.proj(mask_hidden).squeeze(-1)


class ControlESScorer(nn.Module):
    def __init__(self, path: str, device: str | None = None, dtype: torch.dtype = torch.float32, max_len: int = 512):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.name = path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.backbone = AutoModel.from_pretrained(path, dtype=dtype).to(device).eval()
        self.head = ScoreHead(self.backbone.config.hidden_size).to(device, dtype)
        head_path = os.path.join(path, "score_head.pt")
        if os.path.exists(head_path):
            self.head.load_state_dict(torch.load(head_path, map_location=device))
        self.head.eval()
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        cal_path = os.path.join(path, "calibration.json")
        self.calibration = Calibration.load(cal_path) if os.path.exists(cal_path) else Calibration()

    def encode_batch(self, texts: list[str]):
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(self.device)
        out = self.backbone(**enc).last_hidden_state  # [B, T, H]
        return enc, out

    @torch.no_grad()
    def decide(self, state: State, questions: list[Question], temperature: float | None = None, return_logits: bool = False) -> list[Answer]:
        out = []
        for q in questions:
            k = len(q.labels)
            text = render_es_prompt(state, q, self.mask_token)
            enc, hidden = self.encode_batch([text])
            mask_pos = (enc["input_ids"][0] == self.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_pos) < k:
                raise ValueError(f"{q.id}: expected {k} mask tokens, found {len(mask_pos)} (state truncated at max_len={self.max_len}?)")
            mh = hidden[0, mask_pos[:k]]  # [K, H]
            z = self.head(mh).float()  # [K]
            t = temperature if temperature is not None else self.calibration.t(q.type)
            probs = F.softmax(z / t, dim=-1)
            out.append(Answer.from_probs(q, probs.tolist(), raw_logits=z.tolist() if return_logits else None))
        return out
