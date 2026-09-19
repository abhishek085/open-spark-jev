"""Load a saved A0/A2/A3/A4 variant (experimental/variants.py) behind the same ``decide`` interface as
MenuScorer, so the gateway/UI and eval/external.py can serve and score it like any checkpoint.

A checkpoint directory is a variant if it contains ``variant.json`` (written by variants.py).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..model import MenuScorer
from ..prompting import render_prefix, render_prompt
from ..schema import Answer, Question, State
from .variants import QTYPES, VariantModel, a4_probs_logits


def is_variant(path: str) -> bool:
    return os.path.exists(os.path.join(path, "variant.json"))


class VariantScorer:
    def __init__(self, path: str, device: str | None = None):
        from peft import PeftModel

        with open(os.path.join(path, "variant.json")) as f:
            self.meta = json.load(f)
        self.variant = self.meta["variant"]
        self.temperature = self.meta.get("temperature", {})
        base = MenuScorer(self.meta["init"], use_state_cache=False, attn_implementation="sdpa", device=device)
        base.lm = PeftModel.from_pretrained(base.lm, os.path.join(path, "lora"))
        base.lm.eval()
        self.model = VariantModel(base, self.variant)
        head = os.path.join(path, "head.pt")
        if self.model.head is not None:
            self.model.head.load_state_dict(torch.load(head, map_location=base.device))
            self.model.head.eval()
        self.base = base
        self.tokenizer = base.tokenizer
        self.name = f"{os.path.basename(path)} ({self.variant})"

    @torch.no_grad()
    def decide(self, state: State, questions: list[Question], temperature: float | None = None, return_logits: bool = False) -> list[Answer]:
        b = self.base
        dev = b.device
        pre = self.tokenizer.encode(render_prefix(state, max_chars=b.max_state_chars), add_special_tokens=False)
        out = []
        for q in questions:
            ids = self.tokenizer.encode(render_prompt(state, q, max_chars=b.max_state_chars), add_special_tokens=False)
            n = 0
            for x, y in zip(pre, ids):
                if x != y:
                    break
                n += 1
            k = len(q.labels)
            ids_t = torch.tensor([ids], device=dev)
            mask = torch.ones_like(ids_t)
            last = torch.tensor([len(ids) - 1], device=dev)
            lab = torch.tensor([b.labels.ids_for(k)], device=dev)
            lab_mask = torch.ones((1, k), dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                z = self.model.logits(ids_t, mask, last, lab, lab_mask, torch.tensor([n], device=dev), torch.tensor([QTYPES.index(q.type)], device=dev))
            z = z.float()
            if self.variant == "a4":
                z = a4_probs_logits(z, lab_mask)
            z = z[0].cpu().numpy()
            t = temperature if temperature is not None else float(self.temperature.get(q.type, 1.0))
            p = np.exp((z - z.max()) / t)
            p /= p.sum()
            out.append(Answer.from_probs(q, p.tolist(), raw_logits=z.tolist() if return_logits else None))
        return out
