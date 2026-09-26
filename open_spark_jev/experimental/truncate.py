"""A7 / v7.1: build a depth-truncated decision model by deleting the top transformer layers.

The depth probe (``depth_probe.py``) reads the option-letter logits off every layer through the same
tied ``norm`` + ``lm_head`` the model already uses. On v6-4b the answer stops changing well before the
top of the stack:

    own splits (600 rows)   layer 24: 0.8450   full 32 layers: 0.8517   (-0.7 pt)
    JevBench hard (111)     layer 24: 0.6216   full 32 layers: 0.6036   (+1.8 pt)

so layers 25-32 are eight layers of latency buying approximately nothing. Deleting them is exactly
what the probe measured -- ``norm(h_24) -> lm_head`` is what a 24-layer model computes at its last
position -- so the truncated model's accuracy should reproduce the probe's layer-24 row, and any
deviation is a bug rather than a surprise.

Why this is worth doing: Speed is the axis where decider-4b v2 beats us outright (13 ms vs our 75 ms),
it is a quarter of the JevBench Score, and unlike quantisation a depth cut compounds with NVFP4 --
25 % fewer layers and NVFP4 are multiplicative, not competing.

Truncation is a lossy edit, so it writes a NEW checkpoint and never touches the source. A short LoRA
re-adaptation on the truncated stack (the layers below were never trained to be final) is the obvious
follow-up and is deliberately left as a separate step.

  python -m open_spark_jev.experimental.truncate --model checkpoints/v6-4b --keep 24 --out checkpoints/v7.1-4b-L24
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--keep", type=int, required=True, help="number of bottom layers to keep")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    layers = model.model.layers
    n0 = len(layers)
    if not 0 < a.keep < n0:
        raise SystemExit(f"--keep must be in 1..{n0 - 1}, got {a.keep}")

    kept_kinds = list(model.config.layer_types[: a.keep]) if getattr(model.config, "layer_types", None) else None
    model.model.layers = torch.nn.ModuleList(list(layers)[: a.keep])
    model.config.num_hidden_layers = a.keep
    if kept_kinds is not None:
        model.config.layer_types = kept_kinds
    # layer_idx is how each attention/linear-attention layer finds its own cache slot
    for i, layer in enumerate(model.model.layers):
        for mod in layer.modules():
            if hasattr(mod, "layer_idx"):
                mod.layer_idx = i

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    for f in ("calibration.json", "chat_template.jinja"):
        src = os.path.join(a.model, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, f))

    kept_full = sum(1 for k in (kept_kinds or []) if k == "full_attention")
    meta = {"source": a.model, "layers_before": n0, "layers_after": a.keep,
            "full_attention_layers_kept": kept_full,
            "note": "Depth-truncated (A7). Calibration temperatures are inherited from the source "
                    "checkpoint and are NOT refitted; refit before relying on confidences."}
    json.dump(meta, open(os.path.join(a.out, "truncation.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
