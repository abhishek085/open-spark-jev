"""Extract the pure-text decoder out of a Qwen3.5 vision-language checkpoint.

Qwen ships Qwen3.5-0.8B only as a VL checkpoint (`Qwen3_5ForConditionalGeneration`, with
`model.language_model.*` the text decoder, `model.visual.*` the vision tower, `mtp.*` a speculative-
decoding head we don't use). Our menu-readout scorer wants a bare `Qwen3_5ForCausalLM`, the same shape
the 4B and 2B are natively released as (and the same shape Mapika/decider-0.8b's own config.json uses,
confirming this is the standard way to get a text-only 0.8B out of this release).

  python scripts/tools/extract_qwen35_text.py \\
      --src models/Qwen3.5-0.8B-Base-raw --out models/Qwen3.5-0.8B-Base
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    text_cfg = dict(cfg["text_config"])
    text_cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    text_cfg.setdefault("model_type", "qwen3_5_text")
    # we drop mtp.* below (the speculative-decoding head, unused by our menu readout); a converter
    # that still sees mtp_num_hidden_layers > 0 (llama.cpp's GGUF export in particular) expects an
    # extra trailing layer's tensors and fails with "tensor 'blk.N.attn_norm.weight' not found".
    text_cfg["mtp_num_hidden_layers"] = 0
    os.makedirs(a.out, exist_ok=True)
    json.dump(text_cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    shard_files = sorted(set(idx["weight_map"].values()))
    state: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        sd = load_file(os.path.join(a.src, shard))
        for k, v in sd.items():
            if k.startswith("model.language_model."):
                state[k.replace("model.language_model.", "model.", 1)] = v
    dropped = len(idx["weight_map"]) - len(state)
    print(f"kept {len(state)} text-decoder tensors, dropped {dropped} (vision tower + mtp head)")
    save_file(state, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})

    for f in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "special_tokens_map.json"):
        src = os.path.join(a.src, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, f))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
