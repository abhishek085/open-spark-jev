"""NVFP4 post-training quantization of a spark-s1 checkpoint with NVIDIA ModelOpt (run inside the TRT-LLM container, which ships modelopt).

Calibrates on real decision prompts (rendered exactly as at inference), keeps the LM head in bf16 (the option-letter readout reads its logits),
and exports a Hugging Face checkpoint with NVFP4 weights that vLLM / TensorRT-LLM load directly.

  python scripts/quant/ptq_nvfp4.py --model /work/checkpoints/v3-1.7b --calib /work/runs/quant/calib_prompts.jsonl --out /work/engines/spark-s1-1.7b-v3-nvfp4
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tokenizer", default=None, help="original backbone tokenizer dir (the checkpoint tokenizer is saved in a newer transformers format)")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--cfg", default="NVFP4_DEFAULT_CFG")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
texts = [json.loads(line)["text"] for line in open(a.calib)][: a.n]


def forward_loop(m):
    with torch.no_grad():
        for i, t in enumerate(texts):
            ids = tok(t, return_tensors="pt", truncation=True, max_length=a.max_len).to("cuda")
            m(**ids)
            if i % 100 == 0:
                print("calibrated", i, flush=True)


cfg = getattr(mtq, a.cfg)
t0 = time.time()
model = mtq.quantize(model, cfg, forward_loop)
print(f"quantized in {time.time() - t0:.0f}s", flush=True)
mtq.print_quant_summary(model)
export_hf_checkpoint(model, export_dir=a.out)
tok.save_pretrained(a.out)
print("exported", a.out, flush=True)
