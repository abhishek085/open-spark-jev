"""Publish a spark-s1 checkpoint (+ its model card) to a Hugging Face model repo.
Token is read from the HF_TOKEN env var only; never logged or written to disk.

  HF_TOKEN=... python scripts/publish_hf.py --checkpoint checkpoints/v5-4b --repo abhishek085/spark-s1-4b-v5 --card /path/to/README.md
"""
import argparse
import os

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True, help="local checkpoint dir (config.json, tokenizer*, model.safetensors, calibration.json)")
ap.add_argument("--repo", required=True, help="e.g. abhishek085/spark-s1-4b-v5")
ap.add_argument("--card", required=True, help="README.md to upload as the repo's model card")
ap.add_argument("--private", action="store_true")
a = ap.parse_args()

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)

# Same file set as the existing v3 releases: config, tokenizer, weights, calibration.json, generation_config, chat
# template. Internal training artifacts (val_epoch*.json) are excluded on purpose.
KEEP = {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
        "special_tokens_map.json", "model.safetensors", "calibration.json", "generation_config.json", "chat_template.jinja",
        "hf_quant_config.json"}
allow = [f for f in os.listdir(a.checkpoint) if f in KEEP]
print("uploading:", sorted(allow))
api.upload_folder(folder_path=a.checkpoint, repo_id=a.repo, repo_type="model", allow_patterns=allow)
api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md", repo_id=a.repo, repo_type="model")
info = api.repo_info(a.repo, repo_type="model")
print(f"done: https://huggingface.co/{a.repo}  (private={info.private})")
