"""Publish the llama.cpp / Ollama builds as NEW Hugging Face repos. Never modifies an existing repo.

Token is read from the HF_TOKEN env var only; never logged or written to disk. Every target repo must not exist yet: if any
does, nothing is uploaded (the check runs for the whole group before the first upload).

  HF_TOKEN=... python scripts/publish_local_runtimes.py --group v6-gguf
  HF_TOKEN=... python scripts/publish_local_runtimes.py --group v8          # bf16 safetensors + GGUF (reverses "bf16 not published")
  python scripts/publish_local_runtimes.py --group v6-gguf --dry-run       # list what would be uploaded, no token needed
"""
import argparse
import os
import sys

CARDS = "release/local-runtimes/cards"
CKPT_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "calibration.json",
              "generation_config.json", "chat_template.jinja"]


def gguf_repo(release: str) -> dict:
    return {
        "repo": f"abhishek085/spark-s1-4b-{release}-GGUF",
        "card": f"{CARDS}/spark-s1-4b-{release}-GGUF.md",
        # No Q4_K_M: Ollama / Docker Model Runner default to it and it breaks calibration (see the card).
        "files": {f"spark-s1-4b-{release}-{q}.gguf": f"gguf_out/spark-s1-4b-{release}-{q}.gguf" for q in ("Q8_0", "Q6_K", "bf16")}
        | {"calibration.json": f"checkpoints/{release}-4b/calibration.json"},
    }


GROUPS = {
    "v6-gguf": [gguf_repo("v6")],
    # bf16 first so the GGUF repo's base_model link resolves.
    "v8": [{"repo": "abhishek085/spark-s1-4b-v8", "card": f"{CARDS}/spark-s1-4b-v8.md",
            "files": {f: f"checkpoints/v8-4b/{f}" for f in CKPT_FILES}},
           gguf_repo("v8")],
}

ap = argparse.ArgumentParser()
ap.add_argument("--group", choices=sorted(GROUPS), required=True)
ap.add_argument("--private", action="store_true", help="create the repos private (flip to public on the Hub after a look)")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

targets = GROUPS[a.group]
for t in targets:
    missing = [src for src in [t["card"], *t["files"].values()] if not os.path.exists(src)]
    if missing:
        sys.exit(f"{t['repo']}: missing local files {missing}")
    print(f"{t['repo']}  (card {t['card']})")
    for dst, src in t["files"].items():
        print(f"  {dst:40s} <- {src} ({os.path.getsize(src) / 1e9:.2f} GB)")
if a.dry_run:
    sys.exit(0)

from huggingface_hub import HfApi  # noqa: E402

api = HfApi(token=os.environ["HF_TOKEN"])
existing = [t["repo"] for t in targets if api.repo_exists(t["repo"], repo_type="model")]
if existing:
    sys.exit(f"refusing: these repos already exist and this script never modifies existing repos: {existing}")

for t in targets:
    api.create_repo(t["repo"], repo_type="model", private=a.private, exist_ok=False)
    for dst, src in t["files"].items():
        print("uploading", t["repo"], dst)
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=t["repo"], repo_type="model")
    api.upload_file(path_or_fileobj=t["card"], path_in_repo="README.md", repo_id=t["repo"], repo_type="model")
    info = api.repo_info(t["repo"], repo_type="model")
    print(f"done: https://huggingface.co/{t['repo']}  (private={info.private})")
