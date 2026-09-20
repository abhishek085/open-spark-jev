"""Dry-run generation over all harness packs (fake LLM: plumbing only)."""
import subprocess
import sys

packs = ["next_action_router", "tool_action_gate", "retrieval_gate", "termination_gate", "answer_sufficiency", "prompt_injection_gate"]
args = [a for p in packs for a in ("--task-pack", f"harness_{p}_v1")]
sys.exit(subprocess.call([sys.executable, "-m", "os_datagen.cli", "generate", *args, "--count", "60", "--seed", "42", "--dry-run", "--out", "artifacts/harness_dry"]))
