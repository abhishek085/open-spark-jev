"""Dry-run generation over all foundation packs (fake LLM: plumbing only). Real runs: see docs/examples.md."""
import subprocess
import sys

packs = [p for p in ("semantic_entailment", "document_type", "document_relevance", "extraction_validation", "rule_application",
                     "temporal_reasoning", "communication_intent", "communication_urgency", "authorization_gate")]
args = [a for p in packs for a in ("--task-pack", f"foundation_{p}_v1")]
sys.exit(subprocess.call([sys.executable, "-m", "os_datagen.cli", "generate", *args, "--count", "60", "--seed", "42", "--dry-run", "--out", "artifacts/foundation_dry"]))
