# Release checklist

Copy this list into the release issue and tick it off for each release.

- [ ] Tag and release commit created
- [ ] Model revision and adapter hashes pinned (Hugging Face revision, `model.safetensors` sha256; see [MODEL_CARD.md](../MODEL_CARD.md#versioning-and-lineage))
- [ ] Dataset/benchmark revisions pinned
- [ ] Model card reviewed
- [ ] Known limitations reviewed
- [ ] Security policy reviewed ([SECURITY.md](../SECURITY.md))
- [ ] Tests/lint run (`pytest`, `ruff check .`)
- [ ] Fixture smoke benchmark run (`python examples/tool_gate.py`, Decision Lab fixtures)
- [ ] Hardware and inference configuration documented
- [ ] Release notes written ([CHANGELOG.md](../CHANGELOG.md))
- [ ] README demo verified on a clean environment (`pip install -e ".[serve]"` then `osj lab` in a fresh venv, no weights)
- [ ] Weights download verified from the public Hugging Face repos (`scripts/fetch_checkpoints.sh`)
- [ ] Evaluation gates reviewed ([EVALUATION.md](EVALUATION.md#release-gates-template)); missing measurements are stated, not hidden
- [ ] No secrets, private endpoints or local absolute paths in the diff
