## What and why

## Checks
- [ ] Tests pass (`pytest`) and lint is clean (`ruff check .`)
- [ ] New behaviour has tests; safety-relevant changes fail closed
- [ ] No command from user input is executed anywhere
- [ ] Benchmark claims are scoped (diagnostic vs locked, hardware, seeds) and traceable to an artifact
- [ ] Docs updated; no secrets, private endpoints or local absolute paths
- [ ] Per-row data saved for any new evaluation
