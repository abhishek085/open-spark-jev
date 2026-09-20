# Contributing

open-spark-Jev is maintained by the **Nokast AI** open-source community. Contributions are
welcome — especially new domains, new evaluation sources, and results that contradict ours.

## Ground rules for results

This repo's value is that its numbers can be checked, so:

* **Every quoted number needs a command and an artifact.** Add a row to [docs/RUNS.md](docs/RUNS.md)
  with what you ran, on what data, and where the output file is.
* **Never pool evaluation sources.** Each third-party source is scored separately and quoted by
  name, because their labels, annotators and difficulty differ.
* **Vendored data needs a `PROVENANCE.md`** — upstream URL, pinned commit, license, where the
  labels come from, and the caveats you would want a reader to know. `scripts/external/build_external.py`
  writes these; follow its pattern.
* **Report the failures.** Negative results and weak slices belong in `docs/BENCHMARKS.md` next to
  the good ones. A table with no bad rows is a table nobody can trust.
* **In-distribution numbers are not evidence of generalisation.** Say which held-out set a number
  came from.

## Development

```bash
scripts/setup_env.sh
.venv/bin/python -m pytest -q -m "not gpu"     # CPU tests
.venv/bin/ruff check open_spark_jev scripts tests
```

GPU tests need a CUDA device and downloaded weights: `pytest -m gpu`.

If you are running long jobs on a DGX Spark, read [docs/DGX_SPARK.md](docs/DGX_SPARK.md) first —
the box will thermally shut down under sustained load without the guard in `scripts/ops/`.

## Licensing

By contributing you agree your contribution is licensed under Apache-2.0 (see `LICENSE`).
Do not add third-party code or data without recording it in `NOTICE` and, for data, a
`PROVENANCE.md`. Do not commit model weights, API keys, or anything under a license that
forbids redistribution.

## Evaluation rule: keep per-row data

Every evaluation run must save per-row predictions (id, slice, gold, prediction, per-option probabilities)
next to its aggregate metrics, so failures can be analysed without re-running. `eval/external.py` writes
`runs/external/<name>/<source>.rows.jsonl`; new eval code must do the same.
