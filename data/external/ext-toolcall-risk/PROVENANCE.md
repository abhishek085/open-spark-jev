# ext-toolcall-risk

- **Upstream**: https://github.com/themsquared/jev-benchmark
- **Pinned commit**: `daf02b3b59b28db49d3e77b4a8054b6c0849ea22`
- **License**: Apache-2.0
- **Retrieved**: 2026-09-19
- **Records written**: 60  (`data/benchmarks/external/ext-toolcall-risk.jsonl`)

## What it is
60 hand-labelled agent tool calls (34 clear, 14 ambiguous, 12 adversarial) classified as readonly/destructive/privileged/exfiltration.

## Where the labels come from
Hand-labelled by the repo author (single annotator). The author invites label disputes via PR.

## Recorded Jev output
Yes, per row: `jev-latest` (meta.jev) and `jev-preview` (meta.jev_alt), run 2026-09-17: choice, per-option probabilities, confidence. Author reports 91.7% accuracy for both.

## Caveats
Single annotator, n=60, run-to-run variance of a few points (the author says so). Slices: difficulty is used as the domain suffix.

## What we changed
Options sorted alphabetically (as in the upstream harness); criteria text copied verbatim into the question.
