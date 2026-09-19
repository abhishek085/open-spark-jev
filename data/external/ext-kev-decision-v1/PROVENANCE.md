# ext-kev-decision-v1

- **Upstream**: https://github.com/jaredpalmer/kev
- **Pinned commit**: `cc954f2f66d86943688fdbfa07b6db18f82ed065`
- **License**: Apache-2.0 (Kev's conversion/harness). Each underlying dataset keeps its own license, listed in the raw manifest.
- **Retrieved**: 2026-09-19
- **Records written**: 736  (`data/benchmarks/external/ext-kev-decision-v1.jsonl`)
- **Skipped**: 116 choice questions with more than 26 options (e.g. Banking77's 77-way).

## What it is
Test partition of Kev's decision-v1 suite: Banking77, BoolQ, AG News, MNLI, SST-5, Yelp converted to typed questions, plus none-of-the-above variants.

## Where the labels come from
Gold labels of the underlying public datasets (see raw manifest for dataset revisions), converted by Kev into TypeSafe-shaped requests.

## Recorded Jev output
No row-level Jev output. Kev's repo publishes only AGGREGATE Jev accuracy per task on the matching development suite (raw/docs/kev-vs-jev-summary.json).

## Caveats
This is Kev's own suite: sampled and rendered by the Kev authors (option order, wrappers, distractor variants are theirs). Kev was TRAINED on the in-distribution sources of decision-v1, so those slices are in-domain for Kev but not for us; transfer-v4 sources are out-of-domain for Kev.

## What we changed
One record per question; option keys used as labels; score labels are 0-based level indices; choice descriptions rendered as '- key: text'.
