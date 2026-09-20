# ext-kev-transfer-v4-dev

- **Upstream**: https://github.com/jaredpalmer/kev
- **Pinned commit**: `75cc15ddb8e2d84bb85e68200c32d31da7f09f54`
- **License**: Apache-2.0 (Kev's conversion/harness). Each underlying dataset keeps its own license, listed in the raw manifest.
- **Retrieved**: 2026-09-20
- **Records written**: 764  (`data/benchmarks/external/ext-kev-transfer-v4-dev.jsonl`)
- **Skipped**: 0 choice questions with more than 26 options (e.g. Banking77's 77-way).

## What it is
Development partition of Kev's transfer-v4 suite (same sources as the test partition). Kev publishes per-source Kev and Jev accuracy on this partition, so it is the like-for-like chart set.

## Where the labels come from
Gold labels of the underlying public datasets (see raw manifest for dataset revisions), converted by Kev into TypeSafe-shaped requests.

## Recorded Jev output
No row-level Jev output; Kev's cards publish Jev's per-source and overall accuracy on this development partition (0.857 overall).

## Caveats
This is Kev's own suite: sampled and rendered by the Kev authors (option order, wrappers, distractor variants are theirs). Kev was TRAINED on the in-distribution sources of decision-v1, so those slices are in-domain for Kev but not for us; transfer-v4 sources are out-of-domain for Kev.

## What we changed
One record per question; option keys used as labels; score labels are 0-based level indices; choice descriptions rendered as '- key: text'.
