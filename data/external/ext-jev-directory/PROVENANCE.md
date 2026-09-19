# ext-jev-directory

- **Upstream**: https://github.com/everyai-com/jev-directory
- **Pinned commit**: `b4d676758d4bfbb55b1402d0fea7a3f65b4c27b5`
- **License**: MIT
- **Retrieved**: 2026-09-19
- **Records written**: 70  (`data/benchmarks/external/ext-jev-directory.jsonl`)

## What it is
50 runnable judge-model evals (70 typed questions: boolean/choice/score) written by the Jev community, each with the exact `experimental_evaluate` state and questions.

## Where the labels come from
Expected verdicts are pinned in the repo manifest (`capabilities.json` -> `manifest.evals`). They are the directory author's intended answers, not human-adjudicated ground truth.

## Recorded Jev output
None. No Jev outputs are published; this source gives inputs and expected answers only.

## Caveats
Tiny (70 questions) and hand-written for showcase, mostly easy. An eval passes only if every one of its questions matches (see eval-level pass rate). Score expected values are 1-based in the manifest; we convert to 0-based level indices.

## What we changed
Boolean -> our Noul; choice/score criteria rendered into the question text the same way `serve/gateway.py::_from_jev` does. One record per question.
