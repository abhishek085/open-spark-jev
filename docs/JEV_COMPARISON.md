# Comparison against a real Jev evaluation

TypeSafe has not published a downloadable evaluation dataset for Jev, and there is no
official "102-row" or similar artifact - that number, when it appears in searches, comes
from an independent, non-TypeSafe-affiliated researcher's evaluation
([ickma2311/jev-baselines-eval](https://github.com/ickma2311/jev-baselines-eval): "Jev
returns confidence exactly 1.0 on 102 of 200 items" - a count of a subset within a 200-item
CLINC150 run, not a distinct 102-item dataset). This document builds the closest honest
comparison actually possible from public information, using that repo's real, raw, per-item
results (which it publishes and which we've reproduced independently), not any TypeSafe
artifact.

## What this is and isn't

**This is a real, reproducible alignment**, not an approximation:
- The Banking77 test data and sampling (`pandas.sample(n=300, random_state=0)`) are
  replicated exactly from that repo's source; **0 of 208 gold labels mismatched** its
  recorded results when we independently re-sampled, confirming byte-for-byte item alignment.
- Our model receives the *identical* wire-level request Jev received (same state text
  format, same question instructions), run through `serve/gateway.py`'s actual
  `/v1/evaluate` request-translation code, not a hand-written approximation of it.
- Jev's accuracy numbers here are recomputed from its own real per-item results file, not
  just its reported aggregate.

**This is not a same-task comparison**, and the numbers below should never be read as "our
model vs. Jev at the same task":
- Jev answered a **77-way** Choice question (Banking77's full intent set) in one call.
  Our `Choice` primitive is architecturally capped at 26 options (one per letter, see
  `docs/ARCHITECTURE.md`), so we cannot ask the same question at all. The only way to build
  *any* comparison is to reduce the label space - we use the top 20 most frequent intents by
  training-set frequency (the same reduction this repo's own `data/public.py` `banking77()`
  adapter already uses, chosen before this comparison existed, not picked to flatter a
  number). That makes our task strictly *easier* than Jev's (fewer options to choose among),
  and the two accuracy numbers below are not on equal footing.
- **Small sample.** Filtering Jev's 208 real items down to ones whose gold label survives
  the top-20 reduction leaves **46 items** - real, not manufactured to hit a target size, but
  small enough that individual-item noise matters. Treat differences as suggestive, not
  conclusive.
- **Out-of-domain for our model, not for Jev.** This is the most important caveat. Every
  checkpoint trained in this project (`docs/MODELS.md`) was trained exclusively on the six
  synthetic simulator domains (routing, security, risk, moderation, incident, game) -
  `data/raw/public_train.jsonl` (which includes a Banking77 adapter) was built but never
  included in an actual training run, due to a loading-script bug found and fixed mid-project
  (see git history) after the SFT run had already happened. So this is a genuine zero-shot,
  out-of-domain generalization test for our model - a fundamentally different and harder
  challenge than everything else measured in `docs/BENCHMARKS.md`, which is all in-domain.
  Jev, whatever its training data actually is (undisclosed), was evaluated on a task within
  its apparent general-purpose design intent.

## Results

Reproduce with: `.venv/bin/python scripts/jev_alignment_comparison.py --model <checkpoint> --top-k 20`

| model | n | accuracy | ECE | Brier |
|---|---|---|---|---|
| Jev (real, recomputed from raw data) | 46 | 0.761 (35/46) | not reported by that eval | not reported |
| our SFT checkpoint | 46 | 0.478 (22/46) | 0.233 | 0.719 |
| our RLCD-direct checkpoint | 46 | 0.457 (21/46) | **0.121** | 0.738 |

## Reading

**Accuracy: Jev clearly ahead, as expected.** A model evaluated zero-shot on a task within
its design scope, against models trained on entirely unrelated synthetic domains and asked
to generalize cold, losing by this margin is not a surprising or alarming result - it is
close to the expected outcome, and mostly demonstrates that this project's checkpoints
haven't yet been trained on anything resembling real-world text classification. It should
not be read as "our training mechanisms underperform Jev's" - the M8a-M8d comparison in
`docs/BENCHMARKS.md`, which is apples-to-apples in-domain, is the right place to look for
that question.

**Calibration: a real, if small-sample, signal worth noting.** SFT-to-RLCD-direct's ECE
improvement (0.233 -> 0.121) shows up here too, on data neither checkpoint was trained
anywhere near - consistent with the idea that whatever RLCD-direct's calibration training
does, it isn't purely memorizing the training domains' specific posteriors, though n=46 is
too small to lean on this hard.

**The most actionable finding isn't either number - it's the gap this comparison exposes.**
This project has never trained on real-world text at all. `data/raw/public_train.jsonl`
exists, is fixed, and was simply never used in a training run because of sequencing (the bug
fix landed after SFT had already happened). Including it in the next SFT run is a direct,
low-cost way to close a real generalization gap this comparison just made visible - a more
concrete and higher-priority next step than it would have been to guess at from the in-domain
numbers alone.

## Files

- `data/external/jev_baselines_eval/` - the exact third-party source data and results,
  vendored into this repo for independent reproducibility (see its `PROVENANCE.md`).
- `scripts/jev_alignment_comparison.py` - the reproducible comparison script.
- `runs/jev_comparison/`, `runs/jev_comparison_sft/` - saved per-item outputs for each
  checkpoint evaluated.
