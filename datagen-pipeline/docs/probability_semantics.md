# Probability semantics

Numeric probabilities are never trained from a teacher's stated "0.83". Sources are stored separately:

1. **Deterministic truth** — one correct result from rules/sandbox (point mass).
2. **Acceptable sets** — several valid options with a declared cost/risk preference (`acceptable_options`, `preferred_option`).
3. **Reviewer distributions** — independent human votes for inherently ambiguous items (`truth.distribution`, kind `annotator_distribution`).
4. **Rollout distributions** — empirical success rates from repeated execution, with a Wilson lower bound (kind `rollout_distribution`).
5. **Student predictions** — raw logits/probabilities collected at evaluation time.

Per pack: entailment/urgency/etc. are deterministic point masses; Boolean rule application's `false` means *not implied* under closed-world
rule application; urgency is an ordinal rubric so neighbouring levels are closer than distant ones; adjudicated ambiguity families declare
their alternative in `acceptable_options`. Raw softmax scores are not automatically reliable probabilities: calibrate on an independent split.
