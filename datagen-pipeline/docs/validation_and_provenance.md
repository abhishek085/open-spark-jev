# Validation and provenance

Gates (fail-closed, in order): schema → leakage → anchors + structured checks → independent verification → optional judge →
policy re-run → dedupe/diversity → split isolation.

- **Schema:** strict JSON, Pydantic, extra fields forbidden, a code fence is the only tolerated wrapper. A repair pass is
  optional; repaired output re-runs every gate and records `repair_attempt_count`.
- **Leakage:** option ids (underscored ids also match spaced/hyphenated forms), gold-field names, oracle names and answer
  phrases ("the correct action is", "should choose"). Per-pack allow/deny lists live in `configs/taskpacks/*.yaml`;
  all matches, including allowed ones, are kept in `validation_results.jsonl` (false-positive audit).
- **Fact preservation:** the verifier receives the visible text and an extraction schema, never the truth or world;
  extracted facts must equal the world facts (task-defined equivalence). Unknown facts route to the judge or reject.
- **Judge:** verdict `accept|reject|human_review` plus fact/ambiguity/leak fields. It cannot modify labels; a
  `human_review` verdict is written to `rejected.jsonl` and counted in the manifest.
- **Dedupe / diversity:** exact hash, then near-duplicate (n-gram shingles, or an injected embedder with a cosine threshold);
  the harder record is kept; an easy-share cap keeps splits from being dominated by easy templates.
- **Split isolation:** enforced at world assignment and re-checked on rendered records (ids, template families, entity pools,
  cross-split near-duplicates) → `reports/split_isolation.json`.

Every accepted record carries generator/verifier/judge model ids, prompt versions, scenario id and seed, repair count, config
hash, git commit (with a `+dirty_worktree` marker) and the truth tier. `prompts.lock.json` pins prompt template hashes;
`checksums.sha256` covers all artifacts; raw generations are stored artifact-relative, size-capped and redacted.

## Additions after real-model review (2026-09-20)

**The verifier is never the reason a row is accepted.** Truth comes from the code oracle / solver / controlled world / sandbox. The verifier only
confirms that the *text* still carries the world's facts. Acceptance additionally requires a reproducible oracle result and a truth tier other than
`model_consensus` (asserted by tests).

**Decision rules must be visible.** A label is only trainable if the visible state, policy text and option definitions let a reader derive it. For the
policy packs the rules are therefore rendered **by code** (never paraphrased by the LLM) into the state and mirror the oracle's evaluation order:
authorization/tool gate → `policy_excerpt`; router, retrieval, termination, answer sufficiency, urgency → `decision_policy` (urgency's rubric is generated from
`configs/policies.yaml`); injection → `trusted_policy`. The router additionally shows tool `cost_units`, `estimated_complexity` and `disallowed_actions`.
Constant policy text is excluded from leakage scanning and dedupe (`code_rendered_keys`).

**Structured facts are injected, not re-typed.** `finalize_surface` overwrites truth-bearing structure (tool call + metadata, proposed extraction, the temporal question,
policy text, whitelisted metadata) with the world's exact values; the LLM only writes prose. Hidden keys the generator copies into free-form metadata are dropped.

**Scenario tags are first-class.** `required_surface_features` (e.g. `non_english_header`, `typos_or_abbreviations`, `very_short_message`) must be present in the
text: verified by the verifier, or by code where objective (`code_checked_features`). A row whose family promises a feature it does not show is rejected, so per-slice
metrics stay meaningful.

**Supportability audit (blind solve).** After fact verification, the verifier model gets ONLY the model-facing prompt (no label), reasons in a short `analysis`, and returns
`choice`, `also_valid`, `missing_info`, `label_hint_in_text`. For packs with `supportability = "reject"` (policy/judgement packs) a wrong `choice` rejects the row;
hedges (`also_valid`, `label_hint`, `missing_info`) are recorded but never reject on their own (they were measured to be noisy: every contract "hinted" its own class).
For calculation-style packs (temporal, rules, entailment, extraction, document type, relevance) a disagreement is tagged only, because an LLM's arithmetic is not
evidence that the row is under-specified. The challenge split is always tag-only so it is not filtered toward easy rows. Every disagreement is stored in `validation_results.jsonl`.

**Held-out scenario families.** `generation.holdout_families` (default on) reserves up to 34% of each pack's scenario families for calibration + locked test only, chosen so
every label is still learnable from the remaining train families. `reports/split_isolation.json` lists them and fails if one appears in train.

**Option-order counterfactuals.** `os-datagen option-order --dataset D [--model-endpoint ...] --k 4` re-renders each record under K option permutations, runs a baseline and flags
position-sensitive rows (challenge-slice candidates).
