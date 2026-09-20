# Data contract

Three layers, never mixed:

1. **ScenarioWorld** (internal, `scenario_worlds.jsonl`): `facts` (renderable; the only thing the generator sees) and
   `hidden` (bookkeeping). Includes `scenario_family`, `split_family`, `seed`.
2. **TruthRecord** (internal): computed only by code/policy/solver/controlled world/sandbox/declared adjudication.
3. **DatasetRecord** (model-facing, `accepted_*.jsonl`): `decision{type,state,question}`, public `truth`, `quality`,
   `provenance`, and an analysis-only `meta` (namespace, scenario_family, split_family, pool, variant). Never feed `meta`
   or `truth` to a model; `training/render.py` renders only `decision`.

`decision.type` ∈ `choice` (options + definitions; option order is shuffled per variant), `score` (levels),
`boolean` (instructions only; `truth.truth` is true/false).

`truth.label_quality` names the tier: `deterministic`, `executable` (sandbox-checked), `controlled_world`, `rollout`,
`adjudicated` (declared ambiguity policy; needs human audit), `model_consensus` (never used as unqualified truth).

`RejectionRecord` carries `reasons` (`schema:*`, `label_leak:*`, `fact_mismatch:*`, `fact_unverifiable:*`, `judge:*`,
`dedupe:*`, `diversity_cap:*`, `policy:*`, `generation_failed:*`), validator versions, and an artifact-relative raw path.

Run directory: `config.resolved.yaml`, `prompts.lock.json` (sha256 of every template), `manifest.json`, `scenario_worlds.jsonl`,
`candidates_raw.jsonl`, `accepted_{train,calibration,test_locked,challenge}.jsonl`, `rejected.jsonl`,
`execution_results.jsonl` (sandbox/mock executions), `validation_results.jsonl`, `lineage.jsonl`, `reports/`, `checksums.sha256`, `raw/`.

## Model-facing state keys added by code
`decision_policy` (router, retrieval, termination, answer sufficiency, urgency), `policy_excerpt` (authorization, tool gate) and `trusted_policy` (injection) are constant,
code-rendered decision rules. `quality.supportability` is `supported | tagged_disagreement | skipped`. `meta` carries `scenario_family` / `split_family` / `pool` for slicing only.
