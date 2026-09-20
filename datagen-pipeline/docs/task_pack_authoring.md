# Authoring a task pack

Subclass `taskpacks/base.py:BaseTaskPack` and register it in `taskpacks/registry.py`. Implement:

| member | purpose |
|---|---|
| `families`, `challenge_families`, `options`/`levels`, `instruction_variants` (>= 5: train 0-1, calibration 2, locked 3, challenge 4) | coverage and split-specific wording |
| `sample(rng, family, pool, tone)` | returns `(facts, hidden)`. `facts` are what the generator sees: **never the label**, preferred option, oracle rule or scoring detail. Where a label would be a fact (document class, intent), pass *features* and derive the label in the oracle |
| `decide(world)` | the oracle → `Decision(preferred, reason, acceptable, outcomes)`; pure and deterministic. Errors are pipeline errors |
| `surface_fields`, `state_from_surface` | generator output schema (extra fields forbidden) and the model-facing state |
| `verifier_fields`, `expected_extraction(world)` | facts the independent verifier must recover from the visible text; `compare_facts` (override for tolerant fields) |
| `anchors`, `check_surface` | deterministic checks: required strings, exact structured content (e.g. tool arguments) |
| `leak_exclude`, `extra_leak_patterns`, `leak_view` | per-pack leakage policy (e.g. document class words occur naturally in documents) |
| `probability_semantics()` | what a probability means for this pack |
| `execute(world)` (optional) | sandbox / mock-tool execution producing `ExecutionRecord`s and upgrading the truth tier |
| `template_surface(world)` (optional) | offline rendering for `--dry-run` only |

Add `prompts/<ns>/<pack>/surface.j2` (spec-style, `world_facts_json` + `{{ notes }}`) and `verify.j2`, a
`configs/taskpacks/<pack>.yaml`, tests (a golden label per family), and run `os-datagen doctor`.

Rules: world `facts` never contain oracle names or labels; split isolation comes from `controlled_worlds.SPLIT_STYLES`
(entity pools, style families, instruction wording per split); sampling is label-balanced by `scenario_sampler`.

## Hooks added in the real-model iteration
`finalize_surface(world, surface)` injects exact structure/policy after generation; `code_rendered_keys` lists constant policy keys excluded from leakage/dedupe;
`supportability = "reject" | "tag"`; `verifier_defaults` (what silence means, e.g. no revenue impact stated); `required_surface_features` + `code_checked_features`;
`compare_facts` overrides canonicalise verifier notation (never mutate shared state: verification runs in threads). Make every world fact *observable* in the visible
text, and every decision rule visible in the state, or the row is rightly rejected. Regenerate `tests/golden/labels_seed42.json` deliberately when a sampler/oracle changes.
