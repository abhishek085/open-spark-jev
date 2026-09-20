# Architecture

Roles are separated: **truth** (no LLM), **surface realization** (generator LLM), **fact verification** (independent
LLM), **escalation judge** (optional, verdict only), **dedupe/split/provenance** (code).

```
src/os_datagen/
  schemas/      ScenarioWorld, TruthRecord, DatasetRecord, RejectionRecord, ExecutionRecord, manifest
  taskpacks/    BaseTaskPack + 15 packs (foundation/, harness/); registry.py loads lazily
  generation/   scenario_sampler (label-balanced), oracle, renderer (Jinja prompts + repair), pipeline, symbolic solver,
                controlled_worlds (per-split pools/styles), adversary (challenge worlds), variants
  validation/   schema, leakage, fact_extraction (verifier call), fact_comparison, contradiction, policy, dedupe, split_isolation, acceptance
  execution/    sandbox (allowlisted ops), fixtures, mock_tools, controlled_corpus, rollouts (Wilson bound), validators
  datasets/     writer, manifest/checksums, splitter, lineage, reporting
  evaluation/   runner (constrained_generation | candidate_logprob | baselines), metrics, calibration, slices, reports
  training/     render.py (model-facing prompt), export.py (train split only), README.md (handoff)
  llm/          OpenAI-compatible client, structured output, phase scheduler (+ optional server controller), fake clients (CI)
```

## Phases (sequential by default)

`sample_and_oracle` (no model) → `generate` (generator) → `verify` (verifier) → `judge` (only if needed) → `finalize`.
Each phase names the model role it needs; the scheduler checks memory headroom, and with `start_cmd`/`stop_cmd`
configured starts that role's server and unloads the previous one. Generator and verifier are configured separately;
if they point at the same endpoint/model the run warns and marks records `lower_confidence_provenance`.

## Acceptance state machine

```
SAMPLED → ORACLE_OK          oracle error = pipeline error (aborts), never a data rejection
→ GENERATED → SCHEMA_OK      strict JSON + Pydantic (extra fields forbidden); one optional repair pass, all gates re-run
→ LEAK_OK                    option ids, gold-field names, oracle name, answer phrases (per-pack allow/deny; audit of matches kept)
→ ANCHORS_OK                 required strings present + pack structured checks (e.g. exact tool arguments)
→ VERIFIED                   independent extraction == world facts (+ scenario-tag features); unverifiable → judge; mismatch → reject
→ SUPPORTED                  blind-solve audit: a wrong choice rejects policy-pack rows; tagged elsewhere (see validation_and_provenance.md)
→ POLICY_OK                  oracle re-run reproduces truth; options legal
→ UNIQUE                     exact hash + near-duplicate (shingles or injected embedder; the harder record is kept) + diversity cap
→ SPLIT_OK → ACCEPTED        split isolation report; otherwise REJECTED with reasons at any stage
```
