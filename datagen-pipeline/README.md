# os-datagen — open-spark-Jev decision-data factory

`os-datagen` is a **local, open-source synthetic-data and evaluation-data factory** for Jev-like probabilistic
decision models. A decision model receives *shared state + a runtime-defined typed question + allowed options*
and returns *a selected option, a probability for every option, and a calibrated confidence*. This tool builds
the **verifiable datasets, frozen evaluation suites, calibration sets, challenge sets and provenance artifacts**
for such a model *before any fine-tuning starts*. It does not train anything.

It is **not** only an agent-harness or code dataset generator. It is a general decision-data factory with two
namespaces:

| namespace | what it teaches | packs |
|---|---|---|
| `foundation/` | general reusable decision competence | semantic entailment, document type, document relevance, extraction validation, rule application (Boolean), temporal reasoning, communication intent, communication urgency (Score), authorization |
| `harness/` | agent / automation control | next-action routing, tool-action gate, retrieval gate, termination gate, answer sufficiency, prompt-injection gate |

All three primitives are supported: **Choice**, **Score** (ordered rubric level) and **Boolean**.

> **Central principle: code establishes truth; LLMs create surface variation.** Policy engines, symbolic solvers,
> controlled worlds and a restricted sandbox decide the label. A local LLM only renders language, distractors and
> adversarial phrasing; an *independent* verifier model checks that the text still carries the facts. A model never
> generates an example, picks the gold label, and approves itself.

## Pipeline (what makes a row acceptable)

Truth is decided by code. A row is accepted only if: it parses strictly; nothing leaks the label (option ids, family names, narration such as "Input check: details missing");
required facts survive (deterministic anchors + independent verifier); promised scenario features are really present; the oracle reproduces; the decision rules are visible in the
state (code-rendered policy); a blind reasoning solver can re-derive the label (policy packs); it is not a duplicate; and its split is isolated (own entity pools, style families, wording,
and held-out scenario families for calibration/test).

```
scenario sampler ─► ScenarioWorld (facts | hidden) ─► code oracle ─► TruthRecord
                          │ facts only (never the label)
                          ▼
                 generator LLM (surface JSON) ─► schema ─► leak ─► anchors ─► structured checks
                                                                        │
                       independent verifier LLM (sees visible text + schema, never truth) ─► fact comparison
                                                   │ unresolved only
                                                   ▼
                                         optional judge (verdict only, cannot edit labels)
                                                   │
                     policy re-check ─► dedupe / diversity ─► split isolation ─► accepted_* + rejected + reports
```

Validation **fails closed** and every rejection is written to `rejected.jsonl` with machine-readable reasons.

## Foundation vs harness packs

Foundation packs cover competence that is not about tools or agents: does the evidence entail the claim, what kind of
document is this, is the extraction right, what is the urgency of a message, may this requester access this resource.
Harness packs cover control decisions of an agent: which action next, may this tool call run, must it retrieve, is it
finished, is the answer sufficient, is this retrieved text trying to hijack the agent. Both use the same infrastructure,
oracle interface, validators and reports. Everything in `docs/` explains a pack's families, truth source and
probability semantics.

## Quick start (no GPU, no network)

```bash
cd datagen-pipeline
export PYTHONPATH=src
../.venv/bin/python -m pytest -q                      # unit + integration (fake LLM for CI only)
../.venv/bin/python -m os_datagen generate --task-pack foundation_document_type_v1 \
    --count 100 --seed 42 --dry-run --out artifacts/dry_run
../.venv/bin/python -m os_datagen report --run artifacts/dry_run
```

`--dry-run` uses a deterministic **fake** generator/verifier. It exercises the plumbing (validation, splits, reports)
and says nothing about data quality: manifests from dry runs carry `dry_run: true` and a note saying so.

## Real generation on a local model

The pipeline needs only an OpenAI-compatible `/v1/chat/completions` endpoint per role (vLLM, SGLang, TRT-LLM, llama.cpp). See `docs/local_serving.md`.
On one GB10 the roles run **one at a time**, each as a few parallel instances of the same model (`base_urls`, round-robined). Put `start_cmd`/`stop_cmd` per model in the
config (`configs/models.smoke.yaml`) and the scheduler starts each phase's servers, unloads the previous role, and never starts the judge unless there are unresolved cases.

```bash
os-datagen doctor   --models configs/models.smoke.yaml                    # endpoints, memory headroom, prompts, every pack's oracle
os-datagen generate --all --count 50 --models configs/models.smoke.yaml --out artifacts/smoke_e2e     # end to end, all three roles
os-datagen generate --task-pack harness_next_action_router_v1 --models configs/models.smoke.yaml --count 1000 --seed 42 --out artifacts/next_action_run_001
os-datagen generate-mixture --models configs/models.smoke.yaml --count 10000 --out artifacts/general_mixture_001
```

Tuning loop (cheap): `os-datagen generate-only ...` with the generator up, swap, then `os-datagen reverify --run OLD --out NEW` with the verifier up; comparator changes only need `reverify`
on saved generations. `scripts/dev_cycle.sh` automates one cycle. Always trial small first, read `reports/sample_audit.md`, then scale.

## Inspecting results

```bash
os-datagen report   --run artifacts/next_action_run_001              # acceptance, rejection reasons, composition vs target
os-datagen inspect  --run artifacts/next_action_run_001 --record-id osj-nar-000001-v001
os-datagen validate --dataset artifacts/next_action_run_001/accepted_train.jsonl   # re-run gates, no LLM
```

A run directory holds `accepted_{train,calibration,test_locked,challenge}.jsonl`, `rejected.jsonl`, `manifest.json`,
`checksums.sha256`, `lineage.jsonl` and `reports/` (see `docs/data_contract.md`).

## Why four separately generated splits

Train, calibration, locked test and challenge are assigned **at scenario generation time**, not by shuffling rendered
rows. Each split has its own template/style family (`train_family_a`, ...), its own entity pool, its own instruction
wording, and its own seeds, and `reports/split_isolation.json` proves no scenario id, template family, entity pool or
near-duplicate text crosses a split. Calibration must be independent of training, the locked test must never be used
for fitting, and the challenge split is deliberately out-of-distribution.

## Probability calibration (later)

Numeric probabilities are never taken from a teacher LLM's stated confidence. The data stores deterministic point
masses, acceptable-option sets, and (where they exist) reviewer or rollout distributions. Later: fit on train →
temperature-scale on calibration → report NLL/Brier/ECE/selective risk on the locked test → report robustness on the
challenge split. Raw softmax scores are not probabilities; see `docs/probability_semantics.md` and
`os_datagen/training/README.md`. `os-datagen evaluate` already implements the metrics and saves per-row predictions.

## Limitations

- This does **not** reproduce TypeSafe Jev or its proprietary RLCD/data/training process. It is an independent open implementation of the idea.
- Nothing here trains a model. Milestone 7 is a data export and a written handoff only.
- Label quality equals the quality of each oracle. Communication-intent and ambiguity families rest on declared policies and should be human-audited before being used as high-stakes evaluation.
- The `data_code_workflows` family (5% target) has no task pack yet (the sandbox library for it exists) and is reported as 0%.
- Verifier and generator errors can still correlate. Use different model families; runs where both roles are the same model are marked `lower_confidence_provenance`.
- Rollout success rates for model routes are *declared profiles* unless you point the executor at real endpoints; outputs say which.

## Using this data to train models

`os-datagen` only produces data; training happens elsewhere (for this repo, the root `open_spark_jev` menu-scoring trainers, or any SFT stack).
The data was designed for a **decision model** that reads *state + question + options* and returns a probability for every option.

### 1. Produce the data
```bash
# a real mixture under the composition targets (scheduler starts/stops the local models; see docs/local_serving.md)
os-datagen generate-mixture --count 20000 --models configs/models.smoke.yaml --out artifacts/general_mixture_001
```
The run directory contains `mixture_accepted_{train,calibration,test_locked,challenge}.jsonl` (target-balanced; use these for training) and the unbalanced `accepted_*.jsonl`.
Always run `os-datagen validate --dataset <file>` (no LLM) and read `reports/summary.json` + `reports/sample_audit.md` first. Treat a few thousand rows as a **pilot**; decision models need
tens of thousands of varied rows, so scale `--count` and re-check acceptance per pack before relying on it.

### 2. What each split is for (never mix them up)
| file | use | never |
|---|---|---|
| `*_train` | fit weights | — |
| `*_calibration` | fit the calibrator (temperature scaling), pick thresholds | train on it |
| `*_test_locked` | one final measurement (NLL, Brier, ECE, accuracy, selective risk) | tune on it, fit on it, or look at it repeatedly |
| `*_challenge` | out-of-distribution robustness/safety report (hard by construction) | train on it, or use it as the headline number |

Calibration and locked-test use **scenario families that never appear in train** (see `reports/split_isolation.json` → `heldout_families`), plus their own entity pools, style families and
instruction wording. So accuracy on them measures generalisation to new scenario structure, not memorisation. `export-training` refuses non-train splits unless you pass `--splits`.

### 3. Export in the format your trainer wants
```bash
R=artifacts/general_mixture_001
# (a) open-spark-Jev menu-scoring trainers (Choice / Score / Noul; loads with open_spark_jev.data.corpus.Record)
os-datagen export-training --dataset $R/mixture_accepted_train.jsonl --format jev --out ../data/synthetic/osdg_train.jsonl
os-datagen export-training --dataset $R/mixture_accepted_calibration.jsonl --format jev --splits calibration --out ../data/benchmarks/osdg_calibration.jsonl
os-datagen export-training --dataset $R/mixture_accepted_test_locked.jsonl --format jev --splits locked_test --out ../data/benchmarks/osdg_test_locked.jsonl
os-datagen export-training --dataset $R/mixture_accepted_challenge.jsonl --format jev --splits challenge --out ../data/benchmarks/osdg_challenge.jsonl
# (b) plain chat SFT (user = rendered prompt, assistant = option id / level / true|false)
os-datagen export-training --dataset $R/mixture_accepted_train.jsonl --format chat --out osdg_train_chat.jsonl
# (c) prompt/completion + full target distributions
os-datagen export-training --dataset $R/mixture_accepted_train.jsonl --format prompt_completion --out osdg_train_pc.jsonl
```
Mapping to open-spark-Jev: `choice -> Choice` (option ids as labels; definitions are appended to the prompt), `score -> Score` (levels `0..3`, rubric in `rubric`), `boolean -> Noul` (`true->yes`, `false->no`).
`target.dist` is a one-hot point mass for deterministic truth (or a declared distribution when a pack provides one); `meta.acceptable` holds acceptable-option sets for evaluation.

### 4. Train (examples)
**open-spark-Jev (single-token menu heads, RLCD later):** add the exported file to the `data:` list of an SFT config and run the existing trainer.
```yaml
# configs/train/sft_osdg.yaml  (copy of sft_m17.yaml with your data)
model: models/Qwen3-1.7B
output_dir: checkpoints/sft-osdg-qwen3-1.7b
data: [data/synthetic/osdg_train.jsonl]      # optionally + public/teacher corpora, see below
```
```bash
python -m open_spark_jev.train.sft --config configs/train/sft_osdg.yaml
```
**Any SFT stack (TRL sketch):** train on `osdg_train_chat.jsonl` with the assistant turn as the *only* supervised tokens, so the model learns to emit just the label.
```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
ds = load_dataset("json", data_files="osdg_train_chat.jsonl")["train"].select_columns(["messages"])
SFTTrainer(model="Qwen/Qwen3-1.7B", train_dataset=ds,
           args=SFTConfig(output_dir="out", assistant_only_loss=True, max_length=2048, num_train_epochs=2, learning_rate=2e-5)).train()
```
For probabilities, score the options at inference instead of sampling: read next-token logits over the option labels (single-token letters, as the repo's menu scorer does) or use
`os-datagen evaluate --mode candidate_logprob`. Do not ask a model to *write* "0.83" and do not train on a teacher's stated confidence.

### 5. Things that matter when training on this data
* **Keep the policy in the state.** `decision_policy` / `policy_excerpt` / `trusted_policy` are the rules that make the label derivable; the model must read them. Do not strip them.
* **Use the truth tiers.** `truth.label_quality` is `deterministic`, `executable` (sandbox-checked), `controlled_world`, or `adjudicated` (declared ambiguity policy). Filter or down-weight `adjudicated` rows
  and audit them by hand before using them for evaluation; use `acceptable_options` when scoring, but train on the preferred option/point mass unless you have a real distribution.
* **Option order is shuffled** per record (`meta.variant`); do not undo it, and run `os-datagen option-order` on your trained model to find position-sensitive behaviour.
* **Mixing with other data.** Keep this data's own splits; do not put paraphrases of one scenario in both train and evaluation. Domain coverage is synthetic and partly templated (worlds come from code): pair it with real public text
  if you care about third-party generalisation, and measure with an external benchmark.
* **Provenance and licences.** Rows are synthetic and fictional; check the licence terms of the generator/verifier models you used (recorded in `manifest.json` and per-row provenance) before redistributing derived weights or data.

### 6. Calibration and evaluation gates (in this order)
1. Train on `train` only. 2. Fit temperature scaling on `calibration` only (`os-datagen evaluate --calibration-dataset ...`). 3. Report NLL, Brier, ECE, macro accuracy, per-option precision/recall and selective risk on
`test_locked` once (`os-datagen evaluate --dataset ... --model-endpoint ... --model ... --mode candidate_logprob`; per-row predictions are always saved). 4. Report `challenge` separately. 5. Check `reports/split_isolation.json` is `ok: true`.
Raw softmax scores are not calibrated probabilities; make no calibration claim without step 2 on an independent split. Compare against `baseline://uniform` and `baseline://prior` (a sanity floor).

## Measured status (real local models, 2026-09-20)

End-to-end run `os-datagen generate --all --count 50` with `configs/models.smoke.yaml` (random seed; generator `nvidia/Gemma-4-26B-A4B-NVFP4`, verifier and supportability solver
`nvidia/Qwen3.6-35B-A3B-NVFP4`, judge `nvidia/Qwen3.6-27B-NVFP4`; one role at a time, 2 parallel instances each), then `reverify` after comparator fixes.
750 candidates -> **620 accepted (83%)**, 613 after the record-level prune of 7 answer-sufficiency rows that echoed the constraint text. Split isolation OK, all checksums verify,
`validate` reports 0 issues on the pruned files, judge routed 6 cases (0 human-review). Truth tiers: deterministic 315, controlled_world 244, executable 54, adjudicated 7.

Accepted per pack (of 50): entailment 42, document type 46, relevance 44, extraction 47, rule application 46, temporal 38, intent 48, urgency 37, authorization 42,
router 38, tool gate 34, retrieval 38, termination 47, answer sufficiency 33 (26 after prune), injection 40.

Read these numbers as a smoke test, not a benchmark: 50 worlds per pack, one seed, one generator/verifier pair. Known limits: composition vs the long-term target is off (entailment is
6.8% vs 25%, security 13% vs 5%: the smoke run used equal counts per pack; `generate-mixture` applies targets); `data_code_workflows` has no pack yet; the supportability solver is itself an LLM
(it rejects only on a wrong choice for policy packs and is tag-only elsewhere); nothing has been fine-tuned; baselines on the locked test are near chance (prior 0.25 top-1) and the
uniform baseline is position-sensitive on 84% of rows (option-order check), as expected. Throughput: 3 generator instances ~1.3x one (GB10 decode is memory-bandwidth-bound).
Review samples: `artifacts/smoke_e2e_v2/samples_for_review.md`.
