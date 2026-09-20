> Archived copy of the project README as of 2026-09-20 (before the README was reorganised). It is kept unchanged so no detail is lost; the maintained overview is [README.md](../README.md).

# open-spark-Jev

**spark-s1** is the model this repo builds and releases. `open-spark-Jev` is the repo name.

Release ids are `spark-s1-<size>-<recipe>[-v<n>]` — e.g. `spark-s1-1.7b-sft-v2`,
`spark-s1-1.7b-rlcd-direct-v1`. The size is the backbone's parameter count; the recipe is the
training mechanism; the version increments when a recipe is retrained on different data.
**"Menu scoring" is the mechanism** (restricted-vocabulary readout over option labels), never a
model name. spark-s1 is not Jev and not affiliated with TypeSafe: it is built by fine-tuning a
pretrained backbone, not trained from scratch, but follows the same System One contract — typed
Choice / Score / Noul questions in, calibrated per-option probabilities and a confidence out, in
one forward pass with no generated text.

**An open, local System One decision model for NVIDIA DGX Spark.**
Qwen3-4B and Qwen3-1.7B backbones · single-pass menu scoring · calibrated Choice / Score / Noul answers ·
code-labelled decision data · served behind a typed JSON API (TensorRT-LLM optional).

open-spark-Jev does one thing: given a *state* (text, JSON, logs, traces, game observations)
and a set of typed *questions*, it returns typed, calibrated answers in a single forward pass.
It never generates free text, so it cannot produce a malformed or hallucinated output, and it
is fast enough to sit inside request paths, agent loops and control policies.

| primitive | asks | returns |
|---|---|---|
| **Choice** | pick one of N options | selected option, full distribution, confidence |
| **Score** | place the state on an ordered rubric or numeric band | level, distribution, expected value, confidence |
| **Noul** | is this claim true of the state? | calibrated P(true) |

Inspired by the public shape of TypeSafe's Jev; fully open weights and code, nothing leaves the box.

```python
from open_spark_jev.model import MenuScorer
from open_spark_jev.schema import Choice, Noul, Score, State

m = MenuScorer("models/Qwen3-1.7B")
state = State(content={"ticket": "Charged twice for my Pro plan, invoice PDF missing."}, schema_hint="support ticket")
answers = m.decide(state, [
    Choice(prompt="Which support queue?", options=["billing", "technical", "account", "sales", "abuse"], allow_abstain=True),
    Score(prompt="How urgent?", levels=["low", "medium", "high", "critical"]),
    Noul(prompt="The customer was charged more than once."),
])
# 3 decisions, 1 state prefix pass, ~tens of ms on a GB10
```

## The current models (2026-09-20)

| release id | backbone | 60-row tool-call set | p50 latency | best for |
|---|---|---|---|---|
| **spark-s1-4b-v3** | Qwen3-4B + LoRA | **0.850** (Jev, recorded: 0.917) | 66 ms (Jev recorded: 422 ms, hosted) | accuracy; best on 5 of 7 external suites |
| **spark-s1-1.7b-v3** | Qwen3-1.7B + LoRA | 0.783 | 30 ms | speed; 14x faster than Jev's recorded latency |

Both are public Qwen3 backbones fine-tuned (LoRA, merged) on 967 code-labelled decision rows from our data factory
([`datagen-pipeline/`](../datagen-pipeline/), reshuffled into ~4.3k examples with random option orders), then read out the Jev way:
one forward pass, first-token logits restricted to the option letters, temperature fitted on a separate calibration split.
No text is generated. **How they compare with Jev and with an ordinary prompted SLM, and what they are not:**
[docs/ARCHITECTURE.md](ARCHITECTURE.md#0-the-two-current-models-spark-s1-4b-v3-and-spark-s1-17b-v3).
Numbers and caveats (the 60-row set is a soft final check, n=60, Jev's latency is hosted and includes the network):
[docs/BENCHMARKS.md](BENCHMARKS.md) B27, B29-B31.

**This is an early release, and we are going to keep expanding it.** The models were trained on under a thousand rows, and it shows in the weak
packs (agent next-action routing, urgency scoring, retrieval and termination gates, answer sufficiency), in vulnerable-code detection (0.56, near
chance) and in uncalibrated Boolean/Score heads. Next: generate substantially more data with the factory across more task families (including new
capabilities such as code/data workflows and security), retrain, fit calibration for every question type, and apply outcome-based calibration
training (RLCD) on rows the model has not memorised. The plan and the reasoning are in [docs/EXPERIMENTS.md](EXPERIMENTS.md) and
[docs/ROADMAP.md](ROADMAP.md); releases will be versioned (`spark-s1-<size>-v<n>`) as data and capability grow.

## Try it: the playground (UI + API)
```bash
scripts/setup_env.sh && scripts/download_weights.sh Qwen/Qwen3-1.7B   # once
scripts/fetch_checkpoints.sh abhishek085/spark-s1-4b-v3                # released weights (also spark-s1-1.7b-v3) -> checkpoints/; repos are private until the maintainers make them public
scripts/run_ui.sh                                                      # open http://127.0.0.1:8400
```
A single-page web UI with ten example tasks (support triage, retrieval routing, SQL safety, prompt-injection
check, tool-call risk, email triage, RAG sufficiency, incident routing, CI failure triage, form completion),
a confidence-gating slider, side-by-side model comparison, and the exact Jev-compatible API call for whatever you
run. Reach it from a laptop over Tailscale (`--tailscale`) or an SSH tunnel. Details, API examples and honest
notes on what it gets right and wrong: [docs/UI.md](UI.md).

## How it works (short version)
1. The state is rendered once inside hard fences and run through the decoder; its KV cache is kept.
2. Each question is rendered as a menu (`A. ... B. ...`) plus the Qwen3 non-thinking assistant
   header, and appended to a copy of that cache.
3. The logits of the **first assistant token** are restricted to the bare label tokens
   (`A`, `B`, ...), divided by a calibrated temperature, and softmaxed. That distribution *is* the answer.

Because the "menu head" is the backbone's own LM head over ≤26 rows and the prompt is exactly the
chat template's non-thinking output, any OpenAI-compatible chat endpoint that returns
`top_logprobs` serves it unchanged. On Spark that is `trtllm-serve` (PyTorch backend, prefix
caching on); the gateway also has an in-container backend on the TRT-LLM Python API for
full-vocabulary logits. Verified against trtllm-serve 1.2.1. Details: [docs/ARCHITECTURE.md](ARCHITECTURE.md).

## Training
* **Phase 1, supervised** (`train/sft.py`): multi-task menu heads over simulator, public and
  teacher-generated corpora, cross-entropy against *soft targets* + Brier regulariser,
  per-type temperature fitting.
* **Phase 2, toward calibrated decisions** (`train/rlcd.py`): TypeSafe names their method
  "RLCD" but has published only the goal (a reward for "a confidence score that actually
  matches how often it's right"), not a mechanism - so this repo implements and compares
  *two* independent ones. **Mechanism 1**, the academic RLCD (Yang et al. 2023,
  `configs/train/rlcd_contrastive.yaml`): contrastive-principle pairs from a local teacher →
  a reward model that is itself a Noul → the exact menu policy gradient with the RM's
  per-action score folded in. **Mechanism 2**, a direct calibration objective
  (`configs/train/rlcd_direct.yaml`): the same exact policy gradient with the RM term off,
  optimizing proper scoring rules (Brier) directly against ground-truth/teacher soft labels -
  our most literal reading of TypeSafe's stated reward. Both use the same exact menu policy
  gradient machinery (the full action distribution is available in one forward pass, so the
  expected reward is differentiated directly, no PPO/GRPO sampling), with abstention,
  conservative-action and injection-consistency terms shared by both. A sampled-token TRL
  GRPO baseline and a no-RL temperature-only control (`eval/calibration_baseline.py`) round
  out the comparison.

Data: [docs/DATA.md](DATA.md). Training-mechanism experiments: [docs/RESEARCH.md](RESEARCH.md). Architecture-level experiments (including our own proposed novel decision-head design): [docs/NOVELTY.md](NOVELTY.md).

**Want to build your own domain-specific version of this, not just run this one?**
[docs/COOKBOOK.md](COOKBOOK.md) is a start-to-finish, reusable recipe: pick a backbone, write known-posterior simulators for your domains, generate and grade synthetic data with multiple teachers, run SFT + RLCD, evaluate, quantize, serve - with a reusability matrix showing what ports over as-is versus what you rewrite.

## Quickstart on DGX Spark
```bash
git clone <this repo> && cd Open-Spark-Jev
scripts/setup_env.sh                          # repo-local venv, CUDA-13 torch, TRL, PEFT
scripts/download_weights.sh Qwen/Qwen3-1.7B   # -> models/Qwen3-1.7B
.venv/bin/python scripts/smoke_test.py        # HF backend: decisions + latency line

scripts/make_data.sh                          # simulators (+ public sets, + teacher if TEACHER_BASE_URL)
scripts/train_sft.sh                          # Phase 1  -> checkpoints/sft-qwen3-1.7b
scripts/train_rlcd_direct.sh                  # Phase 2, mechanism 2 -> checkpoints/rlcd-direct-qwen3-1.7b (no teacher needed)
scripts/train_rlcd_contrastive.sh             # Phase 2, mechanism 1 -> checkpoints/rlcd-contrastive-qwen3-1.7b (needs a teacher on :8010)
scripts/eval.sh hf checkpoints/rlcd-direct-qwen3-1.7b
scripts/run_ui.sh                             # web UI + API over checkpoints/ (docs/UI.md)

deploy/spark/pull_trtllm.sh                   # TensorRT-LLM container (arm64, CUDA 13)
deploy/spark/quantize.sh configs/quant/fp8.yaml checkpoints/rlcd-qwen3-1.7b
deploy/spark/serve.sh engines/rlcd-qwen3-1.7b-fp8     # trtllm-serve :8355
deploy/spark/gateway.sh checkpoints/rlcd-qwen3-1.7b   # System One API :8400 (chat logprobs path)
#   or: deploy/spark/gateway_trtllm.sh <model>          # gateway inside the container, TRT-LLM Python API
deploy/spark/smoke_curl.sh
```
Spark specifics (container tags, kernels, memory, quantization + temperature re-fit):
[docs/DGX_SPARK.md](DGX_SPARK.md).

## API
`POST /v1/decide` (native, see `schema.DecisionRequest`) and `POST /v1/evaluate`, a
Jev-style wire format (`{"state", "questions": {name: {type, instructions, criteria}}}`) so
existing Jev client code can be pointed at a local endpoint. `examples/` has routing,
a calibration-aware agent gate, and a Jev-format client.

## Evaluation
**Third-party evals, reported per source (never pooled).** Seven suites from public Jev
projects live under `data/external/<source>/` with a `PROVENANCE.md` each (upstream URL, pinned
commit, license, where the labels come from, whether Jev's own outputs were recorded, caveats):
Jev Directory (70 typed questions), agent tool-call risk, prompt injection with and without
deployment context, vulnerable code, and Kev's decision-v1 / transfer-v4 suites. Rebuild with
`python scripts/external/build_external.py`; score a checkpoint with
`python -m open_spark_jev.eval.external --model <ckpt> --name <run>`, which writes one
`runs/external/<run>/<source>.json` per source plus a `SUMMARY.md`. Where a source recorded Jev's
answers we also report Jev's accuracy/ECE on the same rows and our agreement with it. These sets
are other people's, with their own labelers and Jev versions; read each PROVENANCE before quoting.

**Run log.** Every run that produced a quoted number, with its command and artifact path: [docs/RUNS.md](RUNS.md).

**Own simulator benchmark.** `python -m open_spark_jev.eval.benchmark` reports accuracy, macro-F1, Brier, NLL, ECE per
(question type × domain), soft Brier against the **known posterior** on simulator data,
Noul-specific Brier/ECE, and the prompt-injection flip rate. `eval/latency.py` produces the
Spark latency/throughput grid over state length × questions per state.

## Status
Work in progress (2026-09-20), first public release candidate. Built: the serving path and API (including the `POST /v1/decide` contract with per-decision probabilities,
confidence, margin, entropy and latency), the os-datagen data factory, seven third-party evaluation suites reported per source, and two trained models (above).
Measured and recorded: the version ladder v0-v3 and v6 ([docs/EXPERIMENTS.md](EXPERIMENTS.md)), the 60-row final review, five end-to-end API cases and the external sweep.
Refuted at our scale and recorded as such: prefix-LM attention (A2), slot-query head (A3), frozen-backbone head (v2), a small-model cascade (v6), real public text as extra training data (M17).
Not done: outcome-trained calibration on unseen rows (v7), a per-option scorer (v4), more data, FP8/NVFP4 engines. Older checkpoints (`spark-s1-1.7b-sft-v2`, RLCD variants) were trained on
simulators and gemma-graded synthetic data and are kept for comparison. See [docs/ROADMAP.md](ROADMAP.md), [docs/MODELS.md](MODELS.md) and [docs/RUNS.md](RUNS.md).

## Layout
```
datagen-pipeline/ os-datagen: code-labelled decision-data factory (own README, tests, Apache-2.0)
open_spark_jev/   schema · prompting · model · calibration · data/ · train/ · eval/ · serve/ · cli
configs/          model / train (sft, rlcd, rlcd_grpo) / serve (trtllm options, gateway) / quant (fp8, nvfp4)
deploy/spark/     pull_trtllm · quantize · serve · gateway · smoke_curl
scripts/          setup_env · download_weights · make_data · train_* · eval · smoke_test
docs/             ARCHITECTURE · DGX_SPARK · UI · DATA · RESEARCH · ROADMAP · RUNS (run log) · BENCHMARKS
tests/            CPU unit tests + GPU smoke test (pytest -m gpu)
```

## Project, license and attribution

Maintained by the **Nokast AI** open-source community. Our source, docs, configs, simulators and
synthetic-data generators are **Apache-2.0** (`LICENSE`).

Everything we did not write keeps its own license and terms — see [`NOTICE`](../NOTICE) for the full
list. In short: the Qwen3-1.7B backbone is Apache-2.0 and is downloaded at setup, not redistributed
here; the teacher models used only for offline data generation each carry their own terms; the
vendored third-party evaluation suites under `data/external/` are MIT or Apache-2.0 with their
upstream commit pinned and a `PROVENANCE.md` per source; the public datasets in `data/raw/` are
fetched at runtime and not redistributed.

*Jev*, *System One* and *TypeSafe* are TypeSafe AI's; *NVIDIA*, *DGX Spark* and *TensorRT-LLM* are
NVIDIA's. This project is independent and not affiliated with or endorsed by either. Jev's measured
numbers quoted here come from third-party repositories' committed runs, not from runs we made.

Contributing: [`CONTRIBUTING.md`](../CONTRIBUTING.md) · Security and safe-use notes: [`SECURITY.md`](../SECURITY.md)
