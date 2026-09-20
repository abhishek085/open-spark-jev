# Open Spark Jev

<p align="center">
  <img src="docs/assets/nokast-logo.png" alt="Nokast" width="72" />
  <br />
  <sub>Part of <b>Nokast</b>, an open-source AI community</sub>
</p>

<p align="center">
  <strong>An open-source, local implementation inspired by TypeSafe's Jev and System One models, built for NVIDIA DGX Spark. The model is called <code>spark-s1</code>.</strong>
</p>

<p align="center">
  <b>State in. Structured decisions out.</b>
</p>

<p align="center">
  <a href="docs/ARCHITECTURE.md"><img src="https://img.shields.io/badge/Architecture-how%20it%20works-2a78d6?style=for-the-badge" alt="Architecture" /></a>
  <a href="docs/EVALUATION.md"><img src="https://img.shields.io/badge/Evaluation-how%20we%20measure-eb6834?style=for-the-badge" alt="Evaluation" /></a>
</p>

<p align="center">
  <a href="https://huggingface.co/abhishek085/spark-s1-4b-v3"><img src="https://img.shields.io/badge/spark--s1--4b--v3-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black&labelColor=1f2328" alt="spark-s1-4b-v3 on Hugging Face" /></a>
  <a href="https://huggingface.co/abhishek085/spark-s1-1.7b-v3"><img src="https://img.shields.io/badge/spark--s1--1.7b--v3-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black&labelColor=1f2328" alt="spark-s1-1.7b-v3 on Hugging Face" /></a>
</p>

---

## What is Open Spark Jev?

**Open Spark Jev** is an independent, open-source implementation inspired by TypeSafe's Jev and System One, built for local inference on NVIDIA DGX Spark; it is not Jev, not a reproduction of TypeSafe's proprietary system, and not affiliated with TypeSafe. It is for decisions that need a fast, structured answer rather than generated text, such as routing, urgency, tool-call safety, or whether an agent should continue, stop or ask. The model it builds and releases is **`spark-s1`**.

Instead of asking a language model to generate an answer, you provide:

- a state
- one or more typed questions
- a set of possible decisions or criteria

The model returns **structured decisions, probabilities, and confidence**.

> **State + typed questions → structured decisions + probabilities**

The project currently supports three decision primitives:

| Primitive | What it does | Returns |
|---|---|---|
| **Choice** | Pick one option from a set | Selected option, full probability distribution, confidence |
| **Score** | Place a state on an ordered rubric or scale | Level, distribution, expected value, confidence |
| **Noul** (Boolean) | Determine whether a claim is true | Probability that the claim is true |

Several typed questions can be answered from the same state: the state is encoded once and each question adds only a short extra pass.

> **This is a very early release.** The models are trained on under a thousand examples, they are not at parity with Jev, and they have known failure modes (see [Current Limitations](#current-limitations)). We are generating substantially more data and expanding what the models can decide.

---

## Current Models

Two model sizes are currently available:

| Model | Backbone | 60-row tool-call set | p50 latency | Focus |
|---|---|---:|---:|---|
| [`spark-s1-4b-v3`](https://huggingface.co/abhishek085/spark-s1-4b-v3) | Qwen3-4B + LoRA | **0.850** | **66 ms** | Accuracy |
| [`spark-s1-1.7b-v3`](https://huggingface.co/abhishek085/spark-s1-1.7b-v3) | Qwen3-1.7B + LoRA | **0.783** | **30 ms** | Speed |

The 4B model performed best of our models on 5 of the 7 external evaluation suites used in the current release. The models are fine-tuned from public Qwen3 backbones using LoRA and merged for inference. Current training data consists of **967 code-labelled decision rows** from the project's decision-data factory (`os-datagen`), reshuffled into approximately **4.3k training examples** with randomized option ordering. The models use a restricted first-token readout rather than generating a text response.

> These are early release candidates. Performance varies substantially across task families, and the project is under active development. Model details, limitations and pinned revisions: [MODEL_CARD.md](MODEL_CARD.md).

---

## Quickstart

Download a model and run it as a local API in one command:

```bash
git clone https://github.com/abhishek085/open-spark-jev.git && cd open-spark-jev
python3 -m venv .venv && . .venv/bin/activate
pip install torch -e ".[infer]"
osj serve --pull spark-s1-4b-v3        # downloads from Hugging Face once (about 8 GB), then serves http://127.0.0.1:8400
```

In another terminal, ask it about a tool call (nothing is ever executed; the command is just text):

```bash
curl -s localhost:8400/v1/gate -H 'content-type: application/json' \
  -d '{"state": {"tool": "bash", "command": "kubectl get pods -n production"}}'
```

```json
{
  "choice": "allow",
  "probabilities": {"allow": 0.948, "ask": 0.036, "deny": 0.016},
  "confidence": 0.948,
  "policy_action": "ask",
  "policy_trace": ["Model: allow (0.948); P(allow)=0.948, threshold 0.995",
                   "Model allow is below the auto-allow threshold: escalate",
                   "Final policy action: ask"],
  "latency_ms": 66.8
}
```

*(Example output from a real run of `spark-s1-4b-v3`, trimmed.)* `policy_action` is what your agent should do. The default threshold is strict: `allow` needs 99.5% confidence, otherwise the agent asks a human. Lower it per request with `"policy": {"auto_allow_threshold": 0.9}`.

Use `spark-s1-1.7b-v3` for a smaller (3.4 GB) and faster model. From Python, without a server:

```python
from open_spark_jev import classify_tool_call

r = classify_tool_call(tool="bash", command="kubectl get pods -n production", model="checkpoints/spark-s1-4b-v3")
print(r.policy_action, r.probabilities)
```

A GPU is needed for the speeds above (tested on an NVIDIA DGX Spark; other GPUs should work but are untested). More: [docs/API.md](docs/API.md).

### Typed questions over any state

The general interface takes a state and any mix of Choice, Score and Noul questions:

```python
from open_spark_jev.model import MenuScorer
from open_spark_jev.schema import Choice, Noul, Score, State

model = MenuScorer("checkpoints/spark-s1-4b-v3")

state = State(
    content={"ticket": "Charged twice for my Pro plan, invoice PDF missing."},
    schema_hint="support ticket",
)

answers = model.decide(
    state,
    [
        Choice(prompt="Which support queue?", options=["billing", "technical", "account", "sales", "abuse"], allow_abstain=True),
        Score(prompt="How urgent?", levels=["low", "medium", "high", "critical"]),
        Noul(prompt="The customer was charged more than once."),
    ],
)
```

Each answer carries a full probability distribution: for example a Choice over the queues, a Score over the urgency levels, and P(true) for the claim. The important part is that the model returns structured decisions and probabilities, not a natural-language response.

---

## Try It

Two local pages share the same server:

```bash
pip install -e ".[serve]" && osj lab        # Decision Lab: http://127.0.0.1:8400/lab (no model download needed)
scripts/run_ui.sh                            # playground for general typed questions: http://127.0.0.1:8400
```

<p align="center"><img src="docs/img/decision-lab.png" alt="spark-s1 Decision Lab" width="820" /></p>

The **Decision Lab** is a demo page for tool-call decisions: pick a sample command or type your own, and see the model's probabilities next to the deterministic safety rules that can override them, and compare with a same-size generative model on time, tokens and malformed output. Without a model it uses recorded outputs, clearly labelled.

The **playground** has example tasks for support triage, retrieval routing, SQL safety, prompt-injection checks, tool-call risk, email triage, RAG sufficiency, incident routing, CI failure triage and form completion, plus confidence gating, side-by-side model comparison and API inspection. See [docs/UI.md](docs/UI.md).

---

## How It Works

Open Spark Jev uses the language model's own vocabulary distribution as a structured decision mechanism.

**1. Encode the state.** The state is rendered once and passed through the decoder, and its KV cache is retained.

**2. Turn the question into a menu.** A question is rendered as a set of options:

```text
A. billing
B. technical
C. account
D. sales
E. abuse
```

**3. Restrict the possible answers.** Instead of letting the model generate arbitrary text, the logits of the next token are restricted to the option labels (`A`, `B`, `C`, ...) and converted into a probability distribution.

**4. Calibrate.** A temperature is fitted on a separate calibration split. The result is the selected option, the probability distribution, the confidence, margin, entropy and latency. Instead of a generated sentence such as "I think this is probably a billing issue", the system returns something like:

```json
{
  "answer": "billing",
  "probabilities": {"billing": 0.91, "technical": 0.04, "account": 0.03, "sales": 0.01, "abuse": 0.01},
  "confidence": 0.91
}
```

Structured output constrains the *form* of the answer; it does not guarantee that the underlying decision is correct.

```text
                    ┌──────────────────┐
                    │      STATE       │
                    │ text / JSON /    │
                    │ logs / traces    │
                    └────────┬─────────┘
                             │
                        encoded once
                             │
                    ┌────────▼─────────┐
                    │  Qwen3 backbone  │
                    │   cached state   │
                    └────────┬─────────┘
                             │
             ┌───────────────┼────────────────┐
             │               │                │
          Choice           Score            Noul
             │               │                │
             ▼               ▼                ▼
        probabilities   probabilities     P(true)
```

This makes the approach useful for systems that need several decisions about the same piece of state. How it compares with Jev and with a prompted small model: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Evaluation

The project evaluates the models using internal benchmarks and external decision suites. Every result is reported per source and never pooled into one score.

**Internal:** accuracy, macro-F1, Brier score, negative log-likelihood, expected calibration error (ECE), option-order sensitivity, prompt-injection flip rate, latency and throughput.

**External:** seven suites (Jev Directory, agent tool-call risk, prompt injection with and without deployment context, vulnerable code, Kev decision-v1 and Kev transfer-v4), each with provenance covering its upstream source, commit, license, labels and caveats. See `data/external/`, [docs/BENCHMARKS.md](docs/BENCHMARKS.md), [docs/RUNS.md](docs/RUNS.md) and the evaluation protocol in [docs/EVALUATION.md](docs/EVALUATION.md).

### Independent evaluation: Kev's out-of-domain suite

Neither `spark-s1` nor Kev trained on these sources, and the suite was built by someone else, so it checks the models on data we did not write. Kev and Jev numbers are Kev's published ones; `spark-s1` was measured by us on the same 764 items per partition. `spark-s1-4b-v3` scores 0.749 on the locked test (Kev-4B 0.806, Kev-8B 0.780) and 0.706 on the development partition (Kev-4B 0.790, Kev-8B 0.796, Jev 0.857): behind overall, close on some sources, with far less training data.

<p align="center"><img src="docs/img/kev-comparison.png" alt="spark-s1 vs Kev vs Jev on Kev's transfer-v4 suite" width="900" /></p>

### Diagnostic set: 60 tool calls, directly against Jev

<p align="center"><img src="docs/img/toolcall-60-comparison.png" alt="spark-s1 vs Jev on the 60-case tool-call set" width="900" /></p>

| System | Diagnostic-set accuracy | p50 latency | Decisions/s | Notes |
|---|---:|---:|---:|---|
| Recorded hosted Jev reference | 0.917 | 421.6 ms | 2.3 | Hosted measurement; includes network/service overhead |
| `spark-s1-4b-v3` (Qwen3-4B, LoRA) | 0.850 | 65.9 ms | 15.1 | Local, one DGX Spark, batch 1 |
| `spark-s1-1.7b-v3` (Qwen3-1.7B, LoRA) | 0.783 | 29.6 ms | 33.8 | High-throughput routing/triage candidate |

> The 60-case set is a diagnostic benchmark used during development, not a locked final holdout, and it is small (about ±5 points). Results should not be read as broad Jev parity or a pure local-model-speed comparison: the hosted Jev timing includes a network round trip. These numbers are an early-release snapshot, not a general measure of model capability.

### Efficiency: `spark-s1` vs a generative small model (same backbone)

The same 60 tool calls, answered by the **untrained model of the same size writing a JSON answer** and by `spark-s1` reading out the option probabilities. Batch size 1, one DGX Spark.

| | Median time per decision | Decisions/s | Tokens written | Malformed output | Accuracy |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B writing JSON (untrained) | 832.8 ms | 1.2 | 16 | 0 of 60 | 0.833 |
| Qwen3-4B with thinking on (untrained) | 16,418 ms | 0.05 | 360 | 0 of 60 | 0.650 |
| **`spark-s1-4b-v3`** | **65.9 ms** | **15.1** | **0** | **0 of 60** | **0.850** |
| Qwen3-1.7B writing JSON (untrained) | 390.1 ms | 2.5 | 16 | 14 of 60 | 0.433 |
| Qwen3-1.7B with thinking on (untrained) | 7,586 ms | 0.11 | 354 | 0 of 60 | 0.550 |
| **`spark-s1-1.7b-v3`** | **29.6 ms** | **33.8** | **0** | **0 of 60** | **0.783** |

`spark-s1` is about **13x faster** than the same-size model writing JSON and about **250x faster** than the same model reasoning first. At 4B the zero-shot JSON model is about as accurate as `spark-s1` on this small set (0.833 vs 0.850, within the ±5-point noise), so the gain there is efficiency; at 1.7B `spark-s1` is both faster and much more accurate, because the small model often breaks the JSON format. Reading options out with no training gets 0.70 (4B) and 0.60 (1.7B), so training on decision data is what makes the fast readout accurate. **Caveat:** the baseline is an untrained model prompted zero-shot; a fine-tuned or much larger generative model would score differently, and the set is a small diagnostic. The Decision Lab shows this comparison with the real recorded outputs for each sample command.

<p align="center"><img src="docs/img/decision-lab-comparison.png" alt="Decision Lab: spark-s1 vs a generative small model" width="820" /></p>

---

## Current Limitations

This is an early release. The models were trained on a small amount of decision data, and the limits are visible in several task families:

- Agent next-action routing, urgency scoring, retrieval and termination gates, and answer sufficiency
- Vulnerable-code detection (about 0.56, close to chance)
- Boolean and Score calibration (only Choice has a fitted temperature)
- Outcome-based calibration on unseen data (RLCD has not been applied to the released models)
- Commands that weaken security settings, copy credentials out, or hide a cloud copy as a "backup"

The project should not be treated as a production-ready decision engine for high-stakes applications.

> [!WARNING]
> Never use a model decision as the only control before running a tool. Keep permission checks, least-privilege credentials, logging and human approval for sensitive actions. Anything low-confidence or unrecognised should go to a human, not run. The built-in rules (`open_spark_jev/policy.py`) are simple and incomplete: they block obvious cases, such as sending a secrets file over the network, and the model can never override them. See [SECURITY.md](SECURITY.md).

---

## Training

The training pipeline is included in the repository.

**Phase 1: supervised fine-tuning** (`open_spark_jev/train/sft.py`). Multi-task decision examples with soft targets, cross-entropy plus a Brier regulariser, and per-type temperature fitting. The pipeline can use simulator-generated data, public datasets, teacher-generated data and the code-labelled `os-datagen` data. **The released v3 models used only the `os-datagen` rows.**

**Phase 2: RLCD experiments** (`open_spark_jev/train/rlcd.py`). TypeSafe refers to its calibration-oriented approach as RLCD, but has not released the mechanism, so this repository implements and evaluates two independent approaches: an academic contrastive approach and a direct calibration objective, using the full menu distribution from a single forward pass rather than PPO/GRPO-style token sampling. The goal is to investigate whether outcome-aware training improves the relationship between confidence and correctness. **It has not been applied to the released v3 models**; our first attempt showed the objective needs training rows the model has not memorised. See [docs/RESEARCH.md](docs/RESEARCH.md), [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and `configs/train/`.

### Decision data factory

The repository includes **os-datagen** (`datagen-pipeline/`), a code-labelled decision-data factory: policy engines, symbolic solvers, controlled worlds and a sandbox establish the labels, and a language model only writes the surface text, which an independent verifier checks. The released training data is on Hugging Face: [`spark-s1-osdg-v1`](https://huggingface.co/datasets/abhishek085/spark-s1-osdg-v1). The goal is a reusable recipe, not one fixed dataset: [docs/DATA.md](docs/DATA.md), [docs/COOKBOOK.md](docs/COOKBOOK.md).

```text
Choose a backbone → Define decision tasks → Create domain simulators → Generate decision data
   → Grade / label examples → SFT → RLCD experiments → Calibration → Evaluation → Quantization → Local serving
```

---

## NVIDIA DGX Spark

Open Spark Jev is designed for local experimentation on NVIDIA DGX Spark. The repository includes scripts for environment setup, model download, data generation, SFT, RLCD, evaluation, TensorRT-LLM deployment, quantization, serving and smoke testing.

```bash
git clone https://github.com/abhishek085/open-spark-jev.git && cd open-spark-jev
scripts/setup_env.sh                       # repo-local venv with torch, transformers, TRL, PEFT
scripts/download_weights.sh Qwen/Qwen3-4B  # a backbone, to train your own
.venv/bin/python scripts/smoke_test.py
```

```bash
scripts/make_data.sh && scripts/train_sft.sh && scripts/train_rlcd_direct.sh   # train
scripts/eval.sh hf checkpoints/rlcd-direct-qwen3-1.7b                          # evaluate
deploy/spark/pull_trtllm.sh                                                    # TensorRT-LLM container
deploy/spark/quantize.sh configs/quant/fp8.yaml checkpoints/rlcd-qwen3-1.7b    # quantize (no engines released yet)
```

See [docs/DGX_SPARK.md](docs/DGX_SPARK.md) for Spark-specific configuration, containers, kernels, quantization and serving.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/gate` | Tool-call approval: `spark-s1` distribution plus deterministic rules, returns `allow` / `ask` / `deny` |
| `POST /v1/decide` | Typed questions over a state, with per-decision probabilities, confidence, margin, entropy and latency |
| `POST /v1/evaluate` | Jev-style wire format, so existing Jev client patterns can be tried against a local server |

```json
{
  "state": "...",
  "questions": {
    "routing": {"type": "choice", "instructions": "Which queue should handle this?", "criteria": {"billing": "charges, invoices", "technical": "bugs, outages", "account": "login, profile"}}
  }
}
```

Details, request shapes and Python usage: [docs/API.md](docs/API.md). Examples: `examples/` and [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

---

## Repository Structure

```text
open-spark-jev/
├── open_spark_jev/      # library: schema, prompting, model, calibration, gate, policy, data/, train/, eval/, serve/
├── datagen-pipeline/    # os-datagen: the code-labelled decision-data factory
├── configs/             # model, train, serve, quant
├── data/                # external evaluation suites with provenance (data/external/)
├── deploy/spark/        # TensorRT-LLM, quantization and gateway scripts
├── scripts/             # setup, download, data, training, evaluation and analysis scripts
├── examples/            # Python and HTTP examples
├── docs/                # architecture, benchmarks, run log, API, evaluation, UI, roadmap and more
├── tests/
├── MODEL_CARD.md  CONTRIBUTING.md  SECURITY.md  CHANGELOG.md  LICENSE  NOTICE
└── pyproject.toml
```

---

## Roadmap

- Larger and more diverse decision datasets, and more task families
- Better calibration across Choice, Score and Noul
- Outcome-based calibration training on unseen data
- Additional decision-head and per-option scoring experiments
- More extensive external evaluation, and a locked release holdout
- FP8 / NVFP4 inference and more local deployment configurations
- Domain-specific decision recipes

See [docs/ROADMAP.md](docs/ROADMAP.md), [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), and [docs/RUNS.md](docs/RUNS.md).

---

## Contributing

Contributions are welcome, especially new decision tasks, better evaluation datasets, calibration experiments, new simulators, DGX Spark optimizations, serving improvements, documentation, bug fixes and domain-specific examples. Wrong decisions are valuable too: use the **Safety failure** issue template. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/CONTRIBUTION_IDEAS.md](docs/CONTRIBUTION_IDEAS.md) before opening a pull request. If you experiment with the models and find something interesting, open an issue or share your results.

---

## License

The original source code, documentation, configurations, simulators and synthetic-data generators in this repository are licensed under Apache-2.0. See [LICENSE](LICENSE).

Third-party components, datasets, models and evaluation suites retain their respective licenses and terms. See [NOTICE](NOTICE). The Qwen3 backbones are downloaded separately and are not redistributed by this repository (the released `spark-s1` weights are Apache-2.0 derivatives hosted on Hugging Face).

---

## Disclaimer

Open Spark Jev is an independent open-source research and engineering project.

- It is not Jev, and it does not reproduce TypeSafe's proprietary implementation.
- It is not affiliated with or endorsed by TypeSafe, Kev's author, or NVIDIA.
- NVIDIA, DGX Spark and TensorRT-LLM are NVIDIA products and trademarks. TypeSafe, Jev and System One are TypeSafe AI's products and names.

Performance numbers attributed to Jev in this repository come from third-party recorded results; they are not claims of testing performed by TypeSafe. The models are experimental and should be independently evaluated before being used in production or high-stakes decision systems.

---

## Star the Project

If you find the idea, implementation or experiments useful, consider giving the repository a ⭐. It helps more people discover the project and build a community around open, local decision models.

GitHub: https://github.com/abhishek085/open-spark-jev · Models: [spark-s1-1.7b-v3](https://huggingface.co/abhishek085/spark-s1-1.7b-v3) · [spark-s1-4b-v3](https://huggingface.co/abhishek085/spark-s1-4b-v3)
