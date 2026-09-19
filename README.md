# open-spark-Jev

**An open, local System One decision model for NVIDIA DGX Spark.**
Qwen3-1.7B backbone · single-pass menu scoring · calibrated Choice / Score / Noul answers ·
RLCD training · served through TensorRT-LLM behind a typed JSON API.

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
full-vocabulary logits. Verified against trtllm-serve 1.2.1. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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

Data: [docs/DATA.md](docs/DATA.md). Training-mechanism experiments: [docs/RESEARCH.md](docs/RESEARCH.md). Architecture-level experiments (including our own proposed novel decision-head design): [docs/NOVELTY.md](docs/NOVELTY.md).

**Want to build your own domain-specific version of this, not just run this one?**
[docs/COOKBOOK.md](docs/COOKBOOK.md) is a start-to-finish, reusable recipe: pick a backbone, write known-posterior simulators for your domains, generate and grade synthetic data with multiple teachers, run SFT + RLCD, evaluate, quantize, serve - with a reusability matrix showing what ports over as-is versus what you rewrite.

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

deploy/spark/pull_trtllm.sh                   # TensorRT-LLM container (arm64, CUDA 13)
deploy/spark/quantize.sh configs/quant/fp8.yaml checkpoints/rlcd-qwen3-1.7b
deploy/spark/serve.sh engines/rlcd-qwen3-1.7b-fp8     # trtllm-serve :8355
deploy/spark/gateway.sh checkpoints/rlcd-qwen3-1.7b   # System One API :8400 (chat logprobs path)
#   or: deploy/spark/gateway_trtllm.sh <model>          # gateway inside the container, TRT-LLM Python API
deploy/spark/smoke_curl.sh
```
Spark specifics (container tags, kernels, memory, quantization + temperature re-fit):
[docs/DGX_SPARK.md](docs/DGX_SPARK.md).

## API
`POST /v1/decide` (native, see `schema.DecisionRequest`) and `POST /v1/evaluate`, a
Jev-style wire format (`{"state", "questions": {name: {type, instructions, criteria}}}`) so
existing Jev client code can be pointed at a local endpoint. `examples/` has routing,
a calibration-aware agent gate, and a Jev-format client.

## Evaluation
`python -m open_spark_jev.eval.benchmark` reports accuracy, macro-F1, Brier, NLL, ECE per
(question type × domain), soft Brier against the **known posterior** on simulator data,
Noul-specific Brier/ECE, and the prompt-injection flip rate. `eval/latency.py` produces the
Spark latency/throughput grid over state length × questions per state.

## Status
Scaffold complete and smoke-tested on a GB10 (HF backend, bf16). See
[docs/ROADMAP.md](docs/ROADMAP.md) for what is done, what runs next, and open questions.

## Layout
```
open_spark_jev/   schema · prompting · model · calibration · data/ · train/ · eval/ · serve/ · cli
configs/          model / train (sft, rlcd, rlcd_grpo) / serve (trtllm options, gateway) / quant (fp8, nvfp4)
deploy/spark/     pull_trtllm · quantize · serve · gateway · smoke_curl
scripts/          setup_env · download_weights · make_data · train_* · eval · smoke_test
docs/             ARCHITECTURE · DGX_SPARK · DATA · RESEARCH · ROADMAP
tests/            CPU unit tests + GPU smoke test (pytest -m gpu)
```

## License
Apache-2.0. Backbone weights: Qwen3 (Apache-2.0).
