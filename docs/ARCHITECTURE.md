# Architecture

open-spark-Jev is a **System One** decision model: it never generates text. It reads a
*state*, appends a *question* with a closed menu of answers, and returns a probability
distribution over that menu from a single forward pass.

```
                 ┌───────────────────────────────────────────────┐
  State ───────► │ Qwen3-1.7B decoder  (prefix run once, KV kept) │
                 └───────────────┬───────────────────────────────┘
                                 │ shared KV cache
      ┌──────────────────────────┼──────────────────────────┐
      ▼                          ▼                          ▼
  Q1 suffix                  Q2 suffix                  Q3 suffix
  "...Options: A. B. C."      "...Levels: A. B. C. D."    "Claim ... A. Yes B. No"
  + <|im_start|>assistant     + <|im_start|>assistant     + <|im_start|>assistant
    <think></think>             <think></think>             <think></think>
      │                          │                          │
  first-token logits         first-token logits         first-token logits
  gather {"A","B","C"}       gather {"A".."D"}          gather {"A","B"}
  ÷ T_choice, softmax        ÷ T_score, softmax         ÷ T_noul, softmax
      │                          │                          │
  Choice answer              Score answer               Noul answer
  (dist, argmax, conf)       (dist, E[value], conf)     (P(true))
```

## 0. The two current models: spark-s1-4b-v3 and spark-s1-1.7b-v3

Everything below describes the mechanism. This section says what the two best checkpoints actually are, and how they relate to Jev and to an ordinary small language model (SLM).

**What they are.** A public Qwen3 decoder (4B and 1.7B), fine-tuned with LoRA r16 (merged into the weights) for 3 epochs. There is no new head, no new layer and no custom kernel. The training data is
the os-datagen decision-data factory (`datagen-pipeline/`): 967 rows whose labels come from code (policy engines, solvers, controlled worlds, a sandbox), not from an LLM judge, with the LLM used only
to phrase the scenario. Each choice row is also shown in 4 random option orders so the model cannot rely on option position (4,277 training examples). Loss is cross-entropy plus a Brier regulariser.
A separate calibration split fits one temperature per question type; a locked test split and an out-of-distribution challenge split are only measured.

**How one decision runs.** The state is rendered once inside hard fences and run through the model. Each question is appended as a menu (`A. ... B. ...`, with the option definitions in the prompt) and the
logits of the first answer token are restricted to the option-letter tokens, divided by the calibrated temperature and softmaxed. That distribution is the answer; `confidence`, `margin` and `entropy` are
derived from it. Nothing is generated or parsed. Several questions on one state reuse the state's cached prefix, which is why a request with 2 questions took ~127 ms and one with 3 took ~135 ms (B30).

| | spark-s1 (v3-4B / v3-1.7B) | Jev (from public descriptions) | Ordinary SLM prompted to answer |
|---|---|---|---|
| Question interface | typed Choice / Boolean / Score over runtime-defined options | same idea | free text, usually JSON by prompt |
| Answer | full probability distribution from token logits, one forward pass | calibrated distribution and confidence, no generated text | generated text; "confidence" is a number the model writes |
| Output failures | none by construction (closed answer space) | none by construction | malformed JSON: 14 of 60 rows for the same 1.7B backbone, 0 of 60 at 4B (B33) |
| Latency, 60-row set | 65.9 ms (4B), 29.6 ms (1.7B) | 421.6 ms recorded, hosted, includes network | 390 ms writing JSON and 7,586 ms reasoning first (1.7B); 833 ms and 16,418 ms (4B) (B33) |
| Accuracy, 60-row set | 0.850 (4B), 0.783 (1.7B) | 0.917 (recorded) | 0.433 JSON, 0.550 with reasoning (1.7B, untrained); 0.833 JSON, 0.650 with reasoning (4B, untrained); untrained direct-logit 0.60-0.70 (B33) |
| Sensitive to option order | ~5% of os-datagen rows change answer (v3), ~50% for untrained 0.6B/1.7B | not published | not measured |
| Probabilities calibrated on held-out data | yes for Choice (ECE 0.087 at 4B); Boolean/Score not yet fitted | stated goal (RLCD) | no |
| Trained how | supervised fine-tune of a public backbone (CE + Brier) + post-hoc temperature | undisclosed (TypeSafe calls it RLCD) | not trained for this task |
| Weights / data | open | closed, hosted | open |

**Where we are like Jev.** The contract and the mechanism: typed questions with runtime-defined options in, a distribution and a confidence out, from one forward pass with no decoding, cheap enough to sit in
an agent loop, with calibration learned on held-out data and then checked on a locked test.

**Where we are not.** (1) Training: we fine-tune a public backbone with a supervised loss. Jev's method is not public, so we do not claim to reproduce it. (2) We have not applied outcome-based calibration training
(RLCD): the attempt on v3-4B (v7) had no gradient because the model had memorised its ~1k training rows, so calibration is a post-hoc temperature only. (3) Data scale: 967 rows across 15 packs, against decision models
that need far more; the weak packs (router, urgency, retrieval, termination, answer sufficiency) show it. (4) Accuracy: 0.850 vs Jev's recorded 0.917 on the 60-row set, and vulnerable-code detection is near chance (0.56).
(5) Speed comparisons against Jev are against a hosted, network-inclusive recorded latency and are not like-for-like.

**Where we are not an ordinary SLM.** The model is used as a classifier over a closed menu, never as a generator. Compared with the same untrained backbone, training on code-labelled decisions moved the 60-row
accuracy from 0.60 to 0.78 at 1.7B and from 0.70 to 0.85 at 4B, and made answers stable when the option order changes.

**Trade-off between the two.** The 4B is the accuracy pick (Kev transfer 0.749, directory 0.771); the 1.7B is the speed pick (14x faster than Jev's recorded latency, but uneven transfer: directory 0.557).
Numbers: [BENCHMARKS.md](BENCHMARKS.md) B27, B29, B30, B31.

## 1. Backbone
`Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B` (dense; the 1.7B non-embedding size is ~1.4B). Reasons: HF-native, Qwen3 is a
first-class TensorRT-LLM PyTorch-backend model with FP8/NVFP4 recipes, and the 1.7B size
fits Spark's latency budget with headroom for a co-resident teacher. `Qwen3-0.6B` is
the smoke-test size; `Qwen3-4B` is the accuracy-ceiling probe (see RESEARCH.md, Exp. 5).

## 2. Menu scoring = restricted LM head
The "menu head" is the backbone's own LM head restricted to ≤26 rows (bare `A`..`Z`, each a
single Qwen token). The answer is the **first assistant token** after the non-thinking header.
Consequences:

* **Zero extra parameters, zero custom kernels, zero custom server.** Any OpenAI-compatible
  chat endpoint that returns `top_logprobs` serves decisions unchanged. `serve/client.py`
  does this with `/v1/chat/completions` + `max_tokens=1` + `logprobs` +
  `chat_template_kwargs={"enable_thinking": false}`; verified on `trtllm-serve` 1.2.1 (whose
  `/v1/completions` does *not* accept `logprobs`). `serve/trtllm_backend.py` uses the TRT-LLM
  Python API in-container for full-vocabulary first-token logits when a menu exceeds the
  endpoint's `top_logprobs` cap (20).
* **Structured-output error rate is 0 by construction.** The answer space is closed; there is
  nothing to parse.
* **One prompt format for training and serving.** `prompting.render_prompt` is byte-identical
  to `apply_chat_template(render_messages(...), add_generation_prompt=True, enable_thinking=False)`
  (unit-tested), so HF training and every serving path see the same tokens.

An optional learned linear head (`aux_head: true`) exists purely as an ablation.

## 3. KV-cache reuse
`model.py` runs the prefix once and expands the `DynamicCache` across the question batch,
so N questions on one state cost one prefix pass plus N short suffix passes (~60-120 tokens
each). Served through TRT-LLM the same effect comes from `kv_cache_config.enable_block_reuse`
(prefix caching): the gateway sends all questions for a state back-to-back.

## 4. Calibration
* Training minimises cross-entropy against **soft targets** where available (simulator
  posteriors, teacher distributions) plus a Brier regulariser.
* A per-question-type temperature is fitted on validation data and stored in
  `calibration.json`; the gateway applies it. Re-fit after quantization.
* Metrics: ECE, Brier, NLL, reliability bins; and **soft Brier vs. the true posterior** on
  simulator data, which is the metric that single-label datasets cannot provide.

## 5. Abstention and safety
`allow_abstain` adds a menu slot. The simulators put target mass on it when the posterior is
flat; `rewards.abstain_reward` and `conservative_reward` shape it in Phase 2. Thresholding a
calibrated `Noul` is the recommended pattern for act / call-LLM / escalate gating
(`examples/agent_gate.py`).

## 6. Prompt-injection posture
* State is fenced (`<<<STATE ... STATE>>>`), close-fence sequences inside it are neutralised.
* System prompt declares state as untrusted data.
* The closed answer space bounds the blast radius to *biasing a distribution*.
* Training data contains injected twins with unchanged targets; `injection_reward` scores
  agreement between injected and clean decisions; `eval/benchmark.py` reports the flip rate.

## 7. Module map
| path | role |
|---|---|
| `open_spark_jev/schema.py` | State / Choice / Score / Noul / Answer (pydantic) |
| `open_spark_jev/prompting.py` | prefix/suffix rendering, label token space |
| `open_spark_jev/model.py` | HF MenuScorer with state-cache reuse and training forward |
| `open_spark_jev/calibration.py` | ECE / Brier / NLL / reliability / temperature fitting |
| `open_spark_jev/data/` | corpus format, known-posterior simulators, teacher synth, public adapters |
| `open_spark_jev/train/sft.py` | Phase 1 supervised loop |
| `open_spark_jev/train/rlcd.py` | Phase 2: pairs → reward model → exact menu policy gradient |
| `open_spark_jev/train/rlcd_grpo.py` | sampled-token GRPO baseline (TRL) |
| `open_spark_jev/eval/` | decision benchmark + Spark latency grid |
| `open_spark_jev/serve/` | gateway (`/v1/decide`, Jev-style `/v1/evaluate`) and backends |
| `deploy/spark/` | TRT-LLM container, quantization, `trtllm-serve`, gateway |
