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

## 1. Backbone
`Qwen/Qwen3-1.7B` (dense, ~1.4B non-embedding). Reasons: HF-native, Qwen3 is a
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
