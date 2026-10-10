# Featherless onboarding request (draft)

Send to: Featherless `#model-suggestions` Discord channel, or their support email (see https://featherless.ai/docs/model-compatibility).
Why: Featherless serves public HF fine-tunes of supported architectures (Qwen3.5 4B is listed), is a Hugging Face Inference Provider, and is
the realistic route to an OpenRouter listing (OpenRouter only lists models a provider serves).

Before sending, confirm in their API docs that `/v1/completions` returns `logprobs` / `top_logprobs`; the model is only useful with them.
Note their FP8-at-boot quantisation: calibration under FP8 has not been measured.

---

Hi Featherless team, could you onboard **abhishek085/spark-s1-4b-v6**?
https://huggingface.co/abhishek085/spark-s1-4b-v6

- Architecture: `Qwen3_5ForCausalLM`, a full-weight fine-tune of `Qwen/Qwen3.5-4B` (LoRA merged), bf16 safetensors, Apache-2.0, model card present.
- It already has 137 downloads but does not appear on featherless.ai yet.
- It is a decision model: callers read the first-token `top_logprobs` over option letters (`max_tokens: 1`), so logprobs support on
  `/v1/completions` matters more than generation speed.

Thanks!
