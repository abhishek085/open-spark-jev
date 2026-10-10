# Decision Index: run request for spark-s1-4b-v6 (draft)

Target: https://github.com/apolinario/decision-index/issues/new

**Heads-up before posting.** The maintainers have so far declined run requests and asked entrants to self-run and open a PR
(#10, #38; #6 was withdrawn for the same reason). The alternative that needs no local hardware is their own `hf-job` command (one RTX PRO 6000
job on Hugging Face, billed to the submitter's HF account), followed by the results PR. Fill in `<COMMIT>` with a pushed commit of
open-spark-jev before posting.

---

**Title:** Run request: spark-s1-4b-v6 (Qwen3.5-4B LoRA, one-prefill option-letter readout, native `/v1/systemone`, Apache-2.0)

We'd like **spark-s1-4b-v6** added to the Decision Index. We have not run the suite ourselves. If you would rather we self-run with
`hf-job` and open a results PR, say so and we will.

**Model:** https://huggingface.co/abhishek085/spark-s1-4b-v6 at
[`93d49dd`](https://huggingface.co/abhishek085/spark-s1-4b-v6/tree/93d49ddbfb29212e3296635a75a3e80cf69da027). `Qwen/Qwen3.5-4B`
with a LoRA (r=16) merged into full bf16 weights, text-only. Supervised fine-tune on 19,568 decision examples (49 task packs, randomized option
order). One prefill per question; the answer is the softmax of the first-token logits restricted to the option letters, divided by a
calibrated temperature shipped in `calibration.json` (Choice T = 1.481). No tokens are generated and there is no thinking.

**Serving (your `http` engine against our native `/v1/systemone`):**

```
git clone https://github.com/abhishek085/open-spark-jev && cd open-spark-jev && git checkout <COMMIT>
python3 -m venv .venv && . .venv/bin/activate && pip install torch -e ".[infer]"
osj serve --pull spark-s1-4b-v6 --port 8400        # in-process HF Transformers backend
python -m decision_index pipeline --engine http --option base_url=http://127.0.0.1:8400 --out runs/spark-s1-4b-v6
```

Faster path (what we use in production): vLLM serving the same weights, with the gateway in front doing the letter readout:

```
vllm serve abhishek085/spark-s1-4b-v6 --revision 93d49ddbfb29212e3296635a75a3e80cf69da027 --port 8355 --max-model-len 16384
hf download abhishek085/spark-s1-4b-v6 calibration.json --local-dir spark-s1-v6
python -m open_spark_jev.serve.gateway --backend openai --upstream http://127.0.0.1:8355/v1 --model spark-s1-v6 --port 8400
```

**Capacity, declared up front:**

- Options: up to 26 per question (letters A-Z). The vLLM path reads OpenAI-style `top_logprobs` (max 20); with more than 20 options use
  the in-process HF backend (`osj serve`), which gathers all label logits.
- Context: the gateway truncates states to 12,000 characters by default (the training distribution). For your no-truncation rule, start it
  with `--max-state-chars 1000000` and set `--max-model-len` to what the GPU allows; requests beyond that are rejected (HTTP 413), not truncated.
- Question types: Choice, Score and Boolean (Noul).

**Latency (our hardware, for reference only):** p50 74.9 ms per question with vLLM bf16 on one DGX Spark (GB10); HF Transformers in-process
is slower (~120 ms). We expect well under your 1,000 ms limit on an RTX PRO 6000.

**What we have measured** (not Decision Index numbers): own locked test 0.929, JevBench public tiers (easy/standard/hard)
1.000/1.000/0.595. Details: https://huggingface.co/abhishek085/spark-s1-4b-v6

Thanks for running the board.
