# Cookbook: training your own Jev-like System One decision model

This is a reproducible, start-to-finish recipe for building a calibrated, menu-scoring
decision model like open-spark-Jev **for your own domains**, reusing as much of this repo's
code as possible. It assumes no prior context beyond general familiarity with PyTorch,
Hugging Face Transformers, and running a local LLM server. Where a step is specific to this
project's exact choices (Qwen3-1.7B, DGX Spark, these six simulator domains), that is called
out so you know what to swap.

If you just want to run *this* repo's pipeline as-is, see the [README](../README.md)
quickstart instead. This document is for adapting the strategy to different domains, a
different backbone, or different hardware.

## 0. What you are building

A decoder-only LM that answers exactly three question shapes about an arbitrary input
*state* (text, JSON, logs, traces):

* **Choice** - pick one of N options, get a full probability distribution.
* **Score** - place the state on an ordered rubric, get a distribution + expected value.
* **Noul** - a calibrated P(claim is true).

The model never free-generates. Every answer is the softmax of a handful of restricted
next-token logits (the label letters). This is what makes the model fast, impossible to
produce malformed output, and servable by any inference engine that returns logprobs -
including engines with no custom kernels for this use case. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full mechanism.

## 1. Prerequisites

* A CUDA GPU with enough memory for your chosen backbone (bf16 full fine-tune of a 1-2B model
  needs roughly 8x its parameter count in GB during training: gradients + optimizer states +
  activations; a 1.7B model trains comfortably in 30-40GB).
* Python 3.10+, PyTorch matching your CUDA version, Transformers >= 4.51, TRL >= 0.24, PEFT.
* Optionally: Docker, for serving via vLLM/TensorRT-LLM/SGLang and for running teacher models
  as containers instead of pip installs (this repo's convention throughout, see
  [DGX_SPARK.md](DGX_SPARK.md)).
* One or more **teacher models** for synthetic data generation (Section 4). They can be
  anything you can serve behind an OpenAI-compatible `/v1/chat/completions` endpoint - a local
  vLLM/SGLang server, or a hosted API if you are not restricted to fully local operation.

## 2. Get a backbone

Any dense, HF-native, decoder-only causal LM works, subject to one hard constraint checked at
`MenuScorer.__init__` time: your label tokens (`prompting.LabelSpace.build`) must each
tokenize to exactly one token. Bare capital letters (`A`, `B`, ...) are single tokens in
essentially every modern BPE tokenizer, so this is rarely a real constraint - but always
verify it for a new tokenizer, since `LabelSpace.build` raises immediately if not.

Sizing guidance: pick the smallest backbone that clears your accuracy bar. Start at ~0.5-1B
for a smoke test (`Qwen/Qwen3-0.6B` in this repo), and go up (1.7B, 4B, ...) only if your
Phase 1 numbers plateau below what you need - see the size/latency/calibration trade-off
experiment in [RESEARCH.md](RESEARCH.md) R5.

```bash
scripts/setup_env.sh                          # adapt: this creates a repo-local venv
scripts/download_weights.sh <your/backbone-id>
```

## 3. Define your domains and questions

Decide what decisions you need and express each as a `Choice`, `Score`, or `Noul`
(`open_spark_jev/schema.py`). A few rules that keep the model easy to train and calibrate:

* Keep menus small (<=10-15 options is comfortable; the hard ceiling is 26, one per letter).
* Add `allow_abstain=True` wherever a confidently wrong answer is worse than saying "not
  enough information" - the abstain slot is a first-class label the model can learn to use,
  not a post-hoc threshold on confidence.
* For `Score`, prefer named `levels` over a numeric `band` when the levels have real
  semantic differences (a `Score` with a `rubric` string trains better than a bare numeric
  scale, because the rubric text gives the model something to condition on).

## 4. Build a known-posterior benchmark (do this before touching real data)

This is the single most reusable idea in this repo and the one most worth copying verbatim:
**write a small generative model of your domain before you write a classifier for it.**

`open_spark_jev/data/simulators.py` implements each domain as a naive-Bayes generator: a
latent label, a handful of conditionally-independent observable features, and a `render()`
function that turns sampled features into a state. Because you wrote the generative process,
you know the *exact* Bayes posterior for every sample - which means you can measure
calibration against ground truth (soft Brier, ECE) instead of against one noisy realized
label, which is all a normal dataset gives you.

To adapt: copy one function from `simulators.py` (e.g. `_risk()`) as a template. For your
domain, define:
1. `labels` - your answer options.
2. `features` - each a `(possible_values, {label: probability_table})` pair.
3. `render(features, rng)` - turn sampled features into a realistic-looking state.

Then `generate(domain, n, ...)` gives you records with `target.dist` (the true posterior) and
`target.label` (a label *sampled* from that posterior - so even the "hard" label reflects
realistic ambiguity, not an oracle). Also get abstention targets for free (posterior entropy
above a threshold) and prompt-injected twins with unchanged targets (`inject_frac`), which
turn "is my model manipulable via its input" into a measurable number
(`eval/benchmark.py`'s `injection_flip_rate`).

Budget 1-2 hours per domain to write a good simulator; it pays for itself immediately once you
have real calibration numbers instead of only accuracy.

## 5. (Optional) Public datasets

If your domain overlaps existing benchmarks, add an adapter in `data/public.py` - each is
~10-15 lines mapping a HF `datasets` split onto one of your questions. Real language data adds
lexical diversity the simulators can't; it does not give you known posteriors, so keep it as a
supplement, not your only training signal for calibration.

## 6. Multi-teacher synthetic data generation

Synthetic scenarios authored by a strong local LLM fill the gap between hand-built simulators
(structured, exhaustive, but templated) and public datasets (natural language, but generic).
Using **more than one teacher** matters for two independent reasons, and this repo's
`data/synth.py` + `eval/teacher_benchmark.py` + `configs/teachers.yaml` support both:

**(a) Authoring diversity.** Different teachers write different scenarios for the same hidden
label - different vocabulary, structure, and edge cases. Tag every record by its source
teacher (`meta`/`source` already do this) so you can later ablate "trained with teacher X's
data only" vs "trained with the pooled corpus" and check whether diversity actually helped
your student, rather than assuming it did.

**(b) Label quality.** Not every teacher is equally well-calibrated as a *grader*, and you
should not just trust one. `eval/teacher_benchmark.py` answers this with ground truth: run
each candidate teacher's `label_distribution()` over your known-posterior benchmark
(Section 4) and score it with the exact same soft-Brier/ECE metrics you'll use to score the
student. This turns "which teacher should I distill from" into a measured decision instead of
a guess, and gives you an honest floor: no student trained by distillation should be expected
to beat its teacher's own calibration on the same data.

### Recipe

1. **Register your teachers** in a file like `configs/teachers.yaml`: name, HF model id,
   `base_url` of an OpenAI-compatible server, and (important on a shared or memory-constrained
   box) a `max_concurrency` and a `mem_preflight_gb`. Copy the pattern in this repo's
   `configs/teachers.yaml`, which lists three teachers from three different labs
   (`nvidia/Qwen3.6-27B-NVFP4`, `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`,
   `openai/gpt-oss-120b`) specifically to test whether teacher choice moves the needle, not
   because any one of them is required.

2. **Launch teachers one at a time if memory is tight.** Large teachers (100B+ class MoE
   models) typically need 60-70GB of memory even quantized; on a single unified-memory box you
   usually cannot run two at once alongside anything else. `scripts/teachers/_common.sh`'s
   `mem_preflight` function checks `/proc/meminfo`'s `MemAvailable` before every launch and
   refuses rather than risking an OOM of something else already running - copy this pattern
   for your own box rather than skipping the check "just this once."

3. **Grade your known-posterior benchmark with every teacher**:
   ```bash
   python -m open_spark_jev.eval.teacher_benchmark \
     --teachers teacher_a teacher_b teacher_c \
     --data data/benchmarks/sim_test.jsonl --limit-per-domain 40 \
     --out runs/teacher_benchmark.json
   ```
   Unreachable teachers are skipped with a warning, not a crash, so you can re-run this
   incrementally as each teacher comes online. It also scores an equal-weight **ensemble** of
   whichever teachers answered, which is usually a safe default target distribution if no
   single teacher clearly dominates.

4. **Author scenarios and distill**, per teacher, into separate files so provenance is never
   lost:
   ```bash
   TEACHER_BASE_URL=http://localhost:PORT/v1 TEACHER_MODEL=<hf id> \
     osj synth --n 300 --distill --out data/synthetic/teacher_<name>.jsonl
   ```
   `distill()` grades every authored scenario blind and flags `meta.suspect=True` when the
   teacher's own probability on the *hidden* label it was asked to write for falls below
   `min_agreement` - exclude these from training; they are either bad scenarios or genuinely
   ambiguous ones worth a manual look, not free training signal.

5. **Decide your training mix** using the numbers from step 3, not vibes: weight or filter
   per-teacher data by that teacher's measured soft Brier, or just pool everything and let the
   Brier-regularized soft-target loss (Section 7) do the averaging implicitly. Record whichever
   choice you make and why - it is the kind of decision that is easy to forget and hard to
   reconstruct later.

## 7. Phase 1: supervised menu-head training

`train/sft.py` is close to boilerplate multi-task classification, with two deliberate choices
worth keeping:

* **Soft targets wherever you have them** (simulator posteriors, teacher distributions,
  ensemble distributions from step 6). The loss is cross-entropy against whatever
  `target.dist` gives you, falling back to one-hot CE only when a record has just a hard
  label. Training against soft targets directly optimizes calibration, not just accuracy.
* **A Brier regularizer on top** (`lambda_brier` in `configs/train/sft.yaml`) further
  discourages overconfidence beyond what cross-entropy alone penalizes.
* **Temperature fitting per question type** on a held-out split, saved to
  `calibration.json` and applied at inference. This is a five-line addition
  (`calibration.fit_temperature`) that meaningfully closes the ECE gap; do not skip it.

Adapt `configs/train/sft.yaml`'s `data:` list to point at your own corpora; everything else
(collation, length bucketing, LoRA-or-full-finetune switch) is domain-agnostic.

## 8. Phase 2: RLCD (optional but where calibration really tightens)

`train/rlcd.py` implements three stages:

1. **Contrastive pairs** - prompt one of your teachers twice per (state, question), once with
   principles favoring calibration/abstention/state-as-data, once with principles favoring
   overconfidence/instruction-following/automation. No human labels needed (this is the RLCD
   trick: contrast comes from the prompt, not from paired human annotations).
2. **A reward model that is itself a Noul** - "is this proposed decision correct, safe,
   appropriately confident, and uninfluenced by injected instructions?" - trained with a
   standard Bradley-Terry pairwise loss on the contrastive pairs. Reusing the same
   architecture means the RM inherits the calibration machinery for free and can be served the
   same way.
3. **An exact menu policy gradient**, not PPO/GRPO. Because the action space is a handful of
   labels, the full policy distribution is available in one forward pass, so
   `E_a~p[R(a)] + R_dist(p) - beta*KL(p||p_ref)` is differentiated directly - zero sampling
   variance, and critically, it lets the reward include **proper scoring rules evaluated on
   the distribution itself**, which sampled-action RL cannot see. `train/rlcd_grpo.py` is kept
   as a TRL-based sampled-token baseline specifically so you can verify this matters on your
   own data (see [RESEARCH.md](RESEARCH.md) R1) rather than take it on faith.

`train/rewards.py`'s composable terms (Brier, abstention, "conservative action" asymmetric
penalty, injection consistency, RM score) are domain-agnostic; the one thing to set per domain
is `SAFE_LABELS` - which option, if any, is the "when in doubt, do this" conservative choice
your conservative_reward should protect.

## 9. Evaluate

Always score against your known-posterior benchmark, not just accuracy on realized labels:

```bash
python -m open_spark_jev.eval.benchmark --backend hf --model <checkpoint> \
  --data data/benchmarks/<your_domain>_test.jsonl --out runs/eval.json
```

Track, per question type and domain: accuracy, macro-F1, Brier/NLL/ECE, soft Brier vs the true
posterior, and the prompt-injection flip rate. `eval/latency.py` gives you the
latency/throughput grid once you care about serving.

## 10. Quantize and serve

`deploy/spark/quantize.sh` wraps NVIDIA ModelOpt's PTQ script for FP8/NVFP4; the general
lesson (true for any quantization toolchain, not just ModelOpt) is: **calibrate on your
decision prompts, not on generic text**, and **re-fit your temperatures on the quantized
engine** - quantization shifts logit magnitudes enough to measurably move ECE even when
accuracy looks unchanged (see [DGX_SPARK.md](DGX_SPARK.md)).

For serving, the only real constraint is: your inference engine's OpenAI-compatible endpoint
must expose next-token `logprobs`/`top_logprobs` on whichever primitive (`/v1/completions` or
`/v1/chat/completions`) it supports, and your prompt must match your training-time chat
template exactly, including whether the answer token is a prefix continuation or the model's
first free token (this changed the prompt convention mid-project here - see
[DGX_SPARK.md](DGX_SPARK.md)'s "Serving-path facts" and the git history around it for a worked
example of debugging this).

## 11. Reusability matrix

| file / module | reuse as-is | reuse with edits | rewrite for your domain |
|---|---|---|---|
| `schema.py`, `prompting.py` | yes | | |
| `model.py` (MenuScorer) | yes | swap backbone id | |
| `calibration.py` | yes | | |
| `data/corpus.py` (record format) | yes | | |
| `data/simulators.py` | | keep the *pattern* | yes - your domains |
| `data/public.py` | | add adapters | |
| `data/synth.py`, `eval/teacher_benchmark.py` | yes | point at your teachers/registry | |
| `train/sft.py`, `train/rlcd.py`, `train/rewards.py` | mostly | `SAFE_LABELS`, config paths | |
| `eval/benchmark.py`, `eval/latency.py` | yes | | |
| `serve/*` | yes | model paths | |
| `configs/*` | | yes, this is the point | |

## 12. Gotchas we hit, so you don't have to

* **Chat template drift breaks the served-vs-trained parity silently.** We initially trained
  against an explicit `Answer:` continuation, then found our TensorRT-LLM server's chat
  endpoint does not continue assistant-message prefills. Fix: make the answer token the
  model's first free assistant token, and unit-test that your rendered prompt string equals
  `tokenizer.apply_chat_template(..., add_generation_prompt=True)` byte-for-byte
  (`tests/test_prompting.py::test_plain_render_matches_chat_template`). Do this check before
  training, not after.
- **`top_logprobs` is capped** (20 on the engine we used) - fine for menus up to ~15-20
  options; beyond that, use an in-process/in-container backend with full-vocabulary logits
  (`serve/trtllm_backend.py`) rather than the OpenAI-compatible endpoint.
- **A shared inference server is a shared *failure domain*.** Concurrent grading requests
  against another project's long-running teacher triggered a latent vLLM/flashinfer crash in
  our run. If you reuse someone else's server as a teacher, keep concurrency low
  (`max_concurrency` in the registry) and know how to restart it from its owner's own script,
  not a container you invent.
- **Root-owned shared caches.** If your box has a shared, root-owned model cache directory you
  can read but not write, point new downloads at your own `HF_HOME` rather than fighting for
  write access.
- **Pin your RL library version expectations.** TRL's `GRPOConfig` fields moved between minor
  versions during this project; filter kwargs by `inspect.signature` rather than hardcoding a
  field list if you want the script to survive an upgrade.
- **Unified-memory boxes lie about "free" memory.** Use `/proc/meminfo`'s `MemAvailable`, not
  `MemFree` or `nvidia-smi`'s (frequently `N/A` on some platforms) memory field, for any
  preflight check before launching a large model.

## 13. Minimal path

If you want the smallest possible end-to-end proof before investing in all of the above: one
domain, one `Choice` question, the known-posterior simulator for it, no public data, no
teacher, straight to `train/sft.py` with `lambda_brier: 0` and no LoRA, then
`eval/benchmark.py`. That loop is a few hours end-to-end on a single consumer GPU and tells you
immediately whether the mechanism (schema -> prompting -> restricted-logit scoring -> soft-CE
training) is wired correctly before you spend time on domains, teachers, or RL.
