# Novelty ledger: architecture experiments

Purpose: TypeSafe has disclosed a training *objective* for Jev (RLCD) and a one-line
architecture claim ("a new model architecture, parallel sampler... generates all outputs in a
single query") with no mechanism behind either. `docs/RESEARCH.md` R1 tracks the training-side
experiments (four Phase-2 mechanisms competing on the same objective). **This document tracks
the separate, architecture-side question**: is the plain "restrict the LM head to a few
tokens" mechanism (`docs/ARCHITECTURE.md`) the right shape for a decision model at all, or can
a different architecture do better on the same calibration/latency budget?

This is a living ledger, not a one-time writeup. Every entry gets a status and, once measured,
real numbers - update it in place rather than writing a new "part 2" doc. An idea stays
`proposed` until there is a runnable prototype, `implemented` until it has actually been run,
and `measured` only once it has comparable numbers against the baseline in the table at the
bottom.

## What's actually unknown about Jev's architecture

From `typesafe.ai/blog/introducing-system-one-models-and-jev` (the only primary source; see
`docs/ARCHITECTURE.md` and the RLCD naming discussion in `train/rlcd.py`'s module docstring):
"a new model architecture, parallel sampler for maximum efficiency... generates all outputs in
a single query" - contrasted implicitly with token-by-token generation. No architecture
diagram, no parameter count, no statement about whether it's a transformer at all. Three
readings are all consistent with that one sentence:

1. It **is** a transformer, and "single query" means N questions on one state are answered by
   one forward pass with N readout positions (batched decoding, not sequential decode calls).
2. It **is** a transformer, but with a non-causal or partially-causal attention pattern more
   suited to reading a whole state at once than a causal decoder is.
3. It is **not** primarily a next-token-prediction transformer at all - e.g. a set-prediction
   / query-based head (DETR/Perceiver-style) over a shared encoder, which would make "single
   query" literal: one encoder pass, N learned queries, N simultaneous answers.

We can't confirm which. What we *can* do is build and measure all three against our current
baseline (call it **A0**, the mechanism in `model.py` today) on the same calibration suite
already built for the training-side experiments (`eval/benchmark.py`,
`eval/calibration_baseline.py`'s metrics, `eval/latency.py`). That turns "what might Jev be
doing" into "what actually works," independent of whether it matches Jev.

## Baseline (A0) for comparison

Current mechanism, unchanged: causal decoder (Qwen3), state prefix run once, **one forward
pass per question** sharing the state's KV cache, next-token logits restricted to the
question's label tokens. N questions on one state = 1 prefix pass + N short suffix passes.
This is what every number in `docs/BENCHMARKS.md` so far measures.

## Experiment catalog

| id | hypothesis | what changes | cost to prototype | status |
|---|---|---|---|---|
| A1 | Reading 1 above: batching all of a state's questions into **one** forward pass (not N) loses nothing and is strictly what "single query" would mean on an ordinary transformer | pack all question suffixes after the shared state prefix in one sequence with per-question sentinel/readout positions; gather each question's label logits from its own position instead of running N separate suffix passes | low - reuses `MenuScorer` almost entirely, no retraining needed (readout mechanics only) | **implemented**, see below |
| A2 | Causal masking is actively hurting state comprehension: a state token can't attend to *later* state tokens, but nothing about reading a log/JSON blob is inherently left-to-right | convert to a prefix-LM attention mask - bidirectional over the state span, causal only from the question span onward - initialized from the pretrained causal weights, then SFT-adapted (the UL2/GLM recipe, not training from scratch) | medium - needs a custom attention mask through Qwen3's HF attention implementation, and a short adaptation fine-tune before it's comparable | proposed |
| A3 | A dedicated **slot-query decision head** - learned query embeddings that cross-attend to the full state+question hidden sequence, one query per pending question, answered in parallel - captures information a single last-token readout can miss and is a literal implementation of reading 3 above | new small module: K learned query vectors, one cross-attention layer over the backbone's hidden states, output projected to each question's label logits; backbone frozen or lightly fine-tuned, head trained fresh | medium-high - new trainable component, needs its own training loop variant | proposed - **this is our candidate "own novel architecture," see below** |
| A4 | Discretizing Score/Noul into softmax bins throws away the fact that they're naturally continuous/probabilistic; a parametric head (Beta for Noul, Dirichlet for Choice/Score) trained against a continuous proper scoring rule could calibrate better, especially for Score | replace the restricted-softmax readout for Score/Noul with a small MLP outputting Beta/Dirichlet concentration parameters from the last hidden state, loss = negative log-likelihood or CRPS instead of cross-entropy | low-medium - self-contained head swap, Choice can stay as-is (no natural continuous relaxation) | proposed |
| A5 | A single dense backbone spreads capacity evenly across domains; a handful of small per-domain expert adapters behind a lightweight router could specialize without the latency cost of a full MoE | LoRA-style per-domain adapters + a tiny router (predicted from the state's pooled hidden state) selecting which adapter(s) fire; base backbone shared and frozen | medium - training loop change (router + adapter switching), connects to RESEARCH.md R5 | proposed |
| A7 | The top of the stack does no decision work: option-letter logits read through the tied `norm`+`lm_head` stop changing several layers before the top, so those layers are pure latency | delete the top layers and keep the truncated stack as the model (`experimental/truncate.py`); optionally re-adapt the new final layer with a short LoRA pass | **low** - one probe pass measures it, truncation is a checkpoint edit, no retraining needed to test | **measured**, see A7 below |
| A8 | Depth is a free ensemble: every layer's distribution is already computed in one forward pass, so averaging the last K, or using their disagreement, buys calibration at zero extra compute (unlike permutation ensembling, which costs N passes) | average the last-K layers' restricted softmaxes; separately, feed "how often did the argmax flip across late layers" into the confidence model | **low** - falls out of the same probe | **measured**, mixed - see A8 below |
| A9 | v6's 24 linear-attention layers carry a fixed-size recurrent state that *is* a learned summary of the whole state document; reading the decision off that state (and reusing it across questions) is closer to a "parallel sampler" than copying a KV cache, and would restore the multi-question optimisation that this backbone currently falls back from | read/branch the Gated-DeltaNet recurrent state instead of the KV cache; fix the `batch_repeat_interleave` gap that makes `use_state_cache` fall back today | medium - needs to touch the hybrid cache internals | proposed |
| A10 | Reading logits at option *letters* is an arbitrary binding the model must learn, it breaks past 26 options (Jev supports 255), and it is the direct cause of our option-order sensitivity; scoring each option by the similarity between the decision position's hidden state and an encoding of the option's own text is permutation-invariant **by construction** | replace the letter readout with a bi-encoder score over option text, initialised from `lm_head`/embedding weights of the option tokens so it starts at the pretrained solution rather than from scratch | medium - new readout + training variant, but the init is what A3 lacked | proposed - highest-value of the untried readout changes |
| A11 | Non-autoregressive does not have to mean fixed compute: route easy states to an early exit (A7's truncation) and only hard ones to full depth, so mean latency falls without capping hard accuracy | a tiny gate on the early-exit layer's distribution decides whether to continue; the remaining layers run only when it abstains | low-medium - composes A7 with a threshold fitted on calibration | proposed |
| A12 | A single forward pass cannot do multi-step arithmetic, which is exactly where we are weakest (`temporal_numeric` 0.07 on JevBench hard); looping a block of layers k times adds serial computation *without* generating text, so it keeps the latency advantage that generating a chain of thought would destroy | re-enter a middle block of layers k times (universal-transformer / latent-recurrence style) with k fitted per difficulty; cost is k x that block only | high - training-loop change and the only idea here that changes compute shape | proposed - the one route to beating Jev on hard reasoning at one pass |
| A13 | Weight-space averaging of checkpoints trained on the same backbone ("model soup") usually improves calibration for free and costs nothing at inference, unlike output ensembling | average the merged weights (or the LoRA deltas) of two or more sibling checkpoints, e.g. v7.2 and v7.3 | **low** - no training, no inference cost | proposed |
| A6 | Menu answers within one state may be correlated (e.g. `urgency=critical` should shift `queue` probabilities) - independent per-question softmax can't express that; a joint/energy-based scoring over the *combination* of answers might calibrate better on multi-question states | score compatible answer combinations jointly (small joint energy head over pairs of question outputs) instead of treating each question as independent | high - biggest departure from the current mechanism, no cheap prototype | proposed, low priority (speculative, expensive to validate) |

## A1 in detail (implemented, not yet measured)

`open_spark_jev/experimental/parallel_readout.py`. Packs the state prefix plus **all**
questions' suffixes into a single sequence (each suffix still ends at its own non-thinking
assistant header), runs **one** forward pass, and gathers each question's label logits from
its own last-suffix-token position - instead of `model.py`'s N separate suffix passes sharing
a copied KV cache. Kept fully separate from `model.py`/`train/*.py` (new module, own class
`ParallelMenuScorer`) specifically so it doesn't touch anything the in-flight SFT/RLCD run
depends on. Compatible with the same checkpoint format, so it can be pointed at any of this
repo's trained checkpoints without retraining.

**What it tests**: does true single-forward-pass batching change the *answers* at all (it
shouldn't, if attention masking is done correctly - same information available to each
question either way) and how much latency it saves over A0's N-pass approach as the number of
questions per state grows. If answers are numerically identical (up to bf16 noise, like the
cache-vs-full-forward check already in `tests/test_model_gpu.py`) and latency drops
meaningfully at high question-counts, that's a real, low-risk serving optimization regardless
of what Jev does - and it directly operationalizes the one architectural claim TypeSafe made.

**Run** (once the GPU is free of the current SFT/RLCD run):
```bash
python -m open_spark_jev.experimental.parallel_readout --model checkpoints/sft-qwen3-1.7b \
  --data data/benchmarks/sim_test.jsonl --limit 200
```
Reports: max logit disagreement vs `eval/benchmark.py`'s A0 numbers on the same checkpoint,
and a latency comparison at 1/4/16 questions per state (mirrors `eval/latency.py`'s grid).

## A3 in detail: our proposed novel architecture

Working name: **Slot-Query Menu Head**. Where A0-A2 keep answering "which token comes next,"
A3 reframes decision-making as **set prediction**: given a state's hidden representations, a
fixed bank of learned query vectors (borrowed from DETR's object queries / Perceiver's latent
array, applied here to typed decisions instead of objects or modalities) cross-attends to
those representations once, and each query's output is projected to one question's label
logits. Concretely:

- Backbone (Qwen3, frozen or lightly LoRA-adapted) encodes the state once, causally as usual -
  no architecture change to the backbone itself, which keeps this cheap to prototype.
- A small number of **question-type query embeddings** (one per Choice/Score/Noul, or more
  finely one per domain x type if that helps) are prepended as a fixed, trainable set.
- One additional cross-attention layer: queries attend to the backbone's state hidden states
  (and, for Choice, to the tokenized option list so the head knows what it's choosing between).
- Each query's output vector is projected through a small MLP to that question's label logits.

**Why this might matter, independent of Jev**: it decouples "how many questions" from "how
many forward passes through the big backbone" more completely than A1 does - the backbone
runs once regardless of question count, and only the cheap cross-attention + MLP head scales
with question count. It is also the most literal implementation of "generates all outputs in a
single query" among A1-A3, and it is unambiguously *not* just constrained decoding on a
next-token distribution, which makes it a genuinely different mechanism to compare against A0,
not a re-skin of it.

**Cost / risk**: needs a real training loop (the head is randomly initialized, not inherited
from the LM head), so it can't be evaluated zero-shot like A1. Plan: prototype the module
architecture now (queued, not yet started - see status table), train it as a Phase-1-equivalent
run once the current SFT/RLCD/GRPO comparison finishes and the GPU is free, and score it with
the identical `eval/benchmark.py` suite for a fair comparison against A0's SFT numbers.

## Evaluation protocol

Every architecture variant is scored with the *same* metrics as the training-mechanism
comparison, so the two ledgers stay comparable: accuracy, macro-F1, ECE, Brier,
soft-Brier-vs-posterior (`eval/benchmark.py`), and the latency/throughput grid
(`eval/latency.py`) at 1/4/16 questions per state. Architecture changes are expected to mostly
trade off *latency and parameter-efficiency*, not calibration - a variant that improves
calibration is a genuine finding; a variant that only changes latency is still worth recording
here since "kernel-aware architecture decisions" is one of this project's stated research
angles (`docs/RESEARCH.md`, top of file).

Configs and code for anything beyond A0 live under `configs/experimental/` and
`open_spark_jev/experimental/` respectively, kept structurally separate from
`configs/train/` and `open_spark_jev/train/` so experimental architecture work never risks the
main SFT/RLCD/GRPO pipeline's correctness or reproducibility.

## Status ledger

| id | name | status | measured against A0 | date | notes |
|---|---|---|---|---|---|
| A0 | baseline (restricted LM-head decode) | measured | - (this is the baseline) | 2026-09-18 | see docs/BENCHMARKS.md |
| A1 | single-pass parallel multi-question readout | **measured** | equivalent answers (max \|Δp\| 0.022), ~15% faster at 1-16 questions | 2026-09-19 | confirmed; keep as a serving optimisation, BENCHMARKS.md B22 |
| A2 | prefix-LM / bidirectional-state attention | **measured** | **worse on every axis**; -7 to -40 pts on external sources, 30% slower to train | 2026-09-19 | REFUTED as a cheap drop-in (BENCHMARKS.md B22). The predicted re-adaptation cost is real: one LoRA epoch cannot undo causal pretraining. |
| A3 | slot-query menu head | **measured** | worse than A0 on every measure (60-set 0.583 vs 0.767; injection 0.397, below chance) | 2026-09-19 | REFUTED as a drop-in at one epoch (BENCHMARKS.md B26); A4-A6 paused |
| A4 | parametric Beta/Dirichlet output head | implemented, run deferred | pending | 2026-09-19 | evidential head + loss written, smoke-tested; deferred behind M17 |
| A5 | domain-routed adapter mixture | implemented, run deferred | pending | 2026-09-19 | 4 adapters + linear router written; deferred behind M17 |
| A6 | joint/energy-based multi-question scoring | implemented, run deferred | pending | 2026-09-19 | two-question correlated simulator built (0.060 oracle joint-vs-product headroom); deferred behind M17 |

| A7 | depth truncation (drop top layers) | **measured** | layer 24 of 32: own splits -0.7 pt (0.8450 vs 0.8517), JevBench hard **+1.8 pt** (0.6216 vs 0.6036) | 2026-09-26 | **8 of 32 layers (25%) are free.** Probe `runs/v7/depth_probe_v6_*.json`; truncation in `experimental/truncate.py`. Note the decision "crystallises" at layer 24, a full-attention layer: accuracy jumps 0.597 -> 0.845 there. |
| A8 | depth as a free ensemble / uncertainty signal | **measured** | own splits ECE 0.1337 -> 0.1022 (last 8 layers, accuracy unchanged); **JevBench hard: no gain** (0.3125 -> 0.3001) | 2026-09-26 | PARTIAL. The disagreement signal itself is real and sharp on both (own: 0.882 / 0.551 / 0.000 accuracy at 0, 1, >=2 late-layer argmax flips; hard: 0.635 / 0.462 / 0.000) but fires on only 15 of 111 hard items, so it cannot fix hard-tier overconfidence: 96 of 111 items have zero flips and are still 63.5% correct at 94.8% confidence. |
| A9 | linear-attention recurrent-state readout | proposed | - | 2026-09-26 | would also restore `use_state_cache` on the hybrid backbone |
| A10 | option-text bi-encoder readout | proposed | - | 2026-09-26 | permutation-invariant by construction; scales past 26 options |
| A11 | adaptive-depth cascade | proposed | - | 2026-09-26 | composes A7 with a confidence gate |
| A12 | latent recurrence (looped block) | proposed | - | 2026-09-26 | targets `temporal_numeric` 0.07 |
| A13 | weight-space model soup | proposed | - | 2026-09-26 | cheapest untried calibration idea |

Update this table, not just prose above it, whenever an experiment's status changes - it's the
part meant to be skimmable at a glance.

## A7/A8 in detail (measured 2026-09-26)

One forward pass already computes every layer's hidden state, so applying the tied output norm and
`lm_head` to each layer's last position gives a per-layer answer distribution for nothing. That single
probe (`experimental/depth_probe.py`) answers both questions at once.

**A7.** Accuracy as a function of depth is flat from layer 24 upward, on both distributions:

| | layer 23 | layer 24 | layer 28 | layer 32 (full) |
|---|---:|---:|---:|---:|
| own splits, 600 rows | 0.5967 | 0.8450 | 0.8400 | 0.8517 |
| JevBench hard, 111 items | 0.4955 | **0.6216** | 0.5856 | 0.6036 |

The jump at layer 24 is large and it is a `full_attention` layer, which is suggestive: the hybrid
backbone appears to settle the decision at a full-attention layer and then spend eight more layers
not changing it. Truncating there is a 25% depth cut that *helps* hard-tier accuracy, and because it
removes layers rather than precision it should compound with NVFP4 instead of competing with it.

**A8.** Averaging the last K layers helps calibration on our own splits (ECE 0.1337 -> 0.1022 at K=8,
accuracy unchanged) and does essentially nothing on JevBench hard. The *disagreement* signal is much
more interesting than the ensemble: bucketing items by how many times the argmax flips across the last
eight layers separates accuracy cleanly (own splits 0.882 / 0.551 / 0.000; hard 0.635 / 0.462 / 0.000).
It is a genuine, free difficulty signal -- and it is not enough, because it only fires on 15 of 111 hard
items. The remaining 96 are confidently, stably wrong. **That is the finding that matters most for v7:
hard-tier overconfidence has a low post-hoc ceiling, so the lever is training data, not a better
temperature.** `scripts/analysis/calib_v7_study.py` says the same thing from the other side: v6's
shipped per-type temperature is already near-optimal on our own splits (ECE 0.0073 on test_locked,
against 0.0403 raw), and no reweighting of those fits transfers to hard items.
