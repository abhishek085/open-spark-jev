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
| A1 | single-pass parallel multi-question readout | implemented | pending | 2026-09-18 | prototype written, run queued behind the in-flight SFT/RLCD/GRPO pipeline |
| A2 | prefix-LM / bidirectional-state attention | proposed | pending | 2026-09-18 | needs a short adaptation fine-tune before comparable |
| A3 | slot-query menu head (our candidate novel architecture) | proposed | pending | 2026-09-18 | module design specified above; needs a real training run once GPU frees up |
| A4 | parametric Beta/Dirichlet output head | proposed | pending | 2026-09-18 | |
| A5 | domain-routed adapter mixture | proposed | pending | 2026-09-18 | connects to RESEARCH.md R5 |
| A6 | joint/energy-based multi-question scoring | proposed (low priority) | pending | 2026-09-18 | speculative, no cheap prototype identified yet |

Update this table, not just prose above it, whenever an experiment's status changes - it's the
part meant to be skimmable at a glance.
