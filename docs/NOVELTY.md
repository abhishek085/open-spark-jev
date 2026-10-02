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
| A4 | Discretizing Score/Noul into softmax bins throws away the fact that they're naturally continuous/probabilistic; a parametric head (Beta for Noul, Dirichlet for Choice/Score) trained against a continuous proper scoring rule could calibrate better, especially for Score | replace the restricted-softmax readout for Score/Noul with a small MLP outputting Beta/Dirichlet concentration parameters from the last hidden state, loss = negative log-likelihood or CRPS instead of cross-entropy | low-medium - self-contained head swap, Choice can stay as-is (no natural continuous relaxation) | **measured, no improvement** - ported to v7.x/4B pipeline as v7.7-4b (`train/sft_a4.py`, `experimental/eval_a4.py`); trails v7.6 on test_locked (0.807 vs 0.825) and challenge (0.838 vs 0.847), worse calibration (test_locked choice ECE 0.084 vs v7.6's 0.044). Not refuted as badly as A3/A19 (LoRA still adapts the backbone here), but no reason to ship it. |
| A5 | A single dense backbone spreads capacity evenly across domains; a handful of small per-domain expert adapters behind a lightweight router could specialize without the latency cost of a full MoE | LoRA-style per-domain adapters + a tiny router (predicted from the state's pooled hidden state) selecting which adapter(s) fire; base backbone shared and frozen | medium - training loop change (router + adapter switching), connects to RESEARCH.md R5 | proposed |
| A7 | The top of the stack does no decision work: option-letter logits read through the tied `norm`+`lm_head` stop changing several layers before the top, so those layers are pure latency | delete the top layers and keep the truncated stack as the model (`experimental/truncate.py`); optionally re-adapt the new final layer with a short LoRA pass | **low** - one probe pass measures it, truncation is a checkpoint edit, no retraining needed to test | **measured**, see A7 below |
| A8 | Depth is a free ensemble: every layer's distribution is already computed in one forward pass, so averaging the last K, or using their disagreement, buys calibration at zero extra compute (unlike permutation ensembling, which costs N passes) | average the last-K layers' restricted softmaxes; separately, feed "how often did the argmax flip across late layers" into the confidence model | **low** - falls out of the same probe | **measured**, mixed - see A8 below |
| A9 | v6's 24 linear-attention layers carry a fixed-size recurrent state that *is* a learned summary of the whole state document; reading the decision off that state (and reusing it across questions) is closer to a "parallel sampler" than copying a KV cache, and would restore the multi-question optimisation that this backbone currently falls back from | read/branch the Gated-DeltaNet recurrent state instead of the KV cache; fix the `batch_repeat_interleave` gap that makes `use_state_cache` fall back today | medium - needs to touch the hybrid cache internals | proposed |
| A10 | Reading logits at option *letters* is an arbitrary binding the model must learn, it breaks past 26 options (Jev supports 255), and it is the direct cause of our option-order sensitivity; scoring each option by the similarity between the decision position's hidden state and an encoding of the option's own text is permutation-invariant **by construction** | replace the letter readout with a bi-encoder score over option text, initialised from `lm_head`/embedding weights of the option tokens so it starts at the pretrained solution rather than from scratch | medium - new readout + training variant, but the init is what A3 lacked | proposed - highest-value of the untried readout changes |
| A11 | Non-autoregressive does not have to mean fixed compute: route easy states to an early exit (A7's truncation) and only hard ones to full depth, so mean latency falls without capping hard accuracy | a tiny gate on the early-exit layer's distribution decides whether to continue; the remaining layers run only when it abstains | low-medium - composes A7 with a threshold fitted on calibration | **measured, clean win (corrected 2026-09-29)** - see below: zero accuracy loss at 17-18% compute reduction on own splits (full 1500-row splits, not a biased sample) and JevBench hard |
| A12 | A single forward pass cannot do multi-step arithmetic, which is exactly where we are weakest (`temporal_numeric` 0.07 on JevBench hard); looping a block of layers k times adds serial computation *without* generating text, so it keeps the latency advantage that generating a chain of thought would destroy | re-enter a middle block of layers k times (universal-transformer / latent-recurrence style) with k fitted per difficulty; cost is k x that block only | high - training-loop change and the only idea here that changes compute shape | proposed - the one route to beating Jev on hard reasoning at one pass |
| A13 | Weight-space averaging of checkpoints trained on the same backbone ("model soup") usually improves calibration for free and costs nothing at inference, unlike output ensembling | average the merged weights (or the LoRA deltas) of two or more sibling checkpoints, e.g. v7.2 and v7.3 | **low** - no training, no inference cost | proposed |
| A6 | Menu answers within one state may be correlated (e.g. `urgency=critical` should shift `queue` probabilities) - independent per-question softmax can't express that; a joint/energy-based scoring over the *combination* of answers might calibrate better on multi-question states | score compatible answer combinations jointly (small joint energy head over pairs of question outputs) instead of treating each question as independent | high - biggest departure from the current mechanism, no cheap prototype | proposed, low priority (speculative, expensive to validate) |
| A16 | A3's slot-query head was refuted as a drop-in, but never isolated *why* - a controlled 3-way ablation (pooled state -> MLP; pooled state + option text -> shared MLP; option queries cross-attending token states -> shared MLP) on a frozen backbone can tell whether any gain from attention is candidate-conditioning (already in `option_readout.py`) or genuine token-level retrieval | freeze v6, cache dev-set token states once, train three similarly-sized heads on the same frozen features and same examples | low-medium - frozen-backbone linear probes, no backbone retraining, reuses `option_readout.py`'s candidate-conditioned head as arm #2 | proposed - external suggestion, 2026-09-28 |
| A17 | A Perceiver-style memory bottleneck (K learned queries cross-attend once over context states; options then attend only to those K memory vectors) could let the model extract reusable "decision-relevant facts" once and score many candidate options cheaply, instead of re-attending across the full state per option | new learned memory-query module (K in {8, 16, ...}, sweep rather than assume bigger is better) between the frozen backbone and the option scorer | medium-high - new trainable component with its own capacity knob to sweep; heavier than A16 for an unconfirmed gain | proposed - external suggestion, 2026-09-28; deprioritized behind A16/A18/A19 until they show attention beats plain candidate-conditioning |
| A18 | A set-self-attention block over contextualized option vectors could model competition between options (this one is better *because* that one is worse), which independent per-option scoring can't express - distinct from A6's cross-question correlation idea, this is *within-question, cross-option* | one set-self-attention layer over option vectors, deliberately with **no** option-position embeddings, so semantic decisions survive option shuffling | low-medium - new module, but scores directly against existing permutation-invariance eval (`eval/osdg.py`'s per-permutation accuracy and answer-flip rate) with no new eval infra needed | proposed - external suggestion, 2026-09-28 |
| A19 | Extends A7/A8's logit-lens finding (decision crystallises at layer 24/32) into an architecture choice: does cross-attention read intermediate layers (more lexical evidence) or final layers (more decided) better, and does a small learned mixture of two depths beat either alone | compare cross-attention over an intermediate vs. final backbone layer's states, plus a learned two-depth mixture, reusing `depth_probe.py`'s per-layer logit-lens infrastructure | **low** - direct extension of already-built A7/A8 probe code, no new infra | **measured, REFUTED** - see below, both depths and the mixture lose badly to the existing zero-shot readout |

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
| A4 | parametric Beta/Dirichlet output head (v7.7-4b) | **measured, no improvement** | v7.7-4b vs v7.6-4b progsplits: test_locked acc 0.807 vs 0.825, challenge 0.838 vs 0.847, worse ECE throughout | 2026-09-29 | ported from 1.7B/old-data harness to current 4B pipeline; MenuScorer.decide() doesn't read aux_head so needed a dedicated eval script (`experimental/eval_a4.py`) |
| A5 | domain-routed adapter mixture | implemented, run deferred | pending | 2026-09-19 | 4 adapters + linear router written; deferred behind M17 |
| A6 | joint/energy-based multi-question scoring | implemented, run deferred | pending | 2026-09-19 | two-question correlated simulator built (0.060 oracle joint-vs-product headroom); deferred behind M17 |

| A7 | depth truncation (drop top layers) | **measured** | layer 24 of 32: own splits -0.7 pt (0.8450 vs 0.8517), JevBench hard **+1.8 pt** (0.6216 vs 0.6036) | 2026-09-26 | **8 of 32 layers (25%) are free.** Probe `runs/v7/depth_probe_v6_*.json`; truncation in `experimental/truncate.py`. Note the decision "crystallises" at layer 24, a full-attention layer: accuracy jumps 0.597 -> 0.845 there. |
| A8 | depth as a free ensemble / uncertainty signal | **measured** | own splits ECE 0.1337 -> 0.1022 (last 8 layers, accuracy unchanged); **JevBench hard: no gain** (0.3125 -> 0.3001) | 2026-09-26 | PARTIAL. The disagreement signal itself is real and sharp on both (own: 0.882 / 0.551 / 0.000 accuracy at 0, 1, >=2 late-layer argmax flips; hard: 0.635 / 0.462 / 0.000) but fires on only 15 of 111 hard items, so it cannot fix hard-tier overconfidence: 96 of 111 items have zero flips and are still 63.5% correct at 94.8% confidence. |
| A9 | linear-attention recurrent-state readout | proposed | - | 2026-09-26 | would also restore `use_state_cache` on the hybrid backbone |
| A10 | option-text bi-encoder readout | proposed | - | 2026-09-26 | permutation-invariant by construction; scales past 26 options |
| A11 | adaptive-depth cascade | **measured, clean win (corrected)** | v7.6-4b, threshold fit on the FULL 1500-row calibration split (smallest threshold where blended cascade accuracy >= full-depth accuracy): threshold 0.95. **test_locked** (full 1500 rows) exit_frac 0.681 / relative_compute 0.830 / cascade_acc 0.8020 vs full_depth_acc 0.8013 (tied, +0.07pt noise). **challenge** (full 1500 rows) exit_frac 0.711 / rel_compute 0.822 / cascade_acc 0.8080 vs full_depth_acc 0.8067 (tied, +0.13pt noise). **JevBench hard** (full 111 items) exit_frac 0.721 / rel_compute 0.820 / cascade_acc 0.5405 == full_depth_acc 0.5405 exactly. Net: ~17-18% compute reduction, zero accuracy cost, on 3 independent held-out sets. | 2026-09-29 | `experimental/depth_cascade.py`; no new training, pure post-hoc analysis of `depth_probe.py`'s already-free per-layer logit lens. **Correction, same day**: an earlier version of this result used `depth_probe.py --limit 150/200` on `data/synthetic/prog_v1/*.jsonl`, whose rows are grouped by `scenario_family` -- the sample was entirely `temporal_numeric` (a family v7.6 is now strong on), which inflated accuracy (0.95/0.97) and the threshold-fit target. Refit and reran on the full 1500-row splits; JevBench hard and the 5 external sources were already run on full files (no `--limit`) so those numbers were never affected. Candidate to actually wire into serving (unlike A7's static truncation, this costs nothing in accuracy). Shipped as **v7.6.1-4b** (`checkpoints/v7.6.1-4b/cascade.json`; weights are v7.6-4b unchanged, symlinked, not copied). |
| A12 | latent recurrence (looped block) | proposed | - | 2026-09-26 | targets `temporal_numeric` 0.07 |
| A13 | weight-space model soup | **measured - new best, v7.6-4b** | JevBench hard: accuracy identical (0.5586, same argmax on all 111 items), ECE 0.333->0.309, Brier 0.704->0.682. External eval (5 sources): beats or ties v7.5 on 9/10 accuracy+ECE cells. Own splits (v5splits/progsplits): flat to mixed, no regression that isn't noise-level. | 2026-09-28 | `experimental/model_soup.py` averaged v7.2-4b + v7.3-4b (true siblings: identical data, differ only in lambda_brier); calibration refit with `experimental/fit_calibration.py` (averaging invalidates source temperatures). **Promoted to v7.6-4b, now the shipped-best checkpoint** - use this as the base for the next A-series experiment instead of v7.5. |
| A14 | v7.4: diversify prog_v1's fixed boilerplate (paraphrase pools) | **measured, regression not fixed** | JevBench hard 0.559 (v7.2) -> see v7.4 log; own-splits/prog TVD tbd | 2026-09-26 | fixes the template-fingerprint mechanism but v7.3 (different lambda_brier, same templated data) regressed too, so template-fingerprinting is A cause, not necessarily the only one |
| A15 | v7.5: KL-to-frozen-v6 replay rows, borrowed from decider-4b's v2->v2.1 fix | queued behind A14 | - | 2026-09-26 | `experimental/replay_kl.py`; trains a sample of v6's own training rows toward v6's own T=1 distribution instead of hard labels, so new data can't sharpen the model away from behaviour that was already well-calibrated |
| A13 | weight-space model soup (v7.2+v7.3 -> v7.6-4b) | **measured** | JevBench hard ECE 0.333->0.309, Brier 0.704->0.682, accuracy unchanged; external eval better/tied on 9/10 cells | 2026-09-28 | promoted to v7.6-4b, new best; see catalog entry above for full numbers |
| A16 | pooling vs. candidate-conditioned MLP vs. cross-attention readout ablation | proposed | - | 2026-09-28 | external suggestion; isolates whether A3-style attention beats `option_readout.py`'s candidate-conditioning, or just matches it |
| A17 | learned memory-bottleneck (Perceiver-style) between backbone and option scorer | proposed | - | 2026-09-28 | external suggestion; deprioritized behind A16/A18/A19 - highest cost, unconfirmed gain |
| A18 | set-self-attention over option vectors (within-question competition, permutation-invariant) | proposed | - | 2026-09-28 | external suggestion; scores directly against existing `eval/osdg.py` permutation-invariance eval |
| A19 | cross-attention read depth: intermediate vs. final layer vs. learned two-depth mixture | **measured, REFUTED** | v7.6-4b, test_locked n=150: layer 16 acc 0.307 (baseline zero-shot logit-lens 0.340), layer 32 acc 0.320 (baseline **0.953**); mixture collapsed to alpha=0 | 2026-09-28 | `experimental/depth_readout.py`; same failure mode as A3 - freshly-initialised head can't compete with the pretrained readout at ~300 training examples |


| A21 | soup(v6, v7.6) -- recover broad JevBench-hard families lost to targeted prog_v1 training | **measured, tradeoff, not a clean win** | v7.8-4b: JevBench hard beats both parents (acc 0.604 vs v6 0.595/v7.6 0.559, ECE 0.267 vs v6 0.265/v7.6 0.309, Brier 0.636 -- worse than v6's 0.606, better than v7.6's 0.682). progsplits (own domain) REGRESSES vs v7.6: test_locked 0.752 vs 0.825 (-7.3pt), challenge 0.754 vs 0.847 (-9.3pt) -- gives back about a third of the entire v7.x campaign's own-domain gain over v6. | 2026-09-29 | `experimental/model_soup.py checkpoints/v6-4b checkpoints/v7.6-4b`; JevBench hard scored via the official `scripts/run_jevbench.sh` harness, not `depth_probe.py` (the two disagree by a couple points per-checkpoint -- see methodology note below). Checkpoint kept for reference; NOT promoted over v7.6/v7.6.1 as shipped-best, since the own-domain regression outweighs the hard-set gain. |

| A22 | RLDM-9: offline advantage-weighted regression on already-logged eval predictions (v7.9-4b) | **measured, mixed - NOT a clean win, and the mechanism is unproven (uncontrolled)** | vs v7.6, clean held-out splits: progsplits test_locked 0.844 vs 0.825 (+1.9pt), challenge 0.873 vs 0.847 (+2.6pt), calibrated ECE better (0.031 vs 0.044; 0.018 vs 0.040). **JevBench hard (official harness)**: acc 0.559 = 0.559 (flat), Brier 0.719 vs 0.682 and ECE 0.325 vs 0.309 (both WORSE). External (5 sources): mixed/slightly worse on 3 (injection-ctx, injection-noctx, jev-directory), ~tied on kev-decision, better ECE on toolcall-risk. | 2026-09-29 | `experimental/awr_data.py` + `train/awr.py` + `configs/train/awr_v1.yaml`; 7,500 rows = **only 1,500 unique calibration-split items x 5 checkpoints' logged predictions**, weights mild (mean 1.12, std 0.33, 71% of rows in [0.5,1.5]), 2 epochs (~10 passes over each item). **Confounds**: (1) no uniform-weight control was run, so the held-out own-split gain may be just "1,500 extra in-distribution prog_v1 items", not advantage weighting; (2) the calibration split is now training data, so osdg.py's temperature-fit-on-calibration is fit on seen rows (test_locked/challenge remain clean). Checkpoint kept; not promoted over v7.6. A control (same rows, weight=1.0) would be needed to credit or discredit AWR itself. |

| A23 | RLDM-3: learn the A11 cascade gate with an exact expected-reward update (2 actions, reward = correctness - lambda*compute) instead of a hand-fit threshold; state adds A8 layer-disagreement features | **measured, null result** | Operating point fixed on calibration only (RL lambda=0.2 vs threshold 0.94/0.96), applied unchanged to every held-out set. test_locked exit 0.723 vs 0.703 (acc 0.8020 vs 0.8013); challenge 0.750 vs 0.731 (0.8073 vs 0.8087); JevBench hard 0.748 vs 0.721 (0.5495 vs 0.5405, = 1 item of 111); 5 external sources within +-0.02 exit fraction, identical accuracy. Net: ~+1-2.5pt exit fraction = ~0.5pt relative compute (0.819 vs 0.824). | 2026-09-29 | `experimental/gate_rl.py`, CPU-only on cached logit-lens outputs, ~100-param policy. With two actions and full-information rewards this is cost-sensitive classification; the extra A8 features (flips, tvd) added almost nothing over layer-24 confidence. **Do not cite the script's own "zero_loss_summary"** -- it picks the best point per eval set with hindsight and is not deployable; the fixed-operating-point table above is the fair comparison. Not worth wiring in over the A11 threshold. |

| A24 | RLDM-6: evolution-strategy search over soup weights across v6/v7.2/v7.3/v7.4/v7.5 (v7.10-4b); fitness = calibrated Brier on prog_v1 + v5 CALIBRATION splits only | **measured, null result** | Search converged to a simplex vertex: champion = 96.6% v7.5 + 2.4% v7.3 + <1% others (fitness Brier 0.1605 vs v7.5 alone 0.1622, v7.4 0.1615; margin ~0.001-0.002, in-sample on the same 400 items every eval, i.e. within noise). Best-of-gen plateaued at generation 1 (0.1608) and never moved (0.1605-0.1647 over 10 generations). Held-out, official harness, v7.10 vs v7.5 vs v7.6: progsplits test_locked 0.815 / 0.817 / 0.825, challenge 0.837 / 0.833 / 0.847; JevBench hard acc 0.5586 / 0.5586 / 0.5586 (identical), Brier 0.770 / 0.704 / 0.682, ECE 0.380 / 0.333 / 0.309; external accuracy within +-0.002 of v7.5 on all 5 sources (ECE 0.001-0.007 worse). | 2026-09-30 | `experimental/es_soup.py` (88 evals, ~3 h). Nothing beats v7.6. Read: fitness landscape between v7.3/v7.4/v7.5 is flat; the earlier soup gains (A13, A21) came from mixing *dissimilar* checkpoints, not from tuning weights among siblings. **Caveat on the JevBench-hard calibration gap vs v7.5**: my `fit_calibration.py` produced choice T=0.479 for v7.10 (v7.5/v7.6 ship 0.987) because it is fit on an in-distribution validation split the model has memorised; sharper confidences inflate Brier/ECE on out-of-distribution items, so part or all of that gap is the temperature file, not the weights (accuracy is identical). **Fixed-temperature re-score (measured)**: with v7.5's temperature (0.987) v7.10's JevBench hard is acc 0.5586 / Brier 0.706 / ECE 0.332, i.e. identical to v7.5 (0.5586 / 0.704 / 0.333); the calibration gap was purely my temperature file, not the weights. **Operational note**: the first two full runs were OOM-killed (machine hard-reset 17:12 and 20:57 on 2026-09-29; not thermal, temps 57-73C) by holding 5 checkpoints resident plus 16x2048-token full-vocab logits on the shared 121 GB memory pool; rewritten with lazy per-blend checkpoint reads, a token-budget batcher, a MemAvailable preflight and oom_score_adj=1000. |

| A25 | prog_v2: redesign of all five programmatic families for real diversity (data-first direction, 2026-09-30) | **data built + verified + baselined; training effect NOT yet measured** | `os_datagen/programmatic/v2*.py` (effective_dating, policy_precedence, sampling_probability, ambiguous_routing, weighted_tradeoff), data `data/synthetic/prog_v2_full/` (20,000 train + 3 x 1,500 eval, shuffled, 35% soft-target rows), generator `python -m os_datagen.programmatic.v2_all`. All 24,500 rows verified by independent re-derivation from a stored structured world + render-fidelity checks; **0 of 24,500 flagged by the JevBench overlap guard**. Recurring-5-gram share v1 -> v2: effective_dating 0.90 -> 0.33, ambiguous_routing 1.00 -> 0.30, weighted_tradeoff 0.95 -> 0.53, policy_precedence 0.90 -> 0.64, sampling_probability 1.00 -> 0.74 (the last two remain the most repetitive); train->eval phrase overlap v1 0.88-0.99 -> v2 0.08-0.74. Eval splits hold out rule phrasing, layout, precedence statement and (test_locked, challenge) whole domain skins. Pilot baselines (effective_dating only, zero-shot, chance ~0.30): v6 0.653, v7.4 0.718, v7.5 0.724, v7.6 0.732; weakest rule is period-start (0.54-0.62), and wrong answers coincide with the event-date value only ~40-55% of the time (confounded), so the 'always uses the event date' shortcut is only partly supported. | 2026-09-30 | Motivation (`scripts/analysis/data_audit.py`, `runs/v7/data_audit.json`): prog_v1/v5_train hold ~17.8k unique decisions, 95% / 83% boilerplate 5-gram share; prog_v1 eval splits share 81-92% of their phrasing with train (v5 splits are mostly held-out families, 67-69%); prog_v1 also hid fixed constants (long_policy: vacancy limit always 60 days, seepage always 2 weeks; tradeoff: weights always 5/3/2). Known gaps: policy docs median ~1,050 tokens vs JevBench long_policy >2,000; sampling/policy filler still repetitive; all skins are synthetic vocabularies. prog_v1's `_VARIANT_FOR_SPLIT` already held out phrasing variants; v2 also holds out skins/layouts. Next: matched SFT A/B (user chose 'build more families first' before any training). |

**Methodology note (2026-09-29)**: `depth_probe.py`'s zero-shot scoring and the official `scripts/run_jevbench.sh` harness give different JevBench-hard numbers for the *same* checkpoint (v7.6: 0.541 via depth_probe vs 0.559 via the harness) -- likely differences in prompt rendering/serving path. `depth_probe.py` is fine for same-checkpoint layer-vs-layer analysis (A7/A8/A11), where it's internally self-consistent, but do not use it to compare accuracy *across* different checkpoints/lineages -- use the official harness for that, as this entry (A21) does.

Update this table, not just prose above it, whenever an experiment's status changes - it's the
part meant to be skimmable at a glance.

## What decider-4b (Mapika/decider-4b) actually does, and what we borrowed

Read the full model card (`huggingface.co/Mapika/decider-4b`, 2026-09-26) after v7.2 regressed on
JevBench hard. Same base (Qwen3.5-4B-Base), same single-pass letter-logit readout, no RL stage -- so
whatever it's doing better than us is data and training-recipe, not architecture or RLHF/RLCD:

* **Scale is the real gap.** Stage 1: 1,892,408 rows / 742M tokens / 26,729 steps, training
  cross-entropy floors at **0.35**, not near-zero -- at that scale the model cannot memorize, so it is
  forced to generalize. Our v6/v7.2 floor at ~1e-4 on 20-25k rows. Ten programmatic families (we have
  five), each with **a held-out template variant kept out of training** for model selection.
* **They hit our exact bug and fixed it differently than we are.** v2's stage-2 replay rows (6,676
  rows resampled from stage-1 data) were trained on hard labels, sharpened logits everywhere, and blew
  the fitted temperature out to 1.935 (flattening every served answer) -- a calibration regression from
  adding data, same shape as our v7.2 story. Their fix, v2.1: train those replay rows toward the
  **frozen v1 checkpoint's own distribution** via KL(p_v1 || p_model) instead of hard labels. Mean KL
  to v1 dropped 0.112 -> 0.021 nats, temperature came back to 1.099. This is A15 above.
* **Even they don't solve hard-tier calibration.** JevBench public hard: decider-4b v2.1 accuracy
  0.649, ECE 0.184 (per-type map) / 0.210 (global T). Beats our v6 (0.595/0.265) on accuracy, but the
  ECE gap is real and not dramatic -- JevBench's composite "Calibration" axis score exaggerates it.
  Their own card: "On those items, do not read a confidence of 0.8 as an 80% chance of being right."
* Self-consistency-filtered teacher data (write once, independently re-answer twice with shuffled
  options in a fresh context, keep only if all three agree -- 89-91% kept) is their non-programmatic
  diversity source, complementary to code-verified families. Not yet built here; candidate for a v7.6.
* AdamW directly on bf16 params (no FP32 master copy) beat FP32-master AdamW by 3.3 points / 0.072
  nats in their own controlled ablation -- but that was full-parameter tuning; we train LoRA adapters,
  which are already small and close to bf16 resolution, so this is lower-priority for us.

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

## A26 - Hermes-agent data track + v7.11 (in progress, 2026-10-01)
Goal: beat v6 on accuracy AND calibration (confident-wrong rate, ECE, Brier), and improve Hermes-agent decisions (tool choice, `finish`, command risk, tool-result injection). Data: public agent slice (When2Call CC-BY-4.0, xlam-irrelevance CC-BY-4.0, ToolACE Apache-2.0, Gandalf MIT; 0/42,937 flagged by JevBench or eval-source guards) + synthetic `prog_hermes_v1` (7,800 rows, code-oracle labels, held-out phrasing/injection templates, 0 flagged). Baseline on prog_v2_full (test+challenge, calibrated): v6 acc 0.747 / confident-wrong(>=0.8) 2.1%; v7.6 acc 0.799 / 2.9% (policy_precedence 6.6%). Training effect of v7.11 NOT yet measured. Gate: no dataset may regress vs v6 on accuracy; confident-wrong, ECE, Brier must not be worse than v6.

### A26 result - v7.11-4b (2026-10-01): large in-domain gains, NOT a clean win vs v6
Trained from base on 32.5k rows (v5 8k + prog_v2 8k + prog_hermes_v1 6k + public agent 8k + v6 KL replay 2.5k), lambda_brier 1.0, ~5 h. Calibrated, test+challenge, one permutation (scripts/analysis/calib_report.py):
- prog_v2_full acc v6 .745 / v7.6 .798 / v7.11 .894 (in-distribution generator, held-out variants); confident-wrong@.9 .001 / .005 / .014 (worse); policy_precedence cw@.9 .000 / .012 / .035.
- prog_hermes_v1 acc .760 / .760 / .994 (in-distribution; saturated, do not over-read); cw@.8 .081 / .077 / .004.
- public agent evals (When2Call test, ToolACE-heldout tools, Gandalf) acc .764 / .756 / .808; ECE equal; cw@.8 .056 / .048 / .035.
- ext: injection-ctx .924/.924/.938, injection-noctx .867/.854/.908, toolcall-risk .900/.917/.933, kev .800/.799/.799 (Brier .344/.341/.326), jev-directory .871/.886/.857.
- v5 own-domain: acc .929 / .938 / .927, ECE .012/.018/.006, Brier .108/.107/.115.
- JevBench (official harness): original 1.0/.986/1.0, easy 1.0 x3, HARD acc .595 / .559 / .532, Brier .606/.682/.730, ECE .265/.309/.297, conf>=.9 wrong 18/20/25 of ~69. n=111 so +-4.7pt s.e., but direction is worse on every hard metric.
Gate verdict: fails (JevBench-hard accuracy+calibration, prog_v2 confident-wrong@.9, jev-directory -1 item). Checkpoint kept as v7.11-4b.

### A26 follow-ups (2026-10-02): soup v7.12 + calibration variant v7.12.1; v7.13 long-data retrain = no transfer
- v7.12-4b = uniform soup of v6-4b + v7.11-4b. JevBench: original 1.0, easy 1.0, hard .595 (= v6). Accuracy >= v6 on every dataset tried: hermes .953 vs .760, public agent .797 vs .764, prog_v2 .851 vs .745, v5 .932 vs .929, inj-ctx .937 vs .924, inj-noctx .890 vs .867, jev-dir .914 vs .871, toolcall .900 = .900, kev .802 vs .800. v7.12b (v6+v7.6+v7.11) worse (hard .559) - dropped.
- v7.13-4b (v7.11 mix + prog_long_v1 4.5k rows, max_len 4096): learns its own long data (prog_long acc .801 vs v6 .490; temporal_calendar .641 vs .205; multi_hop .730 vs .425) but JevBench hard .523 (long_policy 6/19, multi_hop 13/18, temporal_numeric 0/15): generator-specific learning, no transfer. Note: sft.py silently DROPS rows longer than max_len, so every run before v7.13 excluded long rows.
- Calibration: v6's confident-wrong is mostly a temperature issue. JevBench hard wrong@.9: v6 shipped T=1.48 -> 18, refit T~3 -> 0-4. Shipped-T comparisons are confounded by calibration.json (v7.11 shipped T=1.12). A length-dependent T is not supported by the synthetic data (b~0); a scalar T is used.
- v7.12.1-4b = v7.12 weights + T(choice)=2.5 (noul 1.495, score 1.192). Averages over 5 non-JevBench sets vs v6@1.48: acc .822 vs .737, Brier .238 vs .398, ECE .071 vs .134, wrong@.9 .6% vs 9.1%; JevBench hard wrong@.9 4 vs 18, Brier .551 vs .600; easy/original unchanged.
