# Research agenda

What in open-spark-Jev is a contribution rather than an integration, and the experiments that
would establish it. Each experiment is runnable from this repo.

## R1. What is "RLCD," when the only public definition is a goal?

TypeSafe names their training method "RLCD" - "Reinforcement Learning for Calibrated
Decisions" - and defines it only as an objective: reward "a confidence score that actually
matches how often it's right," contrasted with RLHF (human preference) and RLVR (verifiable
task reward). No reward formulation, loss, or supervised-stage details are published. The
name collides with a *different*, fully published technique: RLCD, Yang et al. 2023,
"Reinforcement Learning from Contrastive Distillation" - generate preference pairs by
prompting one model with contrasting principles, train a reward model on the pairs, optimize
against it. There is no way to know from public information whether TypeSafe's method is that
technique, a relative of it, or something unrelated that happens to share an acronym.

Rather than guess at one mechanism and call it done, this repo runs **four**, all targeting
TypeSafe's stated goal, and scores all four the same way:

| # | mechanism | config | needs a teacher? |
|---|---|---|---|
| 1 | academic RLCD (Yang et al.): contrastive pairs → Bradley-Terry RM → exact policy gradient with the RM's per-action score included | `configs/train/rlcd_contrastive.yaml` | yes |
| 2 | direct calibration objective: same exact policy gradient, RM term off, reward is Brier against ground-truth/teacher soft labels only | `configs/train/rlcd_direct.yaml` | no (simulator posteriors suffice) |
| 3 | sampled-token GRPO (TRL): standard RL baseline, reward is per-sample 0/1 correctness + a conservative-action penalty, policy distribution never directly observed | `configs/train/rlcd_grpo.yaml` | no |
| 4 | no-RL control: post-hoc temperature scaling on the SFT checkpoint only | `eval/calibration_baseline.py` | no |

**Claim A** (exact vs. sampled, independent of the RLCD-naming question). For single-step
decisions over a small menu, the policy distribution is available in closed form, so
`E_a~p[R(a)] + R_dist(p) - β KL(p‖p_ref)` can be differentiated exactly - no PPO/GRPO
sampling variance, and critically the reward can include **proper scoring rules evaluated on
the distribution itself** (Brier vs. posterior), which mechanism 3 cannot express because it
only ever sees one sampled token per rollout. **Prediction**: mechanism 3's 0/1-correctness
reward improves accuracy but *worsens* ECE (argmax collapse); mechanisms 1 and 2 improve
both.

**Claim B** (does the contrastive/RM detour help, or is direct calibration reward enough).
Mechanism 1 spends a full extra training stage (pairs + RM) to get a learned, per-action
reward signal; mechanism 2 skips straight to the proper scoring rule. If 2 matches or beats 1
on `soft_brier_vs_posterior` and `ece`, the contrastive-pairs machinery is not pulling its
weight for *this* objective, which would itself be a useful negative result about what "RLCD"
plausibly needs to contain.

**Claim C** (does RL earn its cost at all). Mechanism 4 costs nothing beyond the SFT run
already required for 1-3's initialization. If it closes most of the SFT-to-RL gap on its own,
that bounds how much credit any RL mechanism, ours or TypeSafe's undisclosed one, can claim
for calibration specifically (as opposed to other things RL might be doing, like shifting
accuracy or abstention behavior).

**Run.** `scripts/train_rlcd_direct.sh`, `scripts/train_rlcd_contrastive.sh`,
`python -m open_spark_jev.train.rlcd_grpo --config configs/train/rlcd_grpo.yaml`,
`python -m open_spark_jev.eval.calibration_baseline`, then `scripts/eval.sh` on each
checkpoint; compare `accuracy`, `ece`, `soft_brier_vs_posterior`, `injection_flip_rate`
across all four in `docs/BENCHMARKS.md`.

**Result (2026-09-19, docs/BENCHMARKS.md B8a-M8d).** Claim A confirmed cleanly: GRPO scored
worse than all three exact-gradient mechanisms on every axis (accuracy -6pts, ECE ~8x worse,
injection robustness 4-10x worse), consistent with argmax collapse from a reward that never
observes the full action distribution. Claim B: mechanism 1's contrastive/RM machinery did
*not* distinguish itself on accuracy or ECE (all three exact mechanisms landed within noise
of each other, ~0.81/~0.02) but won clearly on injection robustness (0.016 vs 0.027
direct-only vs 0.038 plain SFT) - the RM's trained claim about resisting in-state
instructions is doing real work there specifically. Claim C: the no-RL temperature-only
control matched the SFT checkpoint's own numbers almost exactly, which mostly reflects that
SFT already fits its own temperature internally, not a fair test of "does RL help" on its own
- a cleaner Claim-C test (temperature scaling applied to the *base* untrained model,
skipping SFT+RL entirely) is a good follow-up. Known caveat: the GRPO run used a smaller
training budget than the other three for practicality (docs/BENCHMARKS.md B8d); a
matched-budget rerun would isolate the sampled-vs-exact effect from the training-scale effect
more cleanly.

## R2. Known-posterior decision benchmark
**Claim.** Calibration should be measured against the true posterior, not one realised label.
`data/simulators.py` generates states from explicit naive-Bayes models across six domains
with the exact posterior attached, plus abstention targets from posterior entropy and
prompt-injected twins with unchanged targets. This is a reusable benchmark for *any* decision
model (hosted Jev included, via the Jev-format adapter in `serve/gateway.py`).
**Metrics.** soft Brier vs posterior, ECE, abstain precision/recall, injection flip rate.

## R3. A unified RLCD reward for accuracy + calibration + abstention + injection resistance
`train/rewards.py` composes: Brier (proper), abstain (entropy-gated), conservative
(asymmetric around the domain's safe action), injection consistency (TV distance between
injected and clean twins) and a learned RM. Contrastive pairs come from prompting a local
teacher with opposed principles (RLCD), so no human preference labels are needed.
**Question.** Which weights transfer across domains? Ablate one term at a time on the
held-out domains (train on 4 simulator domains, test on the other 2 + public sets).

## R4. The reward model is a Noul
The RM reuses the decision model itself, asking "this decision is correct, safe and
appropriately confident" as a Noul over `{state, question, proposed decision}`. One
architecture, one engine, and the RM inherits the calibration machinery.
**Question.** Does RM pairwise accuracy from a 1.7B menu scorer match a dedicated
sequence-classification RM at the same size?

## R5. Kernel-aware size/latency/calibration trade-off on GB10
Run Exp. 1-3 at 0.6B / 1.7B / 4B and at bf16 / FP8 / NVFP4, and plot accuracy, ECE and
p95 latency (`eval/latency.py`). The hypothesis is that at Spark's memory bandwidth, FP8
1.7B dominates bf16 4B on latency at equal calibration once temperatures are re-fit.

## R6. Calibration-aware agents
`examples/agent_gate.py`: use the calibrated Noul to route between act / full-LLM review /
human escalation. Measure the fraction of LLM calls avoided at a fixed risk budget vs. an
uncalibrated classifier with the same accuracy. This is where a well-calibrated System One
pays for itself.

## Prior art to cite
TypeSafe Jev (System One decision model; Choice/Score/Noul API), RLCD (Yang et al., 2023:
contrastive distillation for preference data), Qwen3 technical report, proper scoring rules
(Gneiting & Raftery 2007), temperature scaling (Guo et al. 2017), TensorRT-LLM PyTorch backend
and ModelOpt NVFP4 recipes, NVIDIA DGX Spark playbooks.
