# Research agenda

What in open-spark-Jev is a contribution rather than an integration, and the experiments that
would establish it. Each experiment is runnable from this repo.

## R1. Exact menu policy gradient vs. sampled-token RL
**Claim.** For single-step decisions over a small menu, the policy distribution is available in
closed form, so the RL objective `E_a~p[R(a)] + R_dist(p) - β KL(p‖p_ref)` can be
differentiated exactly. This removes sampling variance, makes PPO/GRPO machinery
unnecessary, and, more importantly, lets the reward include **proper scoring rules evaluated on
the distribution itself** (Brier vs. posterior), which sampled-action RL cannot express.
**Prediction.** Sampled GRPO with 0/1 correctness improves accuracy but *worsens* ECE
(argmax collapse); the exact objective improves both.
**Run.** `scripts/train_rlcd.sh` (exact) vs. `python -m open_spark_jev.train.rlcd_grpo`
(sampled), evaluate both with `scripts/eval.sh`, compare `ece`, `soft_brier_vs_posterior`.

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
