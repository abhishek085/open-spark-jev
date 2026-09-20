# Run log

Every training / evaluation run that produced a number quoted anywhere in this repo, newest first.
One row per run: what was run, on what data, what came out, and where the raw artifact lives.
Analysis lives in [BENCHMARKS.md](BENCHMARKS.md); this file is the index, so a claim can always be
traced back to a command and a file.

Naming: the model is `spark-s1`; release ids are `spark-s1-<size>-<recipe>[-v<n>]`. Directory names
under `checkpoints/` are historical paths and do not always match the release id (see the mapping at
the bottom).

## 2026-09-19

| # | run | command | data | result | artifact |
|---|---|---|---|---|---|
| R14 | M17-LoRA: A0 LoRA recipe + real public text | `scripts/run_m17_lora.sh` | 16,000 records incl. 7,920 public, LoRA 1 epoch, 46 min | 60-set 0.750 (ECE 0.163) vs A0 0.717; injection worse than A0 (0.784->0.725, ECE 0.063->0.141); text, not adaptation method, is what hurts | `runs/external/spark-s1-1.7b-lora-m17/`, `runs/m17_lora.log`, BENCHMARKS.md B24 |
| R15 | ladder v4 A0 + in-domain tool-call corpus | `scripts/run_variants4.sh` | 12,000 records incl. 2,492 tool-call-risk | A0 60-set 0.750; injection-ctx 0.784->0.648; tool-call corpus found to be tool-name-degenerate (B25); A3/A4/A5 still running on it | `runs/variants4/`, `runs/external/v4-a0/`, `runs/rows60.json` |
| R16 | version ladder v0/v1/v3 on os-datagen splits | `scripts/run_v0.sh`, `scripts/run_v3.sh` | 967 train rows (x5 option orders), frozen bases and LoRA r16 at 0.6B/1.7B/4B | v3-4B test 0.785 / challenge 0.790 / ECE 0.087; 0.6B and 1.7B both 0.603; frozen bases 0.31-0.55 | `runs/osdg/`, `checkpoints/v3-{0.6b,1.7b,4b}`, BENCHMARKS.md B27 |
| R17 | v2 frozen head, v3 speed, v6 offline cascade | `scripts/run_v2.sh`, `scripts/analysis/{speed_osdg,cascade_v6}.py` | os-datagen splits | v2 refuted (0.314 / 0.512 test); v3 p50 19 / 35 / 82 ms; cascade no better than 4B alone | `runs/osdg/`, BENCHMARKS.md B28 |
| R18 | final review: all finished ladder models on the 60-row set | `scripts/analysis/final60.py` | ext-toolcall-risk | **v3-4B 0.850 at 65.9 ms** (Jev 0.917 at 421.6 ms); v3-1.7B 0.783 at 29.6 ms; frozen bases 0.57-0.73 | `runs/osdg/final60.json`, BENCHMARKS.md B29 |
| R19 | five API cases via POST /v1/decide | `scripts/api_bench/five_cases.py` | 5 hand-specified cases, 11 decisions | 4B 9/11 (NLL 0.52), 1.7B 7/11 (NLL 1.87); request p50 ~130 / ~62 ms | `runs/api_bench/`, BENCHMARKS.md B30 |
| R20 | v3 4B / 1.7B on the seven external suites | `scripts/run_ext_v3.sh` | 3,354 records, 7 sources | 4B best on 5/7 (Kev 0.773 / 0.749, directory 0.771, 60-set 0.850); vuln-code 0.56 unsolved; injection calibration worse (binary T=1.0) | `runs/external/spark-s1-v3-*-osdg-trained/`, BENCHMARKS.md B31 |
| R13 | M17: spark-s1-1.7b-sft-m17 (full FT + real public text) | `scripts/run_m17.sh` | 28,412 examples (+7,920 real public), 2 epochs, 2h08 | **60-set 0.733 (unchanged) at 30.0 ms vs Jev 0.917 at 421.6 ms (14.1x)**; +9/+7.5 on Kev suites but those overlap trained-on sources; flat on non-overlapping sources; injection ECE worse (0.059->0.159) | `runs/eval_m17.json`, `runs/external/spark-s1-1.7b-sft-m17/`, `runs/speed60_m17.json`, BENCHMARKS.md B23 |
| R12 | A0 baseline + A2 prefix-LM, matched budget | `scripts/run_variants3.sh` | 9,717 records, LoRA 1 epoch | **A2 refuted**: worse on all 9 measures, 30% slower. Unplanned: A0 (LoRA) beats the full fine-tune on 6/7 external sources | `runs/variants/{a0,a2}/result.json`, `runs/external/variant-{a0,a2}/`, BENCHMARKS.md B22 |
| R11 | external eval: spark-s1-1.7b-sft-v2 | `python -m open_spark_jev.eval.external --model checkpoints/sft-qwen3-1.7b` | 7 third-party sources, 3,354 records | acc 0.502-0.798 per source; ECE 0.023-0.365; **18-22 pts below Jev** where Jev's answers are recorded; directory pass rate 31/50 | `runs/external/spark-s1-1.7b-sft-v2/`, BENCHMARKS.md B21 |
| R10 | A2 trained + externally scored | `run_variants3.sh` (`full_a2`, `external_a2`) | 9,717 records, LoRA 1 epoch (38 min) | sim_test 0.791 / ECE 0.016 but external 0.296-0.600 - near chance off-distribution | `runs/variants/a2/result.json`, `runs/external/variant-a2/` |
| R9 | A0/A2/A3/A4 ladder v3 | `scripts/run_variants3.sh` | 12k records, LoRA, matched budget | A2 done (R10); A0 training; A3/A4 queued | `runs/variants/ladder.log` |
| R8 | mechanism speed benchmark | `python -m open_spark_jev.eval.speed_vs_generation --limit 60` | `ext-toolcall-risk` (60 rows), idle GPU | spark-s1 30.1 ms p50 / 0.733 acc; same backbone as JSON generator 374.7 ms / 0.433 (14 parse failures); with thinking 7165 ms / 0.550. **12.4x / 238x** speedup | `runs/speed_vs_generation.json`, BENCHMARKS.md B20 |
| R7 | external eval suites built | `python scripts/external/build_external.py` | 7 third-party sources | 3,354 records total; Jev's own answers recorded for 3 of them | `data/benchmarks/external/*.jsonl`, `data/external/*/PROVENANCE.md` |
| R6 | A1 parallel readout | `python -m open_spark_jev.experimental.parallel_readout --limit 200` | `sim_test`, spark-s1-1.7b-sft-v2 | answers match baseline to bf16 noise (max \|Δp\| 0.022); ~15% faster (91→76 ms at 1 question, 194→165 ms at 16) | `runs/variants/a1_parallel.log` |
| R5 | eval of spark-s1-1.7b-sft-v2 | `python -m open_spark_jev.eval.benchmark` | `sim_test` (3,000) + `teacher_test` (979) | **acc 0.838, ECE 0.017, Brier 0.243**, injection flip 0.033 | `runs/eval_v2_sft-qwen3-1.7b.json` |
| R4 | SFT v2 training | `scripts/train_sft.sh` | 26,291 records (leak-free simulators + 31-domain gemma corpus), 2 epochs | temps: choice 0.773 / noul 1.004 / score 0.907 | `checkpoints/sft-qwen3-1.7b`, `runs/logs_sft_v2_*.log` |
| R3 | leakage-free corpora regenerated | `osj simulate` | 18,000 train / 3,000 test | **0/500 overlap** on all six domains (was 97-99% on three) | `data/synthetic/sim_train.jsonl`, `data/benchmarks/sim_test.jsonl` |
| R2 | teacher corpus regenerated with gemma26b | `osj synth --distill --concurrency 8` | 31 domains x 300 | 9,270 records, 6,631 usable; teacher self-consistency 99.9% | `data/synthetic/teacher_train.jsonl` |
| R1 | contrastive pairs (gemma26b) | `python -m open_spark_jev.train.rlcd pairs` | 31 domains | 488 pairs | `data/synthetic/rlcd_pairs.jsonl` |

Interrupted, not counted as results: RLCD-contrastive RM stage (power loss at 07:16, see BENCHMARKS.md),
RLCD-direct v2 (stopped at step 100/1331 to free the GPU for the architecture ladder; requeued).

## Earlier runs (simulator-era, before the 31-domain data)

See [BENCHMARKS.md](BENCHMARKS.md) sections B6-B8d for the first four-way mechanism comparison
(SFT / RLCD-direct / RLCD-contrastive / GRPO) and "Teacher comparison" for the teacher benchmark. Those numbers were
produced on corpora with a known train/test leak in three of six domains (fixed in R3) and with the
qwen27b teacher (superseded in R2), so they are kept for history, not for quoting.

## Checkpoint directory -> release id

| directory | release id |
|---|---|
| `checkpoints/v3-4b` | `spark-s1-4b-v3` |
| `checkpoints/v3-1.7b` | `spark-s1-1.7b-v3` |
| `checkpoints/v3-0.6b` | `spark-s1-0.6b-v3` (not released; ties 1.7B) |
| `checkpoints/v2-{1.7b,4b}` | frozen-head experiments (refuted, B28) |
| `checkpoints/sft-qwen3-1.7b` | `spark-s1-1.7b-sft-v2` |
| `checkpoints/rlcd-direct-qwen3-1.7b` | `spark-s1-1.7b-rlcd-direct-v1` (v2 retraining queued) |
| `checkpoints/rlcd-contrastive-qwen3-1.7b` | `spark-s1-1.7b-rlcd-contrastive-v1` |
| `checkpoints/grpo-qwen3-1.7b` | `spark-s1-1.7b-grpo-v1` |
| `checkpoints/variants/a0\|a2\|a3\|a4` | `spark-s1-1.7b-a0-lora` … `-a4-evidential` (experiments) |

Directories will be renamed to match the release ids once the GPU is free; the registry in
`configs/serve/models.yaml` already carries the canonical names the UI and API report.
