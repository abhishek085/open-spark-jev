# Contribution ideas

Open Spark Jev is an early release and the fastest way to make it better is data, failures and integrations. Suggested labels: `good first issue`, `help wanted`, `integration`, `benchmark`, `safety`, `documentation`, `DGX Spark`, `UI`. Open an issue before starting anything large.

| # | Idea | Label | Notes |
|---|---|---|---|
| 1 | LangChain / LangGraph tool gate adapter | `integration` | Call `POST /v1/gate` before a tool node; refuse on `ask`/`deny` unless approved. |
| 2 | Pydantic AI integration example | `integration` | A tool-approval hook using `classify_tool_call`. |
| 3 | MCP tool-approval wrapper | `integration`, `safety` | See the design sketch in [INTEGRATIONS.md](INTEGRATIONS.md); must fail closed. |
| 4 | AutoGen / CrewAI adapter | `integration` | Same contract, different hook point. |
| 5 | More local deployment guides (other CUDA GPUs, Apple silicon, CPU) | `documentation` | We have only run on a DGX Spark. |
| 6 | Option-order invariance benchmark | `benchmark` | Reshuffle options and report answer flips per task. |
| 7 | Cloud-copy and `rclone` adversarial fixture pack | `safety`, `benchmark` | Backup-framed sync and copy commands (a known failure). |
| 8 | Tool-risk benchmark adapter | `benchmark` | Convert another public tool-risk set to our format, per-source, never pooled. |
| 9 | More Decision Lab fixture packs | `UI`, `good first issue` | Add JSON to `open_spark_jev/serve/lab_data/tool_calls.json` with expected floors and a test. |
| 10 | Documentation translations and diagrams | `documentation` | |
| 11 | Security-weakening and credential-exfiltration test cases | `safety` | The two largest known failure families; sanitised fixtures only. |
| 12 | A locked release holdout for tool-call risk | `benchmark`, `safety` | See [EVALUATION.md](EVALUATION.md#what-a-future-locked-release-holdout-should-contain). |

Safety failures are the most valuable contribution: use the **Safety failure** issue template. Never paste credentials, private customer data or proprietary commands.
