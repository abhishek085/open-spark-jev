# Launch assets

Nothing here is a claim beyond what [BENCHMARKS.md](BENCHMARKS.md) and [MODEL_CARD.md](../MODEL_CARD.md) support. The project is Open Spark Jev, part of Nokast, an open-source AI community; the model is `spark-s1`.

## Positioning

Open Spark Jev is an open, local System-1 decision engine for agent harnesses. Do not present it as a general-purpose model, as replacing Jev, or as a security boundary. It is a very early release that will improve as we add data.

## Demo flow, 60-90 seconds

1. `pip install -e ".[serve]"` and `osj lab` (no weights). Show the fixed output space (allow / ask / deny) before running anything.
2. Fixture `git status`: rules-only mode says `ask` ("no model, fail closed"). Switch the source to "Recorded real model output": the distribution appears, and the threshold decides.
3. Fixture `curl -d @/etc/shadow ...`: `deny`, with the policy trace naming the sensitive path and the upload.
4. Fixture `aws s3 sync ... s3://company-backup/`: the model may lean `allow`; the rule forces `ask`. Point at "Deterministic rule overrides the model".
5. Show the Benchmark view and say the caveats out loud (diagnostic set, hosted latency includes the network).

## Technical walkthrough, 5-7 minutes

Problem (generative decision loops are slow and brittle) -> the mechanism (one forward pass over option letters, [ARCHITECTURE.md](ARCHITECTURE.md)) -> the data (code-labelled, split by scenario family) -> live model in the Lab -> the two charts (Kev's out-of-domain suite and the 60-case set) -> known failures and the safety block -> what is next (more data, calibration for every question type, RLCD on unseen rows) -> how to contribute.

## Screen-recording checklist

- [ ] Clean terminal and browser profile, no credentials or private hostnames visible
- [ ] `osj lab` started from a fresh clone; the banner "Nothing here is executed" visible
- [ ] Zoom the browser to 125%; record at 1080p; keep the policy trace on screen
- [ ] Say "recorded" or "live" for every model output, and the model name `spark-s1-4b-v3`
- [ ] End on the limitations and the contribution links

## Messaging caveats (use verbatim where possible)

> The current 60-case result is a diagnostic benchmark, not a locked final holdout. Recorded hosted Jev latency includes network/service overhead and is not a pure local-inference comparison.

> On Kev's out-of-domain suite `spark-s1-4b-v3` is behind Kev-4B and Kev-8B and Jev overall. It was trained on under a thousand rows.

## Launch post draft

> I built an open local System-1 decision model for agent harnesses. It turns state plus fixed choices into typed decisions, probabilities, confidence, and an escalation signal, without generating text. The first release focuses on tool approval, routing, and safe agent-control loops. It includes runnable examples, benchmark code, a Decision Lab UI, DGX Spark notes, and known failure cases. It is not a standalone security boundary: sensitive or uncertain actions should use deterministic controls and human approval. The model is `spark-s1`, part of the Nokast open-source AI community. This is a very early release and we are generating more data to improve it.

## Images and GIFs needed for the README

- [x] Decision Lab screenshot with a decision run (`docs/img/decision-lab.png`)
- [x] Kev comparison chart (`docs/img/kev-comparison.png`)
- [x] 60-case comparison chart (`docs/img/toolcall-60-comparison.png`)
- [ ] 60-90 second demo GIF or video of the flow above
- [ ] Screenshot of the Benchmark view tab
- [ ] Screenshot of a policy override (the `aws s3 sync` fixture, "Deterministic rule overrides the model")
