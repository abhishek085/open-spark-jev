# Threat model and safety

- **Label leakage into the model-facing text** → deterministic leakage gate, verifier that never sees the label, feature-based worlds for packs whose label would otherwise be a fact.
- **Self-approval** (a model generating, labelling and approving) → truth from code; verifier is a different model/prompt; provenance marks correlated roles.
- **Prompt injection inside generated or retrieved text** → the verifier/judge treat candidate text as data; injection examples are harmless, fictional and contain no credentials, exploit payloads or operational wrongdoing.
- **Arbitrary code execution from generated text** → never: the sandbox executes only allowlisted operations or fixture-owned snippets, in a temp dir with limits and no network; external tools are mocks.
- **Secrets** → only env-var *names* are stored; raw generations are redacted (`retention.redact_patterns`) and size-capped; only fictional names/orgs/identifiers are used.
- **Split leakage** → world-level isolation, per-split pools/styles/wording, cross-split near-duplicate check, locked test never used for fitting.
- **Silent data loss** → every rejection is written with reasons; oracle failures abort the run.
