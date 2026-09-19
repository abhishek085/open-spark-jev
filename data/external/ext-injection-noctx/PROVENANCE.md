# ext-injection-noctx

- **Upstream**: https://github.com/Gaurav-Gosain/jev-sec-bench
- **Pinned commit**: `fdb16b94d37535db9bad77f8ef0faa971bd7d69a`
- **License**: MIT (harness and results). Underlying corpora keep their own licenses: deepset/prompt-injections, CyberNative/Code_Vulnerability_Security_DPO.
- **Retrieved**: 2026-09-19
- **Records written**: 662  (`data/benchmarks/external/ext-injection-noctx.jsonl`)

## What it is
The same 662 messages with the deployment context removed.

## Where the labels come from
Labels come from the public corpora (deepset injection label; vulnerable = 'rejected' side, fixed = 'chosen' side of the DPO pairs).

## Recorded Jev output
Yes, per row: Jev's P(yes) (`probability`) from `jev-1.13.0`, run 2026-09-16. Author reports 96.5% accuracy (with context) and ECE 0.0588 for injection.

## Caveats
Our prompt wording is a faithful paraphrase of the upstream Go battery, not byte-identical (their Yes/No criteria are passed as structured fields). The injection labels only make sense given the news-assistant context (see the ctx vs no-context pair). Code snippets are long; ones over our token limit are dropped at eval time.

## What we changed
Only the Noul question is used; upstream's severity Score is not.
