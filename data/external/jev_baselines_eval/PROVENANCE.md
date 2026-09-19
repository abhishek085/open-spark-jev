# Provenance

Downloaded 2026-09-19 from independent (non-TypeSafe-affiliated) researcher ickma2311's
public evaluation of Jev on Banking77 and CLINC150:
https://github.com/ickma2311/jev-baselines-eval

- `banking77_test.csv`, `banking77_train.csv`: the original PolyAI-LDN Banking77 dataset,
  fetched from the exact source URL that repo's README specifies
  (https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/).
- `results_b0.jsonl`: that repo's raw per-item results for its "B0" pilot (n=300 sampled,
  Jev/gpt-5.4-nano/GPT-5.6-Terra/a supervised encoder baseline). Jev's own subset is 208 of
  the 300 items (rate-limiting caused some drops during their run, per their README).

Not an official TypeSafe artifact. Reproduced here (not just linked) so the comparison in
docs/JEV_COMPARISON.md is independently auditable without depending on a third-party repo
staying online. See that repo's own README for full methodology, pre-registration hashes,
and review history.
