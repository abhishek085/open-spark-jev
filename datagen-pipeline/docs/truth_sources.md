# Truth sources

| tier | mechanism here | packs |
|---|---|---|
| deterministic | rule/Datalog engine, date math (`zoneinfo`), points policy, ABAC/RBAC policy, tool schema/permission policy, feature-signature policy | entailment, rule application, temporal, urgency, authorization, doc type, tool gate, injection |
| executable | allowlisted sandbox (`csv_sum`, `csv_count`, `json_validate`, `sqlite_query_readonly`, `file_check`), mock tools | router (`use_python` on CSV fixtures), tool gate, termination |
| controlled world | corpus/coverage/authority/date facts, structured document facts | relevance, extraction validation, retrieval gate, answer sufficiency, intent |
| repeated rollout | `execution/rollouts.py` (Wilson lower bound vs required success probability) | router routes (model routes are declared profiles until real endpoints are used) |
| adjudicated | declared ambiguity policy (`label_quality: adjudicated`) — audit with humans | entailment `ambiguous_wording`, intent `controlled_ambiguity` |
| model consensus | not used as ground truth | — |

A single LLM label is never treated as truth.
