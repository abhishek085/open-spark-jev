# Foundation task packs

(Generated from the pack definitions; edit the pack, not this file.)

## `foundation_semantic_entailment_v1`

- **Decision:** choice; **options:** `entailed`, `contradicted`, `unknown`
- **Truth source:** `rule_engine_v1` (oracle `rule_engine_v1`, tier `deterministic`)
- **Scenario families:** `direct_fact`, `direct_contradiction`, `missing_evidence`, `multi_hop`, `distractor_facts`, `quantifier_set`, `source_conflict`, `ambiguous_wording`
- **Challenge families:** `double_negation`, `similar_entity`, `long_chain`, `salient_distractor`
- **Supportability audit:** tag
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/foundation/semantic_entailment/surface.j2` / `verify.j2`

## `foundation_document_type_v1`

- **Decision:** choice; **options:** `invoice`, `contract`, `support_request`, `report`, `meeting_note`, `receipt`, `other`
- **Truth source:** `structured_document_type_v1` (oracle `structured_document_type_v1`, tier `deterministic`)
- **Scenario families:** `invoice`, `receipt`, `contract`, `support_request`, `report`, `meeting_note`, `other`, `mixed_partial`, `misleading_filename`, `ocr_noise`, `multilingual_header`, `distractor_language`
- **Challenge families:** `invoice_ref_in_email`, `contract_amendment`, `receipt_with_refund`, `report_with_payment_table`, `truncated_scan`
- **Supportability audit:** tag
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/foundation/document_type/surface.j2` / `verify.j2`

## `foundation_document_relevance_v1`

- **Decision:** choice; **options:** `directly_relevant`, `partially_relevant`, `irrelevant`, `conflicting`
- **Truth source:** `controlled_corpus_coverage_v1` (oracle `controlled_corpus_relevance_v1`, tier `controlled_world`)
- **Scenario families:** `direct_answer`, `partial_answer`, `irrelevant_topic_overlap`, `wrong_time_period`, `conflicting_authoritative`, `stale_result`, `wrong_entity_same_name`, `multi_document_requirement`
- **Challenge families:** `keyword_overlap_no_answer`, `short_authoritative`, `misleading_title`, `injection_text_irrelevant`
- **Supportability audit:** tag
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/foundation/document_relevance/surface.j2` / `verify.j2`

## `foundation_extraction_validation_v1`

- **Decision:** choice; **options:** `correct`, `incorrect`, `incomplete`, `ambiguous`
- **Truth source:** `structured_fact_comparison_v1` (oracle `structured_fact_comparison_v1`, tier `controlled_world`)
- **Scenario families:** `dates_times`, `names_roles`, `amounts`, `addresses`, `identifiers`, `relations`, `omitted_required`, `swapped_roles`, `invalid_normalization`, `source_ambiguity`
- **Challenge families:** `multiple_dates`, `similar_names`, `quoted_prior_message`, `wrong_entity_value`, `ocr_errors`
- **Supportability audit:** tag
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/foundation/extraction_validation/surface.j2` / `verify.j2`

## `foundation_rule_application_v1`

- **Decision:** boolean; **options:** `true`, `false`
- **Truth source:** `rule_engine_v1` (oracle `rule_engine_boolean_v1`, tier `deterministic`)
- **Scenario families:** `implication_chain`, `conjunction`, `disjunction`, `exception_override`, `category_membership`, `numeric_threshold`, `nested_conditions`, `incomplete_premise`
- **Challenge families:** `distractor_rules`, `exception_chain`, `rule_order_variation`, `incomplete_premise_hard`
- **Supportability audit:** tag
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set). 'False' means 'not implied under closed-world rule application', not 'the proposition is disproved'.
- **Prompt:** `prompts/foundation/rule_application/surface.j2` / `verify.j2`

## `foundation_temporal_reasoning_v1`

- **Decision:** choice; **options:** `before`, `after`, `during`, `unknown`, `conflicting`
- **Truth source:** `python_datetime_event_graph_v1` (oracle `temporal_calc_v1`, tier `deterministic`)
- **Scenario families:** `deadlines`, `time_zones`, `relative_dates`, `durations`, `recurring_schedule`, `policy_window`, `before_after_ordering`, `missing_time`, `conflicting_sources`
- **Challenge families:** `dst_transition`, `date_only_vs_timestamp`, `conflicting_timestamps`, `long_event_chain`
- **Supportability audit:** tag
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/foundation/temporal_reasoning/surface.j2` / `verify.j2`

## `foundation_communication_intent_v1`

- **Decision:** choice; **options:** `request_information`, `request_action`, `report_problem`, `provide_update`, `schedule_or_reschedule`, `cancel_or_decline`, `feedback_or_complaint`, `other`
- **Truth source:** `structured_scenario_intent_policy_v1` (oracle `intent_policy_v1`, tier `controlled_world`)
- **Scenario families:** `direct_request`, `implicit_request`, `status_update`, `complaint`, `scheduling`, `cancellation`, `multi_intent`, `terse_chat`, `typo_heavy`, `formal_email`, `problem_report`, `information_request`
- **Challenge families:** `polite_complaint`, `urgent_information_request`, `multi_intent_primary_last`, `quoted_other_sender`, `controlled_ambiguity`
- **Supportability audit:** tag
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/foundation/communication_intent/surface.j2` / `verify.j2`

## `foundation_communication_urgency_v1`

- **Decision:** score; **options:** `0`, `1`, `2`, `3`
- **Truth source:** `declared_urgency_policy_v1` (oracle `urgency_policy_v1`, tier `controlled_world`)
- **Scenario families:** `broad_outage`, `single_user_issue`, `deadline_driven`, `safety_report`, `revenue_loss`, `workaround_available`, `partial_outage`, `missing_info`, `routine_request`
- **Challenge families:** `emotional_low_impact`, `calm_critical_outage`, `false_deadline`, `workaround_reduces`, `partial_outage_hard`, `missing_impact_info_hard`
- **Supportability audit:** reject
- **Probability semantics:** Ordinal rubric level from a declared points policy; neighbouring levels are closer than distant ones.
- **Prompt:** `prompts/foundation/communication_urgency/surface.j2` / `verify.j2`

## `foundation_authorization_gate_v1`

- **Decision:** choice; **options:** `allow`, `deny`, `require_approval`
- **Truth source:** `rbac_policy_engine_v1` (oracle `rbac_abac_policy_v1`, tier `deterministic`)
- **Scenario families:** `role_permission_match`, `least_privilege_violation`, `owner_approval_given`, `tenant_boundary`, `restricted_needs_approval`, `temporary_access`, `expired_access`, `delegated_access`, `purpose_limitation`
- **Challenge families:** `conflicting_role_claims`, `aggregate_with_prohibited_field`, `external_collaborator`, `inherited_permission`, `approval_exception`
- **Supportability audit:** reject
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/foundation/authorization_gate/surface.j2` / `verify.j2`

