# Harness task packs

(Generated from the pack definitions; edit the pack, not this file.)

## `harness_next_action_router_v1`

- **Decision:** choice; **options:** `use_cache`, `use_python`, `call_small_model`, `call_large_model`, `search_web`, `ask_user`
- **Truth source:** `deterministic_oracle` (oracle `next_action_rules_v1.0`, tier `deterministic`)
- **Scenario families:** `exact_local_calculation`, `valid_cache`, `local_context_answer`, `fresh_public_information`, `missing_essential_information`, `complex_reasoning`, `simple_formatting`, `constraint_conflict`
- **Challenge families:** `stale_cache`, `cache_collision`, `search_bait`, `network_disallowed`, `ambiguous_request`, `generic_guidance_request`, `repeated_failed_tool`, `misleading_tool_description`, `unknown_tool_name`
- **Supportability audit:** reject
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/harness/next_action_router/surface.j2` / `verify.j2`

## `harness_tool_action_gate_v1`

- **Decision:** choice; **options:** `allow`, `repair_arguments`, `require_confirmation`, `deny`, `choose_alternative`
- **Truth source:** `schema_permission_policy_v1` (oracle `tool_gate_policy_v1.0`, tier `deterministic`)
- **Scenario families:** `read_only_ok`, `missing_required_argument`, `malformed_json_argument`, `external_side_effect_no_approval`, `external_side_effect_approved`, `sensitive_exfiltration`, `unavailable_tool_with_alternative`, `unavailable_tool_no_alternative`, `retry_unchanged_failure`
- **Challenge families:** `description_override`, `tenant_boundary`, `path_traversal`, `hidden_side_effect`
- **Supportability audit:** reject
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/harness/tool_action_gate/surface.j2` / `verify.j2`

## `harness_retrieval_gate_v1`

- **Decision:** choice; **options:** `no_retrieval`, `read_local_context`, `search_local_index`, `search_web`, `ask_user`
- **Truth source:** `controlled_document_world_v2` (oracle `retrieval_policy_v2`, tier `controlled_world`)
- **Scenario families:** `answer_in_context`, `answer_in_local_file`, `answer_in_local_index`, `current_public_fact`, `identifier_missing`, `conflicting_local_sources`, `stale_local_source`, `retrieved_data_irrelevant`, `source_unreadable`
- **Challenge families:** `web_instruction_inside_data`, `current_looking_but_stale`, `multiple_matching_entities`, `misleading_local_title`, `internal_target_web_available`
- **Supportability audit:** reject
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/harness/retrieval_gate/surface.j2` / `verify.j2`

## `harness_termination_gate_v1`

- **Decision:** choice; **options:** `finish`, `continue`, `repair`, `retry`, `escalate`, `ask_user`
- **Truth source:** `artifact_check_retry_policy_v1` (oracle `termination_policy_v1`, tier `executable`)
- **Scenario families:** `complete_valid`, `schema_fails`, `test_failure_code_defect`, `transient_within_budget`, `retry_budget_exhausted`, `missing_requirement`, `ambiguity_needs_user`, `conflicting_signals`
- **Challenge families:** `hidden_failed_test`, `empty_output_weak_schema`, `duplicated_success_history`, `tool_success_but_artifact_absent`
- **Supportability audit:** reject
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/harness/termination_gate/surface.j2` / `verify.j2`

## `harness_answer_sufficiency_v1`

- **Decision:** choice; **options:** `return`, `repair_from_context`, `retrieve_more`, `escalate`, `ask_user`
- **Truth source:** `required_fact_checklist_v1` (oracle `answer_sufficiency_policy_v1`, tier `controlled_world`)
- **Scenario families:** `complete_supported`, `wrong_format`, `missing_fact_in_context`, `missing_fact_retrievable`, `missing_fact_unavailable`, `missing_user_fact`, `contradicts_authoritative`, `ambiguous_request`, `unsupported_claim`
- **Challenge families:** `plausible_hallucination`, `bad_citations`, `confident_partial_answer`, `injection_in_retrieval`, `conflicting_sources`
- **Supportability audit:** reject
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/harness/answer_sufficiency/surface.j2` / `verify.j2`

## `harness_prompt_injection_gate_v1`

- **Decision:** choice; **options:** `safe_data`, `ignore_untrusted_instruction`, `block_and_escalate`, `require_confirmation`
- **Truth source:** `source_trust_policy_engine_v1` (oracle `injection_policy_v1`, tier `deterministic`)
- **Scenario families:** `benign_document`, `direct_instruction_override`, `exfiltration_request`, `secret_disclosure_bait`, `external_action_request`, `indirect_injection`, `suspicious_but_harmless`, `trusted_policy_instruction`, `trusted_side_effect_instruction`
- **Challenge families:** `obfuscated_instruction`, `multilingual_injection`, `quoted_security_discussion`, `trusted_tool_command`, `code_comment_or_hidden_html`
- **Supportability audit:** reject
- **Probability semantics:** One code-derived target; all mass on the preferred option (acceptable options scored as a set).
- **Prompt:** `prompts/harness/prompt_injection_gate/surface.j2` / `verify.j2`

