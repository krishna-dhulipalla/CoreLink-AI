"""
Compatibility facade for shared engine support helpers.

The active implementations live under ``agent.context``.
"""

from agent.context import evidence as _evidence
from agent.context.extraction import derive_market_snapshot, extract_as_of_date, extract_entities, extract_formulas, extract_urls, parse_markdown_tables
from agent.context.profiling import (
    _extract_labeled_json_block,
    allowed_tools_for_profile,
    allowed_tools_for_template,
    apply_profile_contract_rules,
    build_profile_decision,
    detect_ambiguity_flags,
    detect_capability_flags,
    extract_answer_contract,
    infer_task_profile,
    latest_human_text,
    normalize_whitespace,
    select_execution_template,
)
from agent.context.stages import (
    infer_task_complexity_tier,
    initial_solver_stage,
    initial_stage_for_template,
    next_stage_after_review,
    selective_backtracking_allowed,
    selective_checkpoint_policy,
    should_checkpoint_stage,
    stage_is_review_milestone,
)

_flatten_provenance = _evidence._flatten_provenance
_has_prompt_fact = _evidence._has_prompt_fact
_merge_unique_assumptions = _evidence._merge_unique_assumptions
artifact_checkpoint_from_state = _evidence.artifact_checkpoint_from_state
derive_assumption_ledger_entries = _evidence.derive_assumption_ledger_entries
merge_tool_result_into_evidence = _evidence.merge_tool_result_into_evidence


def extract_inline_facts(text: str):
    return _evidence.extract_inline_facts(text, labeled_json_extractor=_extract_labeled_json_block)


def build_evidence_pack(
    task_text,
    answer_contract,
    task_profile,
    capability_flags,
    ambiguity_flags=None,
):
    return _evidence.build_evidence_pack(
        task_text,
        answer_contract,
        task_profile,
        capability_flags,
        ambiguity_flags,
        labeled_json_extractor=_extract_labeled_json_block,
    )

__all__ = [
    "_extract_labeled_json_block",
    "_flatten_provenance",
    "_has_prompt_fact",
    "_merge_unique_assumptions",
    "allowed_tools_for_profile",
    "allowed_tools_for_template",
    "apply_profile_contract_rules",
    "artifact_checkpoint_from_state",
    "build_evidence_pack",
    "build_profile_decision",
    "derive_assumption_ledger_entries",
    "derive_market_snapshot",
    "detect_ambiguity_flags",
    "detect_capability_flags",
    "extract_answer_contract",
    "extract_as_of_date",
    "extract_entities",
    "extract_formulas",
    "extract_inline_facts",
    "extract_urls",
    "infer_task_complexity_tier",
    "infer_task_profile",
    "initial_solver_stage",
    "initial_stage_for_template",
    "latest_human_text",
    "merge_tool_result_into_evidence",
    "next_stage_after_review",
    "normalize_whitespace",
    "parse_markdown_tables",
    "select_execution_template",
    "selective_backtracking_allowed",
    "selective_checkpoint_policy",
    "should_checkpoint_stage",
    "stage_is_review_milestone",
]
