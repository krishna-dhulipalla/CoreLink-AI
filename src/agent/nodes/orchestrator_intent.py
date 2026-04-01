from __future__ import annotations

import re
from typing import Any

from agent.benchmarks import benchmark_task_intent
from agent.contracts import AnswerContract, ProgressSignature, TaskIntent
from agent.runtime_support import detect_ambiguity_flags, detect_capability_flags, infer_task_profile


def _legal_structure_option_count(answer: str) -> int:
    normalized = re.sub(r"\s+", " ", (answer or "").lower())
    patterns = {
        "asset_purchase": r"\basset purchase\b|\basset deal\b",
        "stock_purchase": r"\bstock purchase\b|\bshare purchase\b|\bequity purchase\b",
        "merger": r"\bmerger\b|\bforward triangular merger\b|\breverse triangular merger\b|\brtm\b",
        "carve_out": r"\bcarve[- ]out\b",
        "hybrid_structure": r"\bhybrid\b|\bcontingent consideration\b|\bearn[- ]?out\b",
        "joint_venture": r"\bjoint venture\b|\bpartnership\b",
        "minority_or_staged": r"\bminority investment\b|\bstaged acquisition\b|\boption to acquire\b",
        "commercial_structure": r"\blicen[sc]e\b|\bcommercial agreement\b|\bchannel partnership\b",
    }
    return sum(1 for pattern in patterns.values() if re.search(pattern, normalized))


def _opening_summary_window(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return text[:2200]
    opening = "\n\n".join(paragraphs[:3]).strip()
    return opening[:2200]


def _legal_frontload_gap(answer: str) -> bool:
    preview = _opening_summary_window(answer)
    return _legal_structure_option_count(preview) < 2


def _market_options_signal_count(task_text: str) -> int:
    normalized = (task_text or "").lower()
    signals = [
        any(token in normalized for token in ("iv percentile", "implied volatility", "historical volatility", "skew", "term structure")),
        any(token in normalized for token in ("delta", "gamma", "theta", "vega", "greeks")),
        any(token in normalized for token in ("straddle", "strangle", "iron condor", "credit spread", "vertical spread", "covered call")),
        any(token in normalized for token in ("expiry", "expiration", "days to expiry", "dte", "calls and puts")),
        bool(re.search(r"\b[A-Z]{1,5}\b", task_text or "")),
        any(token in normalized for token in ("premium", "breakeven", "vol seller", "vol buyer")),
    ]
    return sum(1 for signal in signals if signal)


def _looks_like_insurance_or_nonlisted_option(task_text: str) -> bool:
    normalized = (task_text or "").lower()
    return any(
        token in normalized
        for token in (
            "crop",
            "frost",
            "weather insurance",
            "insurance payout",
            "insurance contract",
            "farmer",
            "freeze",
        )
    )


def _supports_options_fast_path(task_text: str) -> bool:
    if _looks_like_insurance_or_nonlisted_option(task_text):
        return False
    return _market_options_signal_count(task_text) >= 2


def _looks_like_analytical_reasoning(task_text: str, capability_flags: list[str]) -> bool:
    normalized = (task_text or "").lower()
    if "needs_analytical_reasoning" in capability_flags:
        return True
    return any(
        token in normalized
        for token in (
            "differentiate",
            "derivative",
            "marginal cost",
            "marginal revenue",
            "integral",
            "optimize",
            "maximise",
            "maximize",
            "minimise",
            "minimize",
        )
    )


def _looks_like_market_scenario(task_text: str, capability_flags: list[str]) -> bool:
    normalized = (task_text or "").lower()
    if "needs_market_scenario" in capability_flags:
        return True
    return any(
        token in normalized
        for token in (
            "crypto",
            "flash crash",
            "liquidity crisis",
            "stress scenario",
            "scenario validation",
            "drawdown scenario",
        )
    )


def _looks_like_unsupported_artifact(task_text: str, capability_flags: list[str]) -> bool:
    normalized = (task_text or "").lower()
    if "needs_artifact_generation" in capability_flags:
        return True
    return any(
        token in normalized
        for token in (".wav", ".mp3", "audio file", "music producer", "render audio", "generate a track", "zip file")
    )


def _source_requests_live_data(task_text: str) -> bool:
    normalized = (task_text or "").lower()
    return any(
        token in normalized
        for token in ("latest", "today", "recent", "look up", "search", "source-backed", "current ", "as of ", "what was", "what is")
    )


def _looks_like_grounded_document_query(task_text: str) -> bool:
    normalized = (task_text or "").lower()
    return any(
        token in normalized
        for token in (
            "according to",
            "treasury bulletin",
            "bulletin",
            "annual report",
            "10-k",
            "10q",
            "filing",
            "document",
            "report",
            "table",
            "page",
            "source document",
        )
    )


def _normalize_tool_families(intent: TaskIntent, task_text: str) -> list[str]:
    normalized: list[str] = []
    mapping = {
        "finance_quant": "market_data_retrieval" if _source_requests_live_data(task_text) else "exact_compute",
        "mathematical_analysis": "analytical_reasoning",
        "transaction_structure_checklist": "transaction_structure_analysis",
        "regulatory_execution_checklist": "regulatory_execution_analysis",
        "tax_structure_checklist": "tax_structure_analysis",
        "risk_analysis": "market_scenario_analysis",
    }
    for family in intent.tool_families_needed:
        value = mapping.get(str(family), str(family))
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_task_intent(intent: TaskIntent, task_text: str, heuristic_intent: TaskIntent) -> TaskIntent:
    candidate = intent.model_copy(deep=True)
    if candidate.task_family == "finance_options" and not _supports_options_fast_path(task_text):
        if _looks_like_market_scenario(task_text, []):
            candidate.task_family = "market_scenario"
            candidate.execution_mode = "advisory_analysis"
            candidate.review_mode = "qualitative_advisory"
            candidate.completion_mode = "compact_sections"
            candidate.tool_families_needed = ["market_scenario_analysis"]
        elif _looks_like_analytical_reasoning(task_text, []):
            candidate.task_family = "analytical_reasoning"
            candidate.execution_mode = "advisory_analysis"
            candidate.review_mode = "analytical_reasoning"
            candidate.completion_mode = "long_form_derivation"
            candidate.tool_families_needed = ["analytical_reasoning", "exact_compute"]
        else:
            candidate.task_family = heuristic_intent.task_family
            candidate.execution_mode = heuristic_intent.execution_mode
            candidate.review_mode = heuristic_intent.review_mode
            candidate.completion_mode = heuristic_intent.completion_mode
            candidate.tool_families_needed = list(heuristic_intent.tool_families_needed)
    if candidate.task_family == "unsupported_artifact":
        candidate.execution_mode = "advisory_analysis"
        candidate.review_mode = "qualitative_advisory"
        candidate.completion_mode = "capability_gap"
        candidate.tool_families_needed = []
    if candidate.task_family in {"external_retrieval", "document_qa"} and _looks_like_grounded_document_query(task_text):
        candidate.execution_mode = "document_grounded_analysis"
        candidate.evidence_strategy = "document_first"
        candidate.review_mode = "document_grounded"
        candidate.completion_mode = "document_grounded"
        candidate.tool_families_needed = list(dict.fromkeys(["document_retrieval", *candidate.tool_families_needed, "external_retrieval"]))
    candidate.tool_families_needed = _normalize_tool_families(candidate, task_text)
    if candidate.task_family == "legal_transactional":
        candidate.tool_families_needed = list(
            dict.fromkeys(
                [
                    *candidate.tool_families_needed,
                    "legal_playbook_retrieval",
                    "transaction_structure_analysis",
                    "regulatory_execution_analysis",
                    "tax_structure_analysis",
                ]
            )
        )
    if candidate.task_family == "external_retrieval":
        candidate.tool_families_needed = list(dict.fromkeys([*candidate.tool_families_needed, "market_data_retrieval", "external_retrieval"]))
    if candidate.task_family == "analytical_reasoning":
        candidate.review_mode = "analytical_reasoning"
        candidate.completion_mode = "long_form_derivation"
        candidate.tool_families_needed = list(dict.fromkeys([*candidate.tool_families_needed, "analytical_reasoning", "exact_compute"]))
    if candidate.task_family == "market_scenario":
        candidate.tool_families_needed = list(dict.fromkeys([*candidate.tool_families_needed, "market_scenario_analysis"]))
        candidate.evidence_strategy = "scenario_first"
    return candidate


def _template_stub(intent: TaskIntent, allowed_tools: list[str] | None = None) -> dict[str, Any]:
    review_stages = ["SYNTHESIZE"]
    if intent.execution_mode == "exact_fast_path":
        review_stages = ["COMPUTE", "SYNTHESIZE"]
    return {
        "template_id": intent.execution_mode,
        "description": f"Execution mode: {intent.execution_mode}",
        "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
        "default_initial_stage": "COMPUTE" if intent.execution_mode == "exact_fast_path" else "SYNTHESIZE",
        "allowed_tool_names": list(allowed_tools or []),
        "review_stages": review_stages,
        "review_cadence": "milestone_and_final" if intent.execution_mode == "exact_fast_path" else "final_only",
        "answer_focus": [intent.routing_rationale] if intent.routing_rationale else [],
        "ambiguity_safe": intent.execution_mode == "exact_fast_path",
    }


def _heuristic_intent(
    task_text: str,
    answer_contract: dict[str, Any],
    benchmark_overrides: dict[str, Any] | None = None,
) -> tuple[TaskIntent, list[str], list[str]]:
    contract_obj = AnswerContract.model_validate(answer_contract or {})
    capability_flags = detect_capability_flags(task_text, contract_obj)
    ambiguity_flags = detect_ambiguity_flags(task_text, capability_flags)
    task_family = infer_task_profile(task_text, capability_flags)
    lowered = (task_text or "").lower()
    has_formula = "=" in task_text or "\\frac" in task_text
    has_table = "|---" in task_text
    exact_contract = bool(answer_contract.get("requires_adapter")) and answer_contract.get("format") == "json"

    if _looks_like_unsupported_artifact(task_text, capability_flags) or task_family == "unsupported_artifact":
        return (
            TaskIntent(
                task_family="unsupported_artifact",
                execution_mode="advisory_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=[],
                evidence_strategy="compact_prompt",
                review_mode="qualitative_advisory",
                completion_mode="capability_gap",
                routing_rationale="Prompt requests a non-finance artifact outside the active engine scope.",
                confidence=0.95,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    benchmark_intent = benchmark_task_intent(task_text, capability_flags, benchmark_overrides)
    if benchmark_intent is not None:
        return benchmark_intent, capability_flags, ambiguity_flags

    if task_family == "finance_quant" and has_formula and has_table and exact_contract and "needs_live_data" not in capability_flags:
        return (
            TaskIntent(
                task_family="finance_quant",
                execution_mode="exact_fast_path",
                complexity_tier="simple_exact",
                tool_families_needed=[],
                evidence_strategy="minimal_exact",
                review_mode="exact_quant",
                completion_mode="scalar_or_json",
                routing_rationale="Prompt contains formula, inline table evidence, and exact output contract.",
                confidence=0.98,
                planner_source="fast_path",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if task_family == "finance_options" and "needs_options_engine" in capability_flags and _supports_options_fast_path(task_text):
        return (
            TaskIntent(
                task_family="finance_options",
                execution_mode="tool_compute",
                complexity_tier="structured_analysis",
                tool_families_needed=["options_strategy_analysis", "options_scenario_analysis"],
                evidence_strategy="compact_prompt",
                review_mode="tool_compute",
                completion_mode="compact_sections",
                routing_rationale="Options tasks should use structured analysis tools before final synthesis.",
                confidence=0.93,
                planner_source="fast_path",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if _looks_like_market_scenario(task_text, capability_flags) or task_family == "market_scenario":
        return (
            TaskIntent(
                task_family="market_scenario",
                execution_mode="advisory_analysis",
                complexity_tier="complex_qualitative",
                tool_families_needed=["market_scenario_analysis"],
                evidence_strategy="scenario_first",
                review_mode="qualitative_advisory",
                completion_mode="compact_sections",
                routing_rationale="Scenario-driven trading and stress tasks need scenario analysis rather than generic advisory prose.",
                confidence=0.85,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if _looks_like_analytical_reasoning(task_text, capability_flags) or task_family == "analytical_reasoning":
        return (
            TaskIntent(
                task_family="analytical_reasoning",
                execution_mode="advisory_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=["analytical_reasoning", "exact_compute"],
                evidence_strategy="compact_prompt",
                review_mode="analytical_reasoning",
                completion_mode="long_form_derivation",
                routing_rationale="The task requires stepwise symbolic or numerical derivation instead of generic advisory output.",
                confidence=0.84,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if task_family == "legal_transactional":
        execution_mode = "document_grounded_analysis" if ("needs_files" in capability_flags or "http" in lowered) else "advisory_analysis"
        tool_families = [
            "legal_playbook_retrieval",
            "transaction_structure_analysis",
            "regulatory_execution_analysis",
            "tax_structure_analysis",
        ]
        if execution_mode == "document_grounded_analysis":
            tool_families.insert(0, "document_retrieval")
        return (
            TaskIntent(
                task_family="legal_transactional",
                execution_mode=execution_mode,
                complexity_tier="complex_qualitative",
                tool_families_needed=tool_families,
                evidence_strategy="document_first" if execution_mode == "document_grounded_analysis" else "compact_prompt",
                review_mode="qualitative_advisory",
                completion_mode="advisory_memo",
                routing_rationale="Legal structure/risk/compliance questions need broader checklist and retrieval capabilities.",
                confidence=0.88,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if _looks_like_grounded_document_query(task_text) and task_family not in {"finance_options", "legal_transactional"}:
        return (
            TaskIntent(
                task_family="document_qa" if task_family == "general" else task_family,
                execution_mode="document_grounded_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=["document_retrieval", "external_retrieval"],
                evidence_strategy="document_first",
                review_mode="document_grounded",
                completion_mode="document_grounded",
                routing_rationale="The task appears grounded in reports or source documents, so use retrieval before answering.",
                confidence=0.83,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if task_family == "document_qa":
        return (
            TaskIntent(
                task_family="document_qa",
                execution_mode="document_grounded_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=["document_retrieval"],
                evidence_strategy="document_first",
                review_mode="document_grounded",
                completion_mode="document_grounded",
                routing_rationale="Document questions should ground the answer in retrieved file evidence.",
                confidence=0.82,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    if task_family == "external_retrieval":
        return (
            TaskIntent(
                task_family="external_retrieval",
                execution_mode="retrieval_augmented_analysis",
                complexity_tier="structured_analysis",
                tool_families_needed=["market_data_retrieval", "external_retrieval"],
                evidence_strategy="retrieval_first",
                review_mode="document_grounded",
                completion_mode="compact_sections",
                routing_rationale="The prompt explicitly requests current or source-backed information.",
                confidence=0.8,
                planner_source="heuristic",
            ),
            capability_flags,
            ambiguity_flags,
        )

    return (
        TaskIntent(
            task_family=task_family,  # type: ignore[arg-type]
            execution_mode="advisory_analysis",
            complexity_tier="complex_qualitative" if ambiguity_flags else "structured_analysis",
            tool_families_needed=["market_data_retrieval"] if _source_requests_live_data(task_text) else [],
            evidence_strategy="compact_prompt",
            review_mode="qualitative_advisory",
            completion_mode="compact_sections",
            routing_rationale="No high-confidence fast path was detected; use a general advisory flow.",
            confidence=0.68,
            planner_source="heuristic",
        ),
        capability_flags,
        ambiguity_flags,
    )
