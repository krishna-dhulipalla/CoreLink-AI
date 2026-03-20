from langchain_core.messages import AIMessage

import agent.nodes.reviewer as reviewer_module
from agent.nodes.reviewer import reviewer
from staged_test_utils import make_state


def test_reviewer_revises_incomplete_legal_final():
    state = make_state(
        "Target company we're acquiring has EU and US compliance gaps.",
        task_profile="legal_transactional",
        capability_flags=["needs_legal_reasoning"],
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(AIMessage(content="Use a stock purchase with escrow."))

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "tax consequences" in [gap.lower() for gap in result["review_feedback"]["missing_dimensions"]]


def test_reviewer_passes_bare_numeric_quant_final_for_output_adapter():
    state = make_state(
        'Financial Leverage Effect = (ROE - ROA) / ROA. Output Format: {"answer": <value>}',
        task_profile="finance_quant",
        capability_flags=["needs_math", "requires_exact_format"],
        execution_template={
            "template_id": "quant_inline_exact",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": [],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        answer_contract={"format": "json", "requires_adapter": True, "wrapper_key": "answer"},
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(AIMessage(content="0.9273"))

    result = reviewer(state)

    assert result["solver_stage"] == "COMPLETE"
    assert result["review_feedback"] is None


def test_reviewer_terminates_repeated_unchanged_final_loop():
    state = make_state(
        "Summarize the acquisition structure options for this deal.",
        task_profile="legal_transactional",
        capability_flags=["needs_legal_reasoning"],
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_results": [],
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
            "repeat_signature": reviewer_module._artifact_signature("Use a stock purchase with escrow."),
            "repeat_count": 2,
            "last_review_reason": "Final legal answer is directionally correct but incomplete.",
            "last_review_verdict": "revise",
        },
    )
    state["messages"].append(AIMessage(content="Use a stock purchase with escrow."))

    result = reviewer(state)

    assert result["solver_stage"] == "COMPLETE"
    assert result["review_feedback"]["verdict"] == "revise"


def test_reviewer_revises_truncated_options_final():
    state = make_state(
        "Should I be a net buyer or seller of options?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(AIMessage(content="- Primary strategy:\n- Delta:"))

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert result["review_feedback"]["repair_target"] == "final"


def test_reviewer_requires_tool_backed_options_compute():
    state = make_state(
        "Should I be a net buyer or seller of options?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="COMPUTE",
        workpad={
            "events": [],
            "stage_outputs": {"COMPUTE": "IV premium is 0.07, so short vol makes sense."},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "COMPUTE",
        },
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "tool-backed strategy analysis" in [gap.lower() for gap in result["review_feedback"]["missing_dimensions"]]


def test_reviewer_revises_undisclosed_options_pricing_assumption():
    state = make_state(
        "Should I be a net buyer or seller of options?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {"net_premium": 9.16},
                    "assumptions": {},
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                }
            ],
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {"required_disclosures": [], "recommendation_class": "scenario_dependent_recommendation"},
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
        assumption_ledger=[
            {
                "key": "spot_price",
                "assumption": "Spot price was assumed as 300.0 from tool arguments because it was not explicit in prompt evidence.",
                "source": "tool_arguments:analyze_strategy",
                "confidence": "medium",
                "requires_user_visible_disclosure": True,
                "review_status": "pending",
            }
        ],
    )
    state["messages"].append(
        AIMessage(
            content=(
                "Recommendation: be a net seller of options.\n"
                "Primary strategy: short strangle.\n"
                "Alternative strategy comparison: iron condor reduces tail risk at lower premium.\n"
                "Key Greeks and breakevens: delta near flat, positive theta, breakevens around short strikes plus collected credit.\n"
                "Risk management: small sizing and defined stop-loss on large moves."
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "disclosed material assumptions" in [gap.lower() for gap in result["review_feedback"]["missing_dimensions"]]


def test_reviewer_revises_options_final_missing_risk_required_disclosures():
    state = make_state(
        "Should I be a net buyer or seller of options?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {"net_premium": 23.98, "total_vega_per_vol_point": -0.6834},
                    "assumptions": {},
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                }
            ],
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {
                "required_disclosures": ["Explicitly disclose short-volatility / volatility-spike risk."],
                "recommendation_class": "scenario_dependent_recommendation",
            },
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(
        AIMessage(
            content=(
                "Recommendation: net seller of options.\n"
                "Primary strategy: short straddle.\n"
                "Alternative strategy: iron condor.\n"
                "Key Greeks and breakevens: delta, theta, vega, breakeven.\n"
                "Risk management: use 1% position sizing and stop-loss."
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "required risk disclosures" in [gap.lower() for gap in result["review_feedback"]["missing_dimensions"]]


def test_reviewer_requires_compliance_guard_pass_for_options_final():
    state = make_state(
        "This is a defined-risk-only options mandate.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy", "scenario_pnl"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        evidence_pack={"policy_context": {"defined_risk_only": True, "action_orientation": True}},
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {"net_premium": 12.0},
                    "assumptions": {},
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                }
            ],
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {"required_disclosures": [], "recommendation_class": "scenario_dependent_recommendation"},
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(
        AIMessage(
            content=(
                "Recommendation: use a defined-risk iron condor.\n"
                "Primary strategy: iron condor.\n"
                "Alternative strategy comparison: vertical spread.\n"
                "Key Greeks and breakevens: delta, theta, vega, breakeven.\n"
                "Risk management: 1% position sizing and stop-loss.\n"
                "Recommendation class: scenario_dependent_recommendation."
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "compliance-guard validation" in [gap.lower() for gap in result["review_feedback"]["missing_dimensions"]]


def test_reviewer_passes_deterministic_options_final_with_hyphenated_disclosures():
    state = make_state(
        "Should I be a net buyer or seller of options?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 15.33,
                        "premium_direction": "CREDIT",
                        "total_delta": -0.0729,
                        "total_gamma": -0.025,
                        "total_theta_per_day": 0.4177,
                        "total_vega_per_vol_point": -0.6467,
                    },
                    "assumptions": {
                        "legs": [
                            {"option_type": "put", "action": "sell", "S": 300.0, "K": 290.0},
                            {"option_type": "call", "action": "sell", "S": 300.0, "K": 310.0},
                        ]
                    },
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                },
                {
                    "type": "scenario_pnl",
                    "facts": {"best_case_pnl": 0.42, "worst_case_pnl": -20.92},
                    "assumptions": {"reference_price": 300.0},
                    "source": {"tool": "scenario_pnl"},
                    "errors": [],
                },
            ],
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {
                "required_disclosures": [
                    "Explicitly disclose short-volatility / volatility-spike risk.",
                    "Explicitly disclose potentially unbounded tail loss and gap risk.",
                    "State downside scenario loss and the exit / sizing response.",
                ],
                "recommendation_class": "scenario_dependent_recommendation",
            },
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
        assumption_ledger=[
            {
                "key": "spot_price",
                "assumption": "Spot price was assumed as 300.0 from tool arguments because it was not explicit in prompt evidence.",
                "source": "tool_arguments:analyze_strategy",
                "confidence": "medium",
                "requires_user_visible_disclosure": True,
                "review_status": "pending",
            }
        ],
    )
    state["messages"].append(
        AIMessage(
            content=(
                "**Recommendation**\n"
                "Be a net seller of options.\n\n"
                "**Primary Strategy**\n"
                "Short strangle with credit premium of 15.33.\n\n"
                "**Alternative Strategy Comparison**\n"
                "Iron condor with defined wings is the cleaner alternative when you want lower premium but better tail-risk control.\n\n"
                "**Key Greeks and Breakevens**\n"
                "delta -0.073, gamma -0.025, theta 0.418/day, vega -0.647 per vol point.\n"
                "Breakevens: 274.67 / 325.33.\n\n"
                "**Risk Management**\n"
                "Use 1-2% position sizing, predefine a stop-loss at a breakeven breach or roughly a 1x premium loss, and hedge or reduce exposure if delta/gamma expands.\n"
                "Scenario summary: base-case P&L about 0.42; stress downside about -20.92.\n\n"
                "**Disclosures**\n"
                "- Assumption: Spot price was assumed as 300.0 from tool arguments because it was not explicit in prompt evidence.\n"
                "- Short-volatility / volatility-spike risk is material: losses can accelerate if implied volatility expands.\n"
                "- Tail loss and gap risk are material, especially if the underlying gaps through the short strikes.\n"
                "- Downside scenario loss is approximately -20.92; the exit / sizing response is to keep exposure at 1-2% of capital and cut risk on a breach.\n\n"
                "Recommendation class: scenario_dependent_recommendation."
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "COMPLETE"
    assert result["review_feedback"] is None


def test_reviewer_skips_llm_for_gather_stage_when_deterministic_checks_pass(monkeypatch):
    state = make_state(
        "Read the attached file and summarize the extracted evidence.",
        task_profile="document_qa",
        capability_flags=["needs_files"],
        execution_template={
            "template_id": "document_qa",
            "allowed_stages": ["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["fetch_reference_file"],
            "review_stages": ["GATHER", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        evidence_pack={
            "document_evidence": [
                {
                    "document_id": "countries_csv",
                    "status": "extracted",
                    "chunks": [{"locator": "Rows 0-5", "text": "Country,Region"}],
                    "tables": [{"headers": ["Country", "Region"], "rows": [["Algeria", "AFRICA"]]}],
                    "citation": "https://example.test/countries.csv",
                }
            ]
        },
        solver_stage="GATHER",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "fetch_reference_file",
                    "facts": {"document_id": "countries_csv"},
                    "assumptions": {},
                    "source": {"tool": "fetch_reference_file"},
                    "errors": [],
                }
            ],
            "review_ready": True,
            "review_stage": "GATHER",
        },
        last_tool_result={
            "type": "fetch_reference_file",
            "facts": {"document_id": "countries_csv"},
            "assumptions": {},
            "source": {"tool": "fetch_reference_file"},
            "errors": [],
        },
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("reviewer LLM should not run for deterministic gather pass")

    monkeypatch.setattr(reviewer_module, "invoke_structured_output", _should_not_run)

    result = reviewer(state)

    assert result["solver_stage"] == "SYNTHESIZE"
    assert result["review_feedback"] is None


def test_reviewer_skips_llm_for_deterministic_options_final_pass(monkeypatch):
    state = make_state(
        "Should I be a net buyer or seller of options?",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 15.33,
                        "premium_direction": "CREDIT",
                        "total_delta": -0.0729,
                        "total_gamma": -0.025,
                        "total_theta_per_day": 0.4177,
                        "total_vega_per_vol_point": -0.6467,
                    },
                    "assumptions": {"legs": [{"S": 300.0}]},
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                },
                {
                    "type": "scenario_pnl",
                    "facts": {"best_case_pnl": 0.42, "worst_case_pnl": -20.92},
                    "assumptions": {"reference_price": 300.0},
                    "source": {"tool": "scenario_pnl"},
                    "errors": [],
                },
            ],
            "risk_results": [{"verdict": "pass"}],
            "risk_requirements": {
                "required_disclosures": [
                    "Explicitly disclose short-volatility / volatility-spike risk.",
                    "Explicitly disclose potentially unbounded tail loss and gap risk.",
                    "State downside scenario loss and the exit / sizing response.",
                ],
                "recommendation_class": "scenario_dependent_recommendation",
            },
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
        assumption_ledger=[
            {
                "key": "spot_price",
                "assumption": "Spot price was assumed as 300.0 from tool arguments because it was not explicit in prompt evidence.",
                "source": "tool_arguments:analyze_strategy",
                "confidence": "medium",
                "requires_user_visible_disclosure": True,
                "review_status": "pending",
            }
        ],
    )
    state["messages"].append(
        AIMessage(
            content=(
                "**Recommendation**\n"
                "Be a net seller of options.\n\n"
                "**Primary Strategy**\n"
                "Short strangle with credit premium of 15.33.\n\n"
                "**Alternative Strategy Comparison**\n"
                "Iron condor with defined wings is the cleaner alternative when you want lower premium but better tail-risk control.\n\n"
                "**Key Greeks and Breakevens**\n"
                "delta -0.073, gamma -0.025, theta 0.418/day, vega -0.647 per vol point.\n"
                "Breakevens: 274.67 / 325.33.\n\n"
                "**Risk Management**\n"
                "Use 1-2% position sizing, predefine a stop-loss at a breakeven breach or roughly a 1x premium loss, and hedge or reduce exposure if delta/gamma expands.\n"
                "Scenario summary: base-case P&L about 0.42; stress downside about -20.92.\n\n"
                "**Disclosures**\n"
                "- Assumption: Spot price was assumed as 300.0 from tool arguments because it was not explicit in prompt evidence.\n"
                "- Short-volatility / volatility-spike risk is material: losses can accelerate if implied volatility expands.\n"
                "- Tail loss and gap risk are material, especially if the underlying gaps through the short strikes.\n"
                "- Downside scenario loss is approximately -20.92; the exit / sizing response is to keep exposure at 1-2% of capital and cut risk on a breach.\n\n"
                "Recommendation class: scenario_dependent_recommendation."
            )
        )
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("reviewer LLM should not run for deterministic options final pass")

    monkeypatch.setattr(reviewer_module, "invoke_structured_output", _should_not_run)

    result = reviewer(state)

    assert result["solver_stage"] == "COMPLETE"
    assert result["review_feedback"] is None


def test_reviewer_revises_equity_research_final_missing_template_dimensions():
    state = make_state(
        "Write an equity research report on MSFT.",
        task_profile="finance_quant",
        capability_flags=["needs_equity_research"],
        execution_template={
            "template_id": "equity_research_report",
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["get_company_fundamentals", "get_price_history"],
            "review_stages": ["GATHER", "COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [
                {
                    "type": "get_company_fundamentals",
                    "facts": {"fundamentals": {"trailingPE": 25.0, "revenueGrowth": 0.12}},
                    "assumptions": {"ticker": "MSFT"},
                    "source": {"tool": "get_company_fundamentals", "timestamp": "2024-10-14"},
                    "errors": [],
                }
            ],
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(AIMessage(content="Thesis: business quality remains solid. Evidence: revenue growth is healthy."))

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "valuation" in [gap.lower() for gap in result["review_feedback"]["missing_dimensions"]]


def test_reviewer_backtracks_bad_gather_result():
    state = make_state(
        "Read the attached file.",
        task_profile="document_qa",
        capability_flags=["needs_files"],
        execution_template={
            "template_id": "document_qa",
            "allowed_stages": ["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["fetch_reference_file", "list_reference_files"],
            "review_stages": ["GATHER", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="GATHER",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "GATHER",
        },
        last_tool_result={
            "type": "fetch_reference_file",
            "facts": {},
            "assumptions": {"url": "https://example.com/report.pdf"},
            "source": {"tool": "fetch_reference_file"},
            "errors": ["SSL failure"],
        },
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert result["review_feedback"]["verdict"] == "backtrack"


def test_reviewer_revises_document_gather_that_only_discovers_urls():
    state = make_state(
        "Read the attached report.",
        task_profile="document_qa",
        capability_flags=["needs_files"],
        execution_template={
            "template_id": "document_qa",
            "allowed_stages": ["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["fetch_reference_file", "list_reference_files"],
            "review_stages": ["GATHER", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        evidence_pack={
            "document_evidence": [
                {
                    "document_id": "report_pdf",
                    "citation": "https://example.com/report.pdf",
                    "status": "discovered",
                    "metadata": {"format": "pdf"},
                    "chunks": [],
                    "tables": [],
                    "numeric_summaries": [],
                }
            ]
        },
        solver_stage="GATHER",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "GATHER",
        },
        last_tool_result={
            "type": "list_reference_files",
            "facts": {"document_count": 1, "documents": [{"document_id": "report_pdf", "citation": "https://example.com/report.pdf"}]},
            "assumptions": {},
            "source": {"tool": "list_reference_files"},
            "errors": [],
        },
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "targeted document evidence" in [gap.lower() for gap in result["review_feedback"]["missing_dimensions"]]


def test_reviewer_revises_quant_final_that_is_not_scalar_for_json_contract():
    state = make_state(
        "Compute the ratio.",
        task_profile="finance_quant",
        capability_flags=["needs_math", "requires_exact_format"],
        answer_contract={"format": "json", "requires_adapter": True, "wrapper_key": "answer"},
        solver_stage="SYNTHESIZE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(AIMessage(content="Here is a full explanation of the ratio and why it matters."))

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "scalar answer matching output contract" in [gap.lower() for gap in result["review_feedback"]["missing_dimensions"]]


def test_reviewer_revises_live_data_quant_gather_without_tool_evidence():
    state = make_state(
        "As of 2024-10-14, retrieve MSFT price history and 1-month return.",
        task_profile="finance_quant",
        capability_flags=["needs_live_data"],
        execution_template={
            "template_id": "quant_with_tool_compute",
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["get_price_history", "get_returns"],
            "review_stages": ["GATHER", "COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="GATHER",
        workpad={
            "events": [],
            "stage_outputs": {"GATHER": "Need MSFT price history and return data."},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "GATHER",
        },
        last_tool_result=None,
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert "retrieval-backed finance evidence" in [gap.lower() for gap in result["review_feedback"]["missing_dimensions"]]


def test_reviewer_pass_from_document_gather_moves_to_synthesize_for_document_template():
    state = make_state(
        "Read the attached report and summarize the covenant breach.",
        task_profile="document_qa",
        capability_flags=["needs_files"],
        execution_template={
            "template_id": "document_qa",
            "allowed_stages": ["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["fetch_reference_file", "list_reference_files"],
            "review_stages": ["GATHER", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="GATHER",
        evidence_pack={
            "document_evidence": [
                {
                    "document_id": "report_pdf",
                    "citation": "https://example.com/report.pdf",
                    "status": "extracted",
                    "metadata": {"file_name": "report.pdf", "format": "pdf"},
                    "chunks": [{"locator": "Pages 1-2", "text": "metric value", "citation": "https://example.com/report.pdf"}],
                    "tables": [],
                    "numeric_summaries": [],
                }
            ]
        },
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "GATHER",
        },
        last_tool_result={
            "type": "fetch_reference_file",
            "facts": {"file_name": "report.pdf", "rows": [["metric", "value"]]},
            "assumptions": {"url": "https://example.com/report.pdf"},
            "source": {"tool": "fetch_reference_file"},
            "errors": [],
        },
    )

    result = reviewer(state)

    assert result["solver_stage"] == "SYNTHESIZE"


def test_reviewer_backtracks_options_compute_to_last_stable_checkpoint():
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template={
            "template_id": "options_tool_backed",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "COMPUTE",
            "allowed_tool_names": ["analyze_strategy"],
            "review_stages": ["COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="COMPUTE",
        evidence_pack={
            "derived_facts": {"iv_premium": 0.07, "analyze_strategy": {"net_premium": 9.16}},
            "document_evidence": [],
        },
        workpad={
            "events": [{"node": "solver", "action": "COMPUTE: branch after new tool"}],
            "stage_outputs": {"COMPUTE": "bad branch"},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {},
                    "assumptions": {},
                    "source": {"tool": "analyze_strategy"},
                    "errors": ["pricing failure"],
                }
            ],
            "review_ready": True,
            "review_stage": "COMPUTE",
            "draft_answer": "bad branch",
        },
        last_tool_result={
            "type": "analyze_strategy",
            "facts": {},
            "assumptions": {},
            "source": {"tool": "analyze_strategy"},
            "errors": ["pricing failure"],
        },
        checkpoint_stack=[
            {
                "template_id": "options_tool_backed",
                "checkpoint_stage": "COMPUTE",
                "reason": "stable_compute",
                "evidence_pack": {
                    "derived_facts": {"iv_premium": 0.07, "analyze_strategy": {"net_premium": 9.16}},
                    "document_evidence": [],
                },
                "assumption_ledger": [],
                "provenance_map": {},
                "last_tool_result": {
                    "type": "analyze_strategy",
                    "facts": {"net_premium": 9.16},
                    "assumptions": {},
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                },
                "draft_answer": "",
                "stage_outputs": {"COMPUTE": "stable compute summary"},
                "review_feedback": None,
            }
        ],
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert result["review_feedback"]["verdict"] == "backtrack"
    assert result["evidence_pack"]["derived_facts"]["analyze_strategy"]["net_premium"] == 9.16
    assert result["last_tool_result"]["errors"] == []
    assert result["workpad"]["stage_outputs"]["COMPUTE"] == "stable compute summary"


def test_reviewer_backtracks_quant_with_tool_compute_branch():
    state = make_state(
        "Compute the ratio from the attached sheet.",
        task_profile="finance_quant",
        capability_flags=["needs_math", "needs_files"],
        execution_template={
            "template_id": "quant_with_tool_compute",
            "allowed_stages": ["GATHER", "COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["calculator", "fetch_reference_file", "list_reference_files"],
            "review_stages": ["GATHER", "COMPUTE", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="COMPUTE",
        evidence_pack={"derived_facts": {"calculator": {"result": 0.9274}}, "document_evidence": []},
        workpad={
            "events": [],
            "stage_outputs": {"COMPUTE": "bad compute branch"},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "COMPUTE",
        },
        last_tool_result={
            "type": "calculator",
            "facts": {},
            "assumptions": {"expression": "bad"},
            "source": {"tool": "calculator"},
            "errors": ["division by zero"],
        },
        checkpoint_stack=[
            {
                "template_id": "quant_with_tool_compute",
                "checkpoint_stage": "COMPUTE",
                "reason": "stable_compute",
                "evidence_pack": {"derived_facts": {"calculator": {"result": 0.9274}}, "document_evidence": []},
                "assumption_ledger": [],
                "provenance_map": {},
                "last_tool_result": {
                    "type": "calculator",
                    "facts": {"result": 0.9274},
                    "assumptions": {"expression": "(a-b)/b"},
                    "source": {"tool": "calculator"},
                    "errors": [],
                },
                "draft_answer": "",
                "stage_outputs": {"COMPUTE": "stable compute"},
                "review_feedback": None,
            }
        ],
    )

    result = reviewer(state)

    assert result["review_feedback"]["verdict"] == "backtrack"
    assert result["evidence_pack"]["derived_facts"]["calculator"]["result"] == 0.9274


def test_reviewer_backtracks_document_gather_to_last_stable_checkpoint():
    state = make_state(
        "Read the attached report.",
        task_profile="document_qa",
        capability_flags=["needs_files"],
        execution_template={
            "template_id": "document_qa",
            "allowed_stages": ["GATHER", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "GATHER",
            "allowed_tool_names": ["fetch_reference_file", "list_reference_files"],
            "review_stages": ["GATHER", "SYNTHESIZE"],
            "review_cadence": "milestone_and_final",
            "answer_focus": [],
        },
        solver_stage="GATHER",
        evidence_pack={
            "document_evidence": [
                {
                    "document_id": "report_pdf",
                    "citation": "https://example.com/report.pdf",
                    "status": "discovered",
                    "metadata": {"format": "pdf"},
                    "chunks": [],
                    "tables": [],
                    "numeric_summaries": [],
                }
            ]
        },
        workpad={
            "events": [],
            "stage_outputs": {"GATHER": "bad gather"},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "GATHER",
        },
        last_tool_result={
            "type": "fetch_reference_file",
            "facts": {},
            "assumptions": {"url": "https://example.com/report.pdf"},
            "source": {"tool": "fetch_reference_file"},
            "errors": ["SSL failure"],
        },
        checkpoint_stack=[
            {
                "template_id": "document_qa",
                "checkpoint_stage": "GATHER",
                "reason": "baseline_artifacts",
                "evidence_pack": {
                    "document_evidence": [
                        {
                            "document_id": "report_pdf",
                            "citation": "https://example.com/report.pdf",
                            "status": "discovered",
                            "metadata": {"format": "pdf"},
                            "chunks": [],
                            "tables": [],
                            "numeric_summaries": [],
                        }
                    ]
                },
                "assumption_ledger": [],
                "provenance_map": {},
                "last_tool_result": None,
                "draft_answer": "",
                "stage_outputs": {},
                "review_feedback": None,
            }
        ],
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert result["review_feedback"]["verdict"] == "backtrack"
    assert result["last_tool_result"] is None
    assert result["evidence_pack"]["document_evidence"][0]["status"] == "discovered"


def test_reviewer_does_not_backtrack_legal_reasoning_only():
    state = make_state(
        "Advise on acquisition structure.",
        task_profile="legal_transactional",
        capability_flags=["needs_legal_reasoning"],
        execution_template={
            "template_id": "legal_reasoning_only",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "SYNTHESIZE",
            "allowed_tool_names": ["calculator"],
            "review_stages": ["SYNTHESIZE"],
            "review_cadence": "final_only",
            "answer_focus": [],
        },
        solver_stage="GATHER",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "GATHER",
        },
        last_tool_result={
            "type": "fetch_reference_file",
            "facts": {},
            "assumptions": {"url": "https://example.com/deal.pdf"},
            "source": {"tool": "fetch_reference_file"},
            "errors": ["blocked"],
        },
    )

    result = reviewer(state)

    assert result["solver_stage"] == "REVISE"
    assert result["review_feedback"]["verdict"] == "revise"


def test_reviewer_pass_does_not_checkpoint_legal_reasoning_only():
    state = make_state(
        "Advise on acquisition structure.",
        task_profile="legal_transactional",
        capability_flags=["needs_legal_reasoning"],
        execution_template={
            "template_id": "legal_reasoning_only",
            "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
            "default_initial_stage": "SYNTHESIZE",
            "allowed_tool_names": ["calculator"],
            "review_stages": ["SYNTHESIZE"],
            "review_cadence": "final_only",
            "answer_focus": [],
        },
        solver_stage="SYNTHESIZE",
        checkpoint_stack=[],
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "review_ready": True,
            "review_stage": "SYNTHESIZE",
        },
    )
    state["messages"].append(
        AIMessage(
            content=(
                "Structure options: asset purchase or reverse triangular merger.\n"
                "Tax consequences: basis step-up versus seller tax deferral.\n"
                "Liability protection: indemnities and escrow.\n"
                "Regulatory and diligence risks: EU and US compliance diligence.\n"
                "Key open questions and assumptions: seller cooperation and diligence findings.\n"
                "Recommended next steps: diligence and draft structure terms."
            )
        )
    )

    result = reviewer(state)

    assert result["solver_stage"] == "COMPLETE"
    assert result["checkpoint_stack"] == []
