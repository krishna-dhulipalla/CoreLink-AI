from langchain_core.messages import AIMessage

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


def test_reviewer_backtracks_bad_gather_result():
    state = make_state(
        "Read the attached file.",
        task_profile="document_qa",
        capability_flags=["needs_files"],
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
