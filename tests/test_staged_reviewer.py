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
