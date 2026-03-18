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
