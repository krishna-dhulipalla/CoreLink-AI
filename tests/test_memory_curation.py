from engine.agent.memory.curation import build_curation_signals, summarize_curation_signals


def test_build_curation_signals_emits_review_assumption_and_tool_patterns():
    state = {
        "task_profile": "finance_options",
        "execution_template": {"template_id": "options_tool_backed"},
        "messages": [],
        "assumption_ledger": [
            {
                "key": "spot_price",
                "assumption": "Spot price was assumed as 300.",
                "source": "tool_arguments:analyze_strategy",
                "confidence": "medium",
                "requires_user_visible_disclosure": True,
                "review_status": "pending",
            }
        ],
    }
    workpad = {
        "review_results": [
            {
                "review_stage": "SYNTHESIZE",
                "verdict": "revise",
                "repair_target": "final",
                "missing_dimensions": ["Risk management", "Key Greeks and breakevens"],
                "reasoning": "Final options answer is missing risk management and Greek detail.",
                "is_final": True,
            }
        ],
        "tool_results": [
            {
                "type": "analyze_strategy",
                "facts": {},
                "assumptions": {},
                "source": {"tool": "analyze_strategy", "solver_stage": "COMPUTE"},
                "errors": ["Spot price missing from tool arguments."],
            }
        ],
    }

    signals = build_curation_signals(state, "Analyze the options trade.", workpad)
    signal_types = {signal.signal_type for signal in signals}

    assert "missing_dimension" in signal_types
    assert "assumption_issue" in signal_types
    assert "tool_failure" in signal_types


def test_build_curation_signals_does_not_flag_disclosed_assumption_or_pass_evidence_note():
    state = {
        "task_profile": "finance_options",
        "execution_template": {"template_id": "options_tool_backed"},
        "messages": [
            type("Msg", (), {"content": "Assumption Disclosure: spot price assumed at 100 for the strategy."})()
        ],
        "assumption_ledger": [
            {
                "key": "spot_price",
                "assumption": "Spot price was assumed as 100.",
                "source": "tool_arguments:analyze_strategy",
                "confidence": "medium",
                "requires_user_visible_disclosure": True,
                "review_status": "pending",
            }
        ],
    }
    workpad = {
        "draft_answer": "",
        "review_results": [
            {
                "review_stage": "GATHER",
                "verdict": "pass",
                "repair_target": "synthesize",
                "missing_dimensions": [],
                "reasoning": "Gather output contains structured document evidence.",
                "is_final": False,
            }
        ],
        "tool_results": [],
    }

    signals = build_curation_signals(state, "Analyze the options trade.", workpad)

    assert len(signals) == 1
    assert signals[0].signal_type == "assumption_issue"
    assert signals[0].success is True


def test_summarize_curation_signals_produces_stable_recommendations_for_repeated_failures():
    signals = [
        {
            "task_profile": "legal_transactional",
            "task_family": "legal",
            "template_id": "legal_reasoning_only",
            "signal_type": "missing_dimension",
            "signal_key": "tax_consequences",
            "summary": "SYNTHESIZE missing dimension: Tax consequences",
            "stage": "SYNTHESIZE",
            "count_hint": 1,
            "metadata": {"dimension": "Tax consequences"},
        },
        {
            "task_profile": "legal_transactional",
            "task_family": "legal",
            "template_id": "legal_reasoning_only",
            "signal_type": "missing_dimension",
            "signal_key": "tax_consequences",
            "summary": "SYNTHESIZE missing dimension: Tax consequences",
            "stage": "SYNTHESIZE",
            "count_hint": 1,
            "metadata": {"dimension": "Tax consequences"},
        },
        {
            "task_profile": "finance_options",
            "task_family": "finance",
            "template_id": "options_tool_backed",
            "signal_type": "assumption_issue",
            "signal_key": "spot_price",
            "summary": "Spot price was assumed as 300.",
            "stage": "COMPUTE",
            "count_hint": 1,
            "metadata": {"source": "tool_arguments:analyze_strategy"},
        },
        {
            "task_profile": "finance_options",
            "task_family": "finance",
            "template_id": "options_tool_backed",
            "signal_type": "assumption_issue",
            "signal_key": "spot_price",
            "summary": "Spot price was assumed as 300.",
            "stage": "COMPUTE",
            "count_hint": 1,
            "metadata": {"source": "tool_arguments:analyze_strategy"},
        },
    ]

    summary = summarize_curation_signals(signals, min_count=2)

    assert summary["recommendation_count"] == 2
    assert summary["recommendations"][0]["signal_key"] == "spot_price"
    assert summary["recommendations"][1]["signal_key"] == "tax_consequences"
