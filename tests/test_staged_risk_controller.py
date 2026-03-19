from agent.nodes.risk_controller import risk_controller
from staged_test_utils import make_state


def _options_template():
    return {
        "template_id": "options_tool_backed",
        "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
        "default_initial_stage": "COMPUTE",
        "allowed_tool_names": ["analyze_strategy", "scenario_pnl", "portfolio_limit_check"],
        "review_stages": ["COMPUTE", "SYNTHESIZE"],
        "review_cadence": "milestone_and_final",
        "answer_focus": [],
    }


def _portfolio_template():
    return {
        "template_id": "portfolio_risk_review",
        "allowed_stages": ["COMPUTE", "SYNTHESIZE", "REVISE", "COMPLETE"],
        "default_initial_stage": "COMPUTE",
        "allowed_tool_names": ["concentration_check", "drawdown_risk_profile", "portfolio_limit_check"],
        "review_stages": ["COMPUTE", "SYNTHESIZE"],
        "review_cadence": "milestone_and_final",
        "answer_focus": [],
    }


def test_risk_controller_revises_options_compute_without_scenario_coverage():
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template=_options_template(),
        solver_stage="COMPUTE",
        workpad={
            "events": [],
            "stage_outputs": {"COMPUTE": "Primary strategy: short straddle credit with negative vega."},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 23.98,
                        "total_delta": -0.0726,
                        "total_gamma": -0.0264,
                        "total_theta_per_day": 0.4393,
                        "total_vega_per_vol_point": -0.6834,
                    },
                    "assumptions": {
                        "legs": [
                            {"option_type": "call", "action": "sell", "S": 300.0, "K": 300.0},
                            {"option_type": "put", "action": "sell", "S": 300.0, "K": 300.0},
                        ]
                    },
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                }
            ],
            "review_ready": True,
            "review_stage": "COMPUTE",
        },
        last_tool_result={
            "type": "analyze_strategy",
            "facts": {"net_premium": 23.98, "total_vega_per_vol_point": -0.6834},
            "assumptions": {"legs": [{"option_type": "call", "action": "sell"}, {"option_type": "put", "action": "sell"}]},
            "source": {"tool": "analyze_strategy"},
            "errors": [],
        },
    )

    result = risk_controller(state)

    assert result["solver_stage"] == "REVISE"
    assert result["risk_feedback"]["verdict"] == "revise"
    assert "MISSING_SCENARIO_ANALYSIS" in result["risk_feedback"]["violation_codes"]
    assert "MISSING_RISK_CONTROLS" in result["risk_feedback"]["violation_codes"]


def test_risk_controller_passes_with_scenario_and_controls():
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template=_options_template(),
        solver_stage="COMPUTE",
        workpad={
            "events": [],
            "stage_outputs": {
                "COMPUTE": (
                    "Primary strategy is a short straddle. Use 1% position sizing and a stop-loss if the downside "
                    "scenario breaches tolerance."
                )
            },
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 23.98,
                        "total_delta": -0.0726,
                        "total_gamma": -0.0264,
                        "total_theta_per_day": 0.4393,
                        "total_vega_per_vol_point": -0.6834,
                    },
                    "assumptions": {
                        "legs": [
                            {"option_type": "call", "action": "sell", "S": 300.0, "K": 300.0},
                            {"option_type": "put", "action": "sell", "S": 300.0, "K": 300.0},
                        ]
                    },
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                },
                {
                    "type": "scenario_pnl",
                    "facts": {
                        "worst_case_pnl": -48.0,
                        "worst_case_scenario": "stress",
                        "scenarios": [{"name": "stress", "approx_pnl": -48.0}],
                    },
                    "assumptions": {"reference_price": 300.0},
                    "source": {"tool": "scenario_pnl"},
                    "errors": [],
                },
            ],
            "review_ready": True,
            "review_stage": "COMPUTE",
        },
        last_tool_result={
            "type": "scenario_pnl",
            "facts": {"worst_case_pnl": -48.0},
            "assumptions": {"reference_price": 300.0},
            "source": {"tool": "scenario_pnl"},
            "errors": [],
        },
    )

    result = risk_controller(state)

    assert result["risk_feedback"] is None
    assert result["workpad"]["risk_requirements"]["required_disclosures"]
    assert result["workpad"]["risk_requirements"]["recommendation_class"] == "scenario_dependent_recommendation"
    assert result["workpad"]["risk_results"][-1]["verdict"] == "pass"


def test_risk_controller_accepts_risk_cap_language_as_control():
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template=_options_template(),
        solver_stage="COMPUTE",
        workpad={
            "events": [],
            "stage_outputs": {
                "COMPUTE": (
                    "Primary strategy is a short straddle. Keep a risk cap in force, cut risk on a breach, "
                    "and use a stop loss if the stress scenario moves outside tolerance."
                )
            },
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {
                        "net_premium": 23.98,
                        "total_delta": -0.0726,
                        "total_gamma": -0.0264,
                        "total_theta_per_day": 0.4393,
                        "total_vega_per_vol_point": -0.6834,
                    },
                    "assumptions": {
                        "legs": [
                            {"option_type": "call", "action": "sell", "S": 300.0, "K": 300.0},
                            {"option_type": "put", "action": "sell", "S": 300.0, "K": 300.0},
                        ]
                    },
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                },
                {
                    "type": "scenario_pnl",
                    "facts": {
                        "worst_case_pnl": -48.0,
                        "worst_case_scenario": "stress",
                        "scenarios": [{"name": "stress", "approx_pnl": -48.0}],
                    },
                    "assumptions": {"reference_price": 300.0},
                    "source": {"tool": "scenario_pnl"},
                    "errors": [],
                },
            ],
            "review_ready": True,
            "review_stage": "COMPUTE",
        },
        last_tool_result={
            "type": "scenario_pnl",
            "facts": {"worst_case_pnl": -48.0},
            "assumptions": {"reference_price": 300.0},
            "source": {"tool": "scenario_pnl"},
            "errors": [],
        },
    )

    result = risk_controller(state)

    assert result["risk_feedback"] is None
    assert result["workpad"]["risk_results"][-1]["verdict"] == "pass"


def test_risk_controller_blocks_limit_breach():
    state = make_state(
        "Compare volatility-selling strategies for META.",
        task_profile="finance_options",
        capability_flags=["needs_options_engine"],
        execution_template=_options_template(),
        solver_stage="COMPUTE",
        workpad={
            "events": [],
            "stage_outputs": {"COMPUTE": "Use 1% position sizing and a hard stop-loss."},
            "tool_results": [
                {
                    "type": "analyze_strategy",
                    "facts": {"net_premium": 23.98, "total_vega_per_vol_point": -0.6834},
                    "assumptions": {"legs": [{"option_type": "call", "action": "sell"}]},
                    "source": {"tool": "analyze_strategy"},
                    "errors": [],
                },
                {
                    "type": "portfolio_limit_check",
                    "facts": {
                        "hard_limit_breached": True,
                        "breaches": [{"code": "LIMIT_BREACH_MAX_LOSS_PCT"}],
                        "warnings": [],
                    },
                    "assumptions": {"limits": {"max_loss_pct": 0.1}},
                    "source": {"tool": "portfolio_limit_check"},
                    "errors": [],
                },
            ],
            "review_ready": True,
            "review_stage": "COMPUTE",
        },
        last_tool_result={
            "type": "portfolio_limit_check",
            "facts": {"hard_limit_breached": True, "breaches": [{"code": "LIMIT_BREACH_MAX_LOSS_PCT"}]},
            "assumptions": {},
            "source": {"tool": "portfolio_limit_check"},
            "errors": [],
        },
    )

    result = risk_controller(state)

    assert result["solver_stage"] == "REVISE"
    assert result["risk_feedback"]["verdict"] == "blocked"
    assert "LIMIT_BREACH_MAX_LOSS_PCT" in result["risk_feedback"]["violation_codes"]


def test_risk_controller_revises_portfolio_review_without_actions():
    state = make_state(
        "Review portfolio concentration and recommend actions.",
        task_profile="finance_quant",
        capability_flags=["needs_portfolio_risk"],
        execution_template=_portfolio_template(),
        solver_stage="COMPUTE",
        workpad={
            "events": [],
            "stage_outputs": {"COMPUTE": "Concentration and drawdown metrics were computed."},
            "tool_results": [
                {
                    "type": "concentration_check",
                    "facts": {
                        "has_breach": True,
                        "name_breaches": [{"name": "AAPL", "weight": 0.35}],
                        "sector_breaches": [],
                    },
                    "assumptions": {},
                    "source": {"tool": "concentration_check"},
                    "errors": [],
                }
            ],
            "review_ready": True,
            "review_stage": "COMPUTE",
        },
        last_tool_result={
            "type": "concentration_check",
            "facts": {"has_breach": True},
            "assumptions": {},
            "source": {"tool": "concentration_check"},
            "errors": [],
        },
    )

    result = risk_controller(state)

    assert result["solver_stage"] == "REVISE"
    assert result["risk_feedback"]["verdict"] == "revise"
    assert "MISSING_PORTFOLIO_ACTIONS" in result["risk_feedback"]["violation_codes"]


def test_risk_controller_passes_portfolio_review_with_actions_and_limits_clear():
    state = make_state(
        "Review portfolio concentration and recommend actions.",
        task_profile="finance_quant",
        capability_flags=["needs_portfolio_risk"],
        execution_template=_portfolio_template(),
        solver_stage="COMPUTE",
        workpad={
            "events": [],
            "stage_outputs": {
                "COMPUTE": (
                    "Main risk is technology concentration. Recommended actions: trim AAPL, rebalance sector weights, "
                    "and keep a drawdown-based hedge trigger."
                )
            },
            "tool_results": [
                {
                    "type": "concentration_check",
                    "facts": {
                        "has_breach": True,
                        "name_breaches": [{"name": "AAPL", "weight": 0.35}],
                        "sector_breaches": [{"sector": "Technology", "weight": 0.65}],
                    },
                    "assumptions": {},
                    "source": {"tool": "concentration_check"},
                    "errors": [],
                },
                {
                    "type": "drawdown_risk_profile",
                    "facts": {"max_drawdown_decimal": 0.12, "drawdown_severity": "moderate"},
                    "assumptions": {},
                    "source": {"tool": "drawdown_risk_profile"},
                    "errors": [],
                },
                {
                    "type": "portfolio_limit_check",
                    "facts": {"hard_limit_breached": False, "breaches": [], "warnings": []},
                    "assumptions": {},
                    "source": {"tool": "portfolio_limit_check"},
                    "errors": [],
                },
            ],
            "review_ready": True,
            "review_stage": "COMPUTE",
        },
        last_tool_result={
            "type": "portfolio_limit_check",
            "facts": {"hard_limit_breached": False, "breaches": [], "warnings": []},
            "assumptions": {},
            "source": {"tool": "portfolio_limit_check"},
            "errors": [],
        },
    )

    result = risk_controller(state)

    assert result["risk_feedback"] is None
    assert result["workpad"]["risk_results"][-1]["verdict"] == "pass"
