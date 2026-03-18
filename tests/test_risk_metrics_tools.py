import pytest

pytest.importorskip("mcp.server.fastmcp")

from mcp_servers.risk_metrics import server as risk_server


def test_scenario_pnl_returns_structured_scenarios():
    result = risk_server.scenario_pnl(
        net_premium=23.98,
        total_delta=-0.0726,
        total_gamma=-0.0264,
        total_theta_per_day=0.4393,
        total_vega_per_vol_point=-0.6834,
        reference_price=300.0,
    )

    assert result["errors"] == []
    assert result["facts"]["worst_case_scenario"] == "stress"
    assert len(result["facts"]["scenarios"]) == 4
    assert result["quality"]["is_estimated"] is True


def test_portfolio_limit_check_flags_hard_breach():
    result = risk_server.portfolio_limit_check(
        metrics={"worst_case_pnl_pct": -0.14, "var_decimal": 0.09, "total_delta": 0.45},
        limits={"max_loss_pct": 0.10, "max_var_pct": 0.05, "max_abs_delta": 0.30},
    )

    assert result["errors"] == []
    assert result["facts"]["hard_limit_breached"] is True
    assert any(item["code"] == "LIMIT_BREACH_MAX_LOSS_PCT" for item in result["facts"]["breaches"])
    assert any(item["code"] == "WARNING_DIRECTIONAL_DELTA" for item in result["facts"]["warnings"])


def test_calculate_portfolio_greeks_returns_machine_usable_facts():
    result = risk_server.calculate_portfolio_greeks(
        [
            {
                "S": 300.0,
                "K": 300.0,
                "T_days": 30,
                "r": 0.05,
                "sigma": 0.35,
                "option_type": "call",
                "contracts": 1,
                "action": "sell",
            }
        ]
    )

    assert result["errors"] == []
    assert "total_delta" in result["facts"]
    assert result["facts"]["volatility_exposure"] in {"long_vol", "short_vol", "flat"}
