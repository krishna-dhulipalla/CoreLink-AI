import pytest

pytest.importorskip("engine.mcp.server.fastmcp")

from engine.mcp.mcp_servers.risk_metrics import server as risk_server


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


def test_factor_exposure_summary_groups_sector_risk():
    result = risk_server.factor_exposure_summary(
        [
            {"ticker": "AAPL", "weight": 0.35, "sector": "Technology"},
            {"ticker": "MSFT", "weight": 0.30, "sector": "Technology"},
            {"ticker": "XOM", "weight": 0.20, "sector": "Energy"},
        ]
    )

    assert result["errors"] == []
    assert result["facts"]["largest_factor"] in {"technology", "energy", "other"}
    assert result["facts"]["largest_factor_weight"] > 0


def test_drawdown_risk_profile_summarizes_drawdown_severity():
    result = risk_server.drawdown_risk_profile([0.02, -0.03, -0.04, 0.01, -0.02, 0.03])

    assert result["errors"] == []
    assert result["facts"]["drawdown_severity"] in {"contained", "moderate", "elevated"}
    assert result["facts"]["max_drawdown_decimal"] >= 0


def test_liquidity_stress_flags_tight_portfolio():
    result = risk_server.liquidity_stress(
        [
            {"ticker": "SMALL", "weight": 0.25, "liquidation_days": 8},
            {"ticker": "MID", "weight": 0.25, "liquidation_days": 6},
            {"ticker": "LARGE", "weight": 0.50, "liquidation_days": 2},
        ],
        redemption_pct=0.30,
    )

    assert result["errors"] == []
    assert result["facts"]["stress_assessment"] in {"manageable", "tight"}
