import pytest

pytest.importorskip("mcp.server.fastmcp")

from mcp_servers.finance_analytics.server import (
    bond_spread_duration,
    cashflow_waterfall,
    dcf_sensitivity_grid,
    du_pont_analysis,
    liquidity_ratio_pack,
    valuation_multiples_compare,
)


def test_du_pont_analysis_computes_roe_from_raw_inputs():
    result = du_pont_analysis(
        net_income=100.0,
        revenue=1000.0,
        average_assets=500.0,
        average_equity=250.0,
    )

    assert result["errors"] == []
    assert result["facts"]["net_margin_decimal"] == pytest.approx(0.1)
    assert result["facts"]["asset_turnover"] == pytest.approx(2.0)
    assert result["facts"]["equity_multiplier"] == pytest.approx(2.0)
    assert result["facts"]["roe_decimal"] == pytest.approx(0.4)


def test_liquidity_ratio_pack_returns_core_liquidity_metrics():
    result = liquidity_ratio_pack(
        current_assets=250.0,
        current_liabilities=100.0,
        inventory=40.0,
        cash=30.0,
        marketable_securities=10.0,
        receivables=50.0,
    )

    assert result["errors"] == []
    assert result["facts"]["working_capital"] == pytest.approx(150.0)
    assert result["facts"]["current_ratio"] == pytest.approx(2.5)
    assert result["facts"]["quick_ratio"] == pytest.approx(2.1)
    assert result["facts"]["cash_ratio"] == pytest.approx(0.4)


def test_valuation_multiples_compare_builds_peer_comparison_grid():
    result = valuation_multiples_compare(
        subject={"name": "Target", "ev_to_ebitda": 11.0, "pe": 22.0},
        peers=[
            {"name": "PeerA", "ev_to_ebitda": 10.0, "pe": 20.0},
            {"name": "PeerB", "ev_to_ebitda": 12.0, "pe": 24.0},
        ],
    )

    assert result["errors"] == []
    assert result["facts"]["peer_count"] == 2
    assert result["facts"]["metrics"]["ev_to_ebitda"]["peer_mean"] == pytest.approx(11.0)
    assert result["facts"]["metrics"]["pe"]["premium_discount_to_mean"] == pytest.approx(0.0)


def test_dcf_sensitivity_grid_marks_estimated_outputs():
    result = dcf_sensitivity_grid(
        base_fcf=100.0,
        growth_rates=[0.03, 0.05],
        discount_rates=[0.09, 0.11],
        terminal_growth_rate=0.02,
        years=5,
    )

    assert result["errors"] == []
    assert result["quality"]["is_estimated"] is True
    assert len(result["facts"]["grid"]) == 2
    assert len(result["facts"]["grid"][0]["values"]) == 2


def test_cashflow_waterfall_returns_structured_cash_bridge():
    result = cashflow_waterfall(
        operating_cash_flow=220.0,
        capex=80.0,
        taxes=20.0,
        debt_service=15.0,
        working_capital_change=10.0,
        dividends=25.0,
    )

    assert result["errors"] == []
    assert result["facts"]["free_cash_flow"] == pytest.approx(95.0)
    assert result["facts"]["residual_cash"] == pytest.approx(70.0)


def test_bond_spread_duration_decomposes_rate_and_spread_impact():
    result = bond_spread_duration(
        modified_duration=4.8,
        spread_duration=4.2,
        spread_change_bps=35,
        benchmark_yield_change_bps=-10,
    )

    assert result["errors"] == []
    assert result["quality"]["is_estimated"] is True
    assert result["facts"]["spread_price_impact_decimal"] < 0
