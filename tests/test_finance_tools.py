"""
Test Finance Tools: Exact benchmark-aligned output validation.
===============================================================
Asserts that the deterministic finance tools produce exact values
matching benchmark expected outputs. These tests lock the tool
outputs to prevent regression when the tools are modified.
"""

import pytest
pytest.importorskip("engine.mcp.server.fastmcp")

from engine.mcp.mcp_servers.finance.server import black_scholes_price, option_greeks, mispricing_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_structured(output: str) -> dict:
    """Extract key-value pairs from the STRUCTURED_RESULTS line."""
    lines = output.splitlines()
    assert lines[0] == "STRUCTURED_RESULTS:", f"Missing STRUCTURED_RESULTS header. Got: {lines[0]!r}"
    kv_line = lines[1]
    result = {}
    for pair in kv_line.split(";"):
        pair = pair.strip()
        if ":" in pair:
            k, v = pair.split(":", 1)
            result[k.strip()] = v.strip()
    return result


def _call(tool_fn, payload: dict) -> str:
    return tool_fn(**payload)


# ---------------------------------------------------------------------------
# black_scholes_price — opt_pricing_001 benchmark values
# AAPL: S=175, K=180, T=30d, r=5%, sigma=25% → call=3.22, put=7.48
# ---------------------------------------------------------------------------

class TestBlackScholesPrice:

    def test_has_structured_results_header(self):
        out = _call(black_scholes_price,
            {"S": 175.0, "K": 180.0, "T_days": 30, "r": 0.05, "sigma": 0.25}
        )
        assert out.startswith("STRUCTURED_RESULTS:"), "Output must start with STRUCTURED_RESULTS:"

    def test_call_price_matches_benchmark(self):
        out = _call(black_scholes_price,
            {"S": 175.0, "K": 180.0, "T_days": 30, "r": 0.05, "sigma": 0.25}
        )
        kv = _parse_structured(out)
        assert float(kv["call_price"]) == pytest.approx(3.22, abs=0.01), \
            f"call_price mismatch: {kv['call_price']}"

    def test_put_price_matches_benchmark(self):
        out = _call(black_scholes_price,
            {"S": 175.0, "K": 180.0, "T_days": 30, "r": 0.05, "sigma": 0.25}
        )
        kv = _parse_structured(out)
        assert float(kv["put_price"]) == pytest.approx(7.48, abs=0.01), \
            f"put_price mismatch: {kv['put_price']}"

    def test_method_field_present(self):
        out = _call(black_scholes_price,
            {"S": 175.0, "K": 180.0, "T_days": 30, "r": 0.05, "sigma": 0.25}
        )
        kv = _parse_structured(out)
        assert "Black-Scholes" in kv["method"], "method field must say Black-Scholes"

    def test_greeks_present_in_structured(self):
        out = _call(black_scholes_price,
            {"S": 175.0, "K": 180.0, "T_days": 30, "r": 0.05, "sigma": 0.25}
        )
        kv = _parse_structured(out)
        for greek in ("delta", "gamma", "theta", "vega"):
            assert greek in kv, f"Missing {greek} in STRUCTURED_RESULTS"

    def test_error_on_invalid_inputs(self):
        out = _call(black_scholes_price,
            {"S": -1.0, "K": 180.0, "T_days": 30, "r": 0.05, "sigma": 0.25}
        )
        assert out.startswith("Error"), "Should return Error for negative S"


# ---------------------------------------------------------------------------
# option_greeks — greeks_001 benchmark values
# TSLA: S=245, K=250, T=21d, r=5%, sigma=55%
# Expected: delta=0.474, gamma=0.012, theta=-0.321, vega=0.234
# ---------------------------------------------------------------------------

class TestOptionGreeks:

    def test_has_structured_results_header(self):
        out = _call(option_greeks,
            {"S": 245.0, "K": 250.0, "T_days": 21, "r": 0.05, "sigma": 0.55}
        )
        assert out.startswith("STRUCTURED_RESULTS:")

    def test_delta_matches_benchmark(self):
        out = _call(option_greeks,
            {"S": 245.0, "K": 250.0, "T_days": 21, "r": 0.05, "sigma": 0.55}
        )
        kv = _parse_structured(out)
        assert float(kv["delta"]) == pytest.approx(0.474, abs=0.002), \
            f"delta mismatch: {kv['delta']}"

    def test_gamma_matches_benchmark(self):
        out = _call(option_greeks,
            {"S": 245.0, "K": 250.0, "T_days": 21, "r": 0.05, "sigma": 0.55}
        )
        kv = _parse_structured(out)
        assert float(kv["gamma"]) == pytest.approx(0.012, abs=0.001), \
            f"gamma mismatch: {kv['gamma']}"

    def test_theta_matches_benchmark(self):
        out = _call(option_greeks,
            {"S": 245.0, "K": 250.0, "T_days": 21, "r": 0.05, "sigma": 0.55}
        )
        kv = _parse_structured(out)
        assert float(kv["theta"]) == pytest.approx(-0.321, abs=0.002), \
            f"theta mismatch: {kv['theta']}"

    def test_vega_matches_benchmark(self):
        out = _call(option_greeks,
            {"S": 245.0, "K": 250.0, "T_days": 21, "r": 0.05, "sigma": 0.55}
        )
        kv = _parse_structured(out)
        assert float(kv["vega"]) == pytest.approx(0.234, abs=0.002), \
            f"vega mismatch: {kv['vega']}"


# ---------------------------------------------------------------------------
# mispricing_analysis — opt_pricing_002 benchmark values
# NVDA: market=18.5, S=450, K=460, T=45d, r=5.25%, sigma=45%
# Expected: theoretical_price=25.18, assessment=underpriced, discrepancy_pct=-26.5
# ---------------------------------------------------------------------------

class TestMispricingAnalysis:

    def test_has_structured_results_header(self):
        out = _call(mispricing_analysis, {
            "market_price": 18.50, "S": 450.0, "K": 460.0,
            "T_days": 45, "r": 0.0525, "sigma": 0.45
        })
        assert out.startswith("STRUCTURED_RESULTS:")

    def test_theoretical_price_matches_benchmark(self):
        out = _call(mispricing_analysis, {
            "market_price": 18.50, "S": 450.0, "K": 460.0,
            "T_days": 45, "r": 0.0525, "sigma": 0.45
        })
        kv = _parse_structured(out)
        assert float(kv["theoretical_price"]) == pytest.approx(25.18, abs=0.05), \
            f"theoretical_price mismatch: {kv['theoretical_price']}"

    def test_assessment_is_underpriced(self):
        out = _call(mispricing_analysis, {
            "market_price": 18.50, "S": 450.0, "K": 460.0,
            "T_days": 45, "r": 0.0525, "sigma": 0.45
        })
        kv = _parse_structured(out)
        assert kv["assessment"] == "underpriced", \
            f"assessment mismatch: {kv['assessment']}"

    def test_discrepancy_pct_matches_benchmark(self):
        out = _call(mispricing_analysis, {
            "market_price": 18.50, "S": 450.0, "K": 460.0,
            "T_days": 45, "r": 0.0525, "sigma": 0.45
        })
        kv = _parse_structured(out)
        assert float(kv["discrepancy_pct"]) == pytest.approx(-26.5, abs=0.2), \
            f"discrepancy_pct mismatch: {kv['discrepancy_pct']}"

    def test_market_price_preserved(self):
        out = _call(mispricing_analysis, {
            "market_price": 18.50, "S": 450.0, "K": 460.0,
            "T_days": 45, "r": 0.0525, "sigma": 0.45
        })
        kv = _parse_structured(out)
        assert float(kv["market_price"]) == pytest.approx(18.50, abs=0.01)
