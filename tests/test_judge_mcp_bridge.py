import asyncio

import httpx
import pytest

from agent.model_config import get_model_name
from judge_mcp_bridge import JudgeMcpConnectionError, call_judge_tool, discover_judge_tools, validate_tool_call


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    calls: list[tuple[str, str, dict | None, dict | None]] = []
    get_payload = []
    post_payload = {}

    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        self.calls.append(("GET", url, params, None))
        return _FakeResponse(self.get_payload)

    async def post(self, url, json=None):
        self.calls.append(("POST", url, None, json))
        return _FakeResponse(self.post_payload)


class _FailingAsyncClient:
    def __init__(self, timeout=None):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None):
        raise httpx.ConnectError("[Errno 11001] getaddrinfo failed")


def test_discover_judge_tools_uses_session_scoped_tools_endpoint(monkeypatch):
    monkeypatch.setenv("ENABLE_JUDGE_MCP_DISCOVERY", "1")
    monkeypatch.setenv("BENCHMARK_JUDGE_MCP_URL", "http://judge:9009/mcp/tools")
    monkeypatch.setattr("judge_mcp_bridge.httpx.AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.get_payload = [{"name": "search_treasury_bulletins", "description": "Search bulletins"}]

    tools = asyncio.run(discover_judge_tools(session_id="session-123"))

    assert tools[0]["name"] == "search_treasury_bulletins"
    assert _FakeAsyncClient.calls[0][0] == "GET"
    assert _FakeAsyncClient.calls[0][1] == "http://judge:9009/mcp/tools"
    assert _FakeAsyncClient.calls[0][2] == {"session_id": "session-123"}


def test_call_judge_tool_validates_required_params_before_network():
    tools = [
        {
            "name": "read_treasury_bulletin",
            "inputSchema": {
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
            },
        }
    ]

    valid, error = validate_tool_call("read_treasury_bulletin", {}, tools)

    assert valid is False
    assert "missing required params" in error


def test_call_judge_tool_posts_expected_payload(monkeypatch):
    monkeypatch.setenv("ENABLE_JUDGE_MCP_DISCOVERY", "1")
    monkeypatch.setenv("BENCHMARK_JUDGE_MCP_URL", "http://judge:9009/mcp")
    monkeypatch.setattr("judge_mcp_bridge.httpx.AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.post_payload = {"result": {"value": "2602"}}

    tools = [
        {
            "name": "read_treasury_bulletin",
            "inputSchema": {
                "type": "object",
                "properties": {"document_id": {"type": "string"}},
                "required": ["document_id"],
            },
        }
    ]

    payload = asyncio.run(
        call_judge_tool(
            "read_treasury_bulletin",
            {"document_id": "tb_1940"},
            session_id="session-456",
            tools_list=tools,
        )
    )

    assert payload["result"]["value"] == "2602"
    assert _FakeAsyncClient.calls[0][0] == "POST"
    assert _FakeAsyncClient.calls[0][1] == "http://judge:9009/mcp"
    assert _FakeAsyncClient.calls[0][3] == {
        "tool": "read_treasury_bulletin",
        "params": {"document_id": "tb_1940"},
        "session_id": "session-456",
    }


def test_discover_judge_tools_wraps_connection_failures_with_actionable_guidance(monkeypatch):
    monkeypatch.setenv("ENABLE_JUDGE_MCP_DISCOVERY", "1")
    monkeypatch.setenv("BENCHMARK_JUDGE_MCP_URL", "http://judge:9009/mcp")
    monkeypatch.setattr("judge_mcp_bridge.httpx.AsyncClient", _FailingAsyncClient)

    with pytest.raises(JudgeMcpConnectionError) as excinfo:
        asyncio.run(discover_judge_tools(session_id="session-xyz"))

    message = str(excinfo.value)
    assert "Judge MCP discovery failed" in message
    assert "BENCHMARK_JUDGE_MCP_URL" in message
    assert "benchmark Docker network" in message


def test_officeqa_benchmark_uses_officeqa_model_profile(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    monkeypatch.delenv("MODEL_PROFILE", raising=False)
    monkeypatch.delenv("SOLVER_MODEL", raising=False)
    monkeypatch.delenv("REVIEWER_MODEL", raising=False)

    assert get_model_name("solver") == "deepseek-ai/DeepSeek-V3.2"
    assert get_model_name("reviewer") == "Qwen/Qwen3-32B-fast"
