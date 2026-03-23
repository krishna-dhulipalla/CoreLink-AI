import asyncio
import sys
import types

import pytest
from langchain_core.tools import tool

from agent.capabilities import build_capability_registry
from mcp_client import _parse_server_config
from mcp_client import load_mcp_tools_from_env

try:
    from executor import Executor
except ModuleNotFoundError:
    Executor = None


def test_parse_server_config_auto_adds_benchmark_judge(monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URLS", raising=False)
    monkeypatch.delenv("MCP_SERVER_STDIO", raising=False)
    monkeypatch.delenv("BENCHMARK_JUDGE_MCP_URL", raising=False)
    monkeypatch.delenv("JUDGE_MCP_URL", raising=False)
    monkeypatch.setenv("ENABLE_JUDGE_MCP_DISCOVERY", "1")

    config = _parse_server_config()

    assert config["judge"]["url"] == "http://judge:9009/mcp"
    assert config["judge"]["transport"] == "http"


def test_parse_server_config_normalizes_tools_suffix(monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URLS", raising=False)
    monkeypatch.delenv("MCP_SERVER_STDIO", raising=False)
    monkeypatch.setenv("ENABLE_JUDGE_MCP_DISCOVERY", "1")
    monkeypatch.setenv("BENCHMARK_JUDGE_MCP_URL", "http://judge:9009/mcp/tools")

    config = _parse_server_config()

    assert config["judge"]["url"] == "http://judge:9009/mcp"


def test_parse_server_config_prunes_irrelevant_stdio_servers_for_officeqa(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")
    monkeypatch.setenv(
        "MCP_SERVER_STDIO",
        "market_data=uv run python src/mcp_servers/market_data/server.py,file_handler=uv run python src/mcp_servers/file_handler/server.py",
    )
    monkeypatch.delenv("MCP_SERVER_URLS", raising=False)
    monkeypatch.setenv("ENABLE_JUDGE_MCP_DISCOVERY", "0")

    config = _parse_server_config(include_judge=False)

    assert "file_handler" in config
    assert "market_data" not in config


def test_build_capability_registry_infers_roles_for_external_document_tools():
    @tool
    def search_treasury_bulletins(query: str, top_k: int = 5) -> dict:
        """Search the Treasury Bulletin corpus for relevant documents."""
        return {}

    @tool
    def read_treasury_bulletin(document_id: str = "", url: str = "", page_start: int = 0, page_limit: int = 5) -> str:
        """Read a Treasury Bulletin PDF by document id or URL."""
        return ""

    registry = build_capability_registry([search_treasury_bulletins, read_treasury_bulletin])

    assert registry["search_treasury_bulletins"]["descriptor"]["tool_family"] == "document_retrieval"
    assert registry["search_treasury_bulletins"]["descriptor"]["tool_role"] == "search"
    assert registry["read_treasury_bulletin"]["descriptor"]["tool_family"] == "document_retrieval"
    assert registry["read_treasury_bulletin"]["descriptor"]["tool_role"] == "fetch"


def test_load_mcp_tools_from_env_keeps_working_servers_when_one_server_fails(monkeypatch):
    monkeypatch.delenv("BENCHMARK_NAME", raising=False)
    monkeypatch.delenv("MCP_SERVER_STDIO", raising=False)
    monkeypatch.setenv("MCP_SERVER_URLS", "local=http://local:9001/mcp")
    monkeypatch.setenv("ENABLE_JUDGE_MCP_DISCOVERY", "1")

    class FakeTool:
        def __init__(self, name: str):
            self.name = name
            self.description = name

    class FakeClient:
        def __init__(self, config):
            self._name = next(iter(config))

        async def get_tools(self):
            if self._name == "judge":
                raise RuntimeError("judge unavailable")
            return [FakeTool(f"{self._name}_tool")]

    pkg = types.ModuleType("langchain_mcp_adapters")
    client_mod = types.ModuleType("langchain_mcp_adapters.client")
    client_mod.MultiServerMCPClient = FakeClient
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", pkg)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", client_mod)

    tools = asyncio.run(load_mcp_tools_from_env())

    assert [tool.name for tool in tools] == ["local_tool"]


def test_executor_refreshes_even_when_nonempty_tools_exist_if_judge_discovery_is_enabled(monkeypatch):
    if Executor is None:
        pytest.skip("A2A runtime dependencies are not installed in this environment.")

    @tool
    def local_tool() -> str:
        """Local tool."""
        return "ok"

    async def fake_loader(*, include_judge: bool = True):
        return [local_tool]

    monkeypatch.setenv("ENABLE_JUDGE_MCP_DISCOVERY", "1")
    monkeypatch.setattr("executor.build_agent_graph", lambda external_tools=None: "graph")
    monkeypatch.setattr("executor.startup_compatibility_warnings", lambda: [])
    monkeypatch.setattr("executor.load_mcp_tools_from_env", fake_loader)

    executor = Executor()

    assert executor._mcp_tools
    assert executor._should_refresh_mcp_tools() is True
    executor._runtime_mcp_refresh_attempted = True
    assert executor._should_refresh_mcp_tools() is False
