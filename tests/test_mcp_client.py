from langchain_core.tools import tool

from agent.capabilities import build_capability_registry
from mcp_client import _parse_server_config


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
