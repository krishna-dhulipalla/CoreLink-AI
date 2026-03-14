import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

a2a_client = pytest.importorskip("a2a.client")
a2a_types = pytest.importorskip("a2a.types")

A2ACardResolver = a2a_client.A2ACardResolver
ClientConfig = a2a_client.ClientConfig
ClientFactory = a2a_client.ClientFactory
Message = a2a_types.Message
Part = a2a_types.Part
Role = a2a_types.Role
TextPart = a2a_types.TextPart

from executor import Executor
from mcp_client import load_mcp_tools_from_env


def _configured_mcp() -> bool:
    return bool(
        os.getenv("MCP_SERVER_URLS", "").strip()
        or os.getenv("MCP_SERVER_STDIO", "").strip()
    )


def _live_agent_tests_enabled() -> bool:
    return os.getenv("RUN_LIVE_AGENT_TESTS", "").strip().lower() in {"1", "true", "yes"}


def _load_tools_or_skip():
    tools = asyncio.run(load_mcp_tools_from_env())
    if not tools:
        pytest.skip("Configured MCP servers are not reachable in this environment.")
    return tools


async def _send_text_message(
    text: str, url: str, context_id: str | None = None, streaming: bool = True
):
    async with httpx.AsyncClient(timeout=20) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        factory = ClientFactory(config)
        client = factory.create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
            context_id=context_id,
        )

        return [event async for event in client.send_message(msg)]


@pytest.mark.skipif(
    not _configured_mcp(),
    reason="No MCP server configured. Set MCP_SERVER_URLS or MCP_SERVER_STDIO.",
)
def test_mcp_tools_load_from_env():
    """Verify the configured MCP endpoint is reachable and exposes tools."""
    tools = _load_tools_or_skip()

    assert tools, (
        "Expected at least one MCP tool from the configured server. "
        "Check that the MCP server is running and the .env values are correct."
    )
    assert all(getattr(tool, "name", "").strip() for tool in tools), (
        "Every loaded MCP tool should have a non-empty name."
    )


@pytest.mark.skipif(
    not _configured_mcp(),
    reason="No MCP server configured. Set MCP_SERVER_URLS or MCP_SERVER_STDIO.",
)
def test_executor_registers_mcp_tools():
    """Verify the A2A executor wires discovered MCP tools into the graph."""
    expected_tools = _load_tools_or_skip()
    executor = Executor()

    loaded_names = {tool.name for tool in expected_tools}
    executor_names = {tool.name for tool in executor._mcp_tools}

    assert loaded_names, "MCP discovery returned no tools."
    assert executor_names == loaded_names, (
        "Executor should register the same MCP tools discovered from env."
    )
    assert executor._mcp_loaded, "Executor should mark MCP initialization as loaded."


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _configured_mcp() or not _live_agent_tests_enabled(),
    reason="Enable RUN_LIVE_AGENT_TESTS=1 and configure MCP/agent endpoints to run live A2A MCP tests.",
)
async def test_live_mcp_tool_visible_in_status_stream(agent):
    """Exercise the running A2A server and assert an MCP tool is advertised in status."""
    expected_tool = os.getenv("EXPECTED_MCP_TOOL_NAME", "").strip()
    prompt = os.getenv(
        "MCP_TEST_PROMPT",
        "Use the external MCP tool to answer this request.",
    ).strip()

    if not expected_tool:
        pytest.skip(
            "Set EXPECTED_MCP_TOOL_NAME to a concrete MCP tool name for "
            "live end-to-end verification."
        )

    events = await _send_text_message(prompt, agent, streaming=True)
    rendered = []

    for event in events:
        if isinstance(event, tuple):
            task, update = event
            if task:
                rendered.append(str(task.model_dump()))
            if update:
                rendered.append(str(update.model_dump()))
        else:
            rendered.append(str(event.model_dump()))

    full_stream = "\n".join(rendered)
    assert expected_tool in full_stream, (
        f"Expected streamed events to mention MCP tool '{expected_tool}'. "
        "If this fails, the model may be choosing a built-in tool or no tool. "
        "Use a prompt that uniquely requires the external MCP tool."
    )
