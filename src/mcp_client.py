"""
MCP Client: Dynamic Tool Discovery
====================================
Connects to external MCP servers (Green Agents) and loads their tools
as LangChain-compatible tool objects for use in the LangGraph agent.

Configuration:
    Set MCP_SERVER_URLS in your .env file as comma-separated name=url pairs:
        MCP_SERVER_URLS=math=http://localhost:3000/mcp,weather=http://localhost:8000/mcp

    For stdio-based local servers, set MCP_SERVER_STDIO as comma-separated name=command:arg pairs:
        MCP_SERVER_STDIO=math=python:/path/to/math_server.py

Architecture Reference: docs/DESIGN.md (Section 2 - Foundational Reasoning)
"""

import os
import logging
import shlex
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _parse_legacy_stdio_args(cmd_args: str) -> tuple[str, list[str]]:
    """Parse legacy colon-delimited stdio config.

    Supports the original `command:arg1:arg2` format and repairs a Windows
    drive-letter path in the first argument, e.g.:

        python:C:\\path\\to\\server.py
    """
    parts = cmd_args.strip().split(":")
    command = parts[0]
    args = parts[1:] if len(parts) > 1 else []

    if len(args) >= 2 and len(args[0]) == 1 and args[1].startswith("\\"):
        args = [f"{args[0]}:{args[1]}", *args[2:]]

    return command, args


def _parse_stdio_command(cmd_args: str) -> tuple[str, list[str]]:
    """Parse stdio config from either shell-style or legacy colon syntax.

    Preferred format:
        name=python src/mock_mcp_server.py --transport stdio

    Backward-compatible format:
        name=python:src/mock_mcp_server.py
    """
    raw = cmd_args.strip()
    if not raw:
        return "", []

    if any(ch.isspace() for ch in raw) or '"' in raw or "'" in raw:
        parts = shlex.split(raw, posix=False)
        if not parts:
            return "", []
        return parts[0], parts[1:]

    return _parse_legacy_stdio_args(raw)


def _parse_server_config() -> dict[str, dict[str, Any]]:
    """Parse MCP server configurations from environment variables.

    Returns a dict compatible with MultiServerMCPClient constructor:
        {
            "server_name": {
                "url": "http://...",
                "transport": "http",
            },
            ...
        }
    """
    config: dict[str, dict[str, Any]] = {}

    # HTTP-based MCP servers (for remote Green Agents)
    http_urls = os.getenv("MCP_SERVER_URLS", "").strip()
    if http_urls:
        for entry in http_urls.split(","):
            entry = entry.strip()
            if "=" not in entry:
                logger.warning(f"Skipping malformed MCP_SERVER_URLS entry: {entry}")
                continue
            name, url = entry.split("=", 1)
            config[name.strip()] = {
                "url": url.strip(),
                "transport": "http",
            }

    # Stdio-based MCP servers (for local development)
    stdio_entries = os.getenv("MCP_SERVER_STDIO", "").strip()
    if stdio_entries:
        for entry in stdio_entries.split(","):
            entry = entry.strip()
            if "=" not in entry:
                logger.warning(f"Skipping malformed MCP_SERVER_STDIO entry: {entry}")
                continue
            name, cmd_args = entry.split("=", 1)
            command, args = _parse_stdio_command(cmd_args)
            if not command:
                logger.warning(f"Skipping empty MCP_SERVER_STDIO entry: {entry}")
                continue
            config[name.strip()] = {
                "command": command,
                "args": args,
                "transport": "stdio",
            }

    return config


async def load_mcp_tools_from_env() -> list:
    """Load tools from all MCP servers configured via environment variables.

    Returns:
        A list of LangChain-compatible tool objects.
        Returns an empty list if no MCP servers are configured or on error.
    """
    config = _parse_server_config()
    if not config:
        logger.info("No MCP servers configured. Using built-in tools only.")
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        logger.info(f"Connecting to MCP servers: {list(config.keys())}")
        client = MultiServerMCPClient(config)
        tools = await client.get_tools()
        logger.info(
            f"Loaded {len(tools)} tools from MCP servers: "
            f"{[t.name for t in tools]}"
        )
        return tools

    except ImportError:
        logger.error(
            "langchain-mcp-adapters not installed. "
            "Run: uv add langchain-mcp-adapters"
        )
        return []
    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}")
        return []
