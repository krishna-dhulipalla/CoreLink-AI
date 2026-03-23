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

load_dotenv(override=False)

logger = logging.getLogger(__name__)


def _normalize_mcp_url(url: str) -> str:
    normalized = (url or "").strip().rstrip("/")
    if normalized.endswith("/tools"):
        normalized = normalized[: -len("/tools")]
    return normalized


def judge_mcp_discovery_enabled() -> bool:
    return os.getenv("ENABLE_JUDGE_MCP_DISCOVERY", "1").strip().lower() not in {"0", "false", "no", "off"}


def _benchmark_name() -> str:
    return os.getenv("BENCHMARK_NAME", "").strip().lower()


def _officeqa_server_allowed(name: str, server_config: dict[str, Any]) -> bool:
    if name == "judge":
        return True
    flattened = " ".join(
        [
            name,
            str(server_config.get("url", "") or ""),
            str(server_config.get("command", "") or ""),
            " ".join(str(item) for item in (server_config.get("args", []) or [])),
        ]
    ).lower()
    return any(
        token in flattened
        for token in (
            "file",
            "document",
            "doc",
            "corpus",
            "reference",
            "treasury",
            "bulletin",
            "archive",
            "pdf",
        )
    )


def _filter_server_config_for_benchmark(config: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if _benchmark_name() != "officeqa":
        return config
    filtered = {
        name: server_config
        for name, server_config in config.items()
        if _officeqa_server_allowed(name, server_config)
    }
    if filtered != config:
        logger.info("Pruned MCP servers for OfficeQA benchmark: kept=%s dropped=%s", list(filtered.keys()), [name for name in config if name not in filtered])
    return filtered


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


def _parse_server_config(*, include_judge: bool = True) -> dict[str, dict[str, Any]]:
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
                "url": _normalize_mcp_url(url),
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

    if include_judge and judge_mcp_discovery_enabled():
        judge_url = (
            os.getenv("BENCHMARK_JUDGE_MCP_URL", "").strip()
            or os.getenv("JUDGE_MCP_URL", "").strip()
            or "http://judge:9009/mcp"
        )
        judge_url = _normalize_mcp_url(judge_url)
        if judge_url and "judge" not in config:
            config["judge"] = {
                "url": judge_url,
                "transport": "http",
            }

    return _filter_server_config_for_benchmark(config)


async def load_mcp_tools_from_env(*, include_judge: bool = True) -> list:
    """Load tools from all MCP servers configured via environment variables.

    Returns:
        A list of LangChain-compatible tool objects.
        Returns an empty list if no MCP servers are configured or on error.
    """
    config = _parse_server_config(include_judge=include_judge)
    if not config:
        logger.info("No MCP servers configured. Using built-in tools only.")
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.error(
            "langchain-mcp-adapters not installed. "
            "Run: uv add langchain-mcp-adapters"
        )
        return []

    loaded_tools: list[Any] = []
    seen_names: set[str] = set()
    for server_name, server_config in config.items():
        try:
            logger.info("Connecting to MCP server '%s'", server_name)
            client = MultiServerMCPClient({server_name: server_config})
            server_tools = await client.get_tools()
            logger.info(
                "Loaded %s MCP tools from '%s': %s",
                len(server_tools),
                server_name,
                [tool.name for tool in server_tools],
            )
            for tool in server_tools:
                tool_name = str(getattr(tool, "name", "") or "")
                if not tool_name or tool_name in seen_names:
                    continue
                seen_names.add(tool_name)
                loaded_tools.append(tool)
        except Exception as exc:
            logger.error("Failed to load MCP tools from server '%s': %s", server_name, exc)

    if not loaded_tools:
        logger.warning("No MCP tools were loaded successfully from configured servers: %s", list(config.keys()))
        return []

    return loaded_tools
