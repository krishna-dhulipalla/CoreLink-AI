"""Request-scoped Judge MCP discovery and tool wrappers.

This bridge is intentionally benchmark-focused. Unlike generic MCP loading,
Judge tools are discovered per task session and validated against the
server-provided schema before each call.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model


_TOOL_TIMEOUT = float(os.getenv("JUDGE_MCP_TIMEOUT_SECONDS", "25"))
logger = logging.getLogger(__name__)


class JudgeMcpConnectionError(RuntimeError):
    """Raised when the benchmark Judge MCP endpoint cannot be reached."""
    pass


def _judge_enabled() -> bool:
    return os.getenv("ENABLE_JUDGE_MCP_DISCOVERY", "1").strip().lower() not in {"0", "false", "no", "off"}


def _strict_judge_mcp_discovery() -> bool:
    return os.getenv("STRICT_JUDGE_MCP_DISCOVERY", "0").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_judge_mcp_url(url: str) -> str:
    normalized = (url or "").strip().rstrip("/")
    if normalized.endswith("/tools"):
        normalized = normalized[: -len("/tools")]
    if not normalized.endswith("/mcp"):
        normalized = normalized + "/mcp"
    return normalized


def judge_mcp_base_url() -> str:
    raw = (
        os.getenv("BENCHMARK_JUDGE_MCP_URL", "").strip()
        or os.getenv("JUDGE_MCP_URL", "").strip()
        or "http://judge:9009/mcp"
    )
    return _normalize_judge_mcp_url(raw)


def validate_tool_call(tool_name: str, params: dict[str, Any], tools_list: list[dict[str, Any]]) -> tuple[bool, str]:
    """Verify a tool exists and required params are present.

    Mirrors Purple's benchmark bridge behavior closely so the judge schema is
    respected before any network round-trip happens.
    """
    if not tools_list:
        return True, ""

    tool_schema = next((t for t in tools_list if str(t.get("name", "")) == tool_name), None)
    if tool_schema is None:
        available = [str(t.get("name", "")) for t in tools_list[:10]]
        return False, f"Tool '{tool_name}' not in available tools. Available: {available}"

    input_schema = tool_schema.get("input_schema") or tool_schema.get("inputSchema") or {}
    required = list(input_schema.get("required", []) or [])
    missing = [name for name in required if name not in params or params.get(name) in (None, "")]
    if missing:
        return False, f"Tool '{tool_name}' missing required params: {missing}"

    return True, ""


async def discover_judge_tools(session_id: str = "") -> list[dict[str, Any]]:
    """Discover benchmark tools for the given task session."""
    if not _judge_enabled():
        return []

    url = f"{judge_mcp_base_url()}/tools"
    params = {"session_id": session_id} if session_id else None
    try:
        async with httpx.AsyncClient(timeout=_TOOL_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.HTTPError, OSError) as exc:
        message = _judge_discovery_failure_message(url, exc)
        if _strict_judge_mcp_discovery():
            raise JudgeMcpConnectionError(message) from exc
        logger.warning("%s Falling back to built-in/local tools.", message)
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        tools = payload.get("tools", payload.get("data", []))
        if isinstance(tools, list):
            return [item for item in tools if isinstance(item, dict)]
    return []


def _judge_discovery_failure_message(url: str, exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None and exc.response.status_code == 404:
        return (
            f"Judge MCP endpoint not available at {url} (404). "
            "The local lightweight benchmark runner may not expose MCP tools."
        )
    return (
        f"Judge MCP discovery failed at {url}: {exc}. "
        "If this agent is running outside the benchmark Docker network, "
        "set BENCHMARK_JUDGE_MCP_URL to a host-reachable Judge MCP URL or run the agent inside the benchmark network."
    )


async def call_judge_tool(
    tool_name: str,
    params: dict[str, Any],
    *,
    session_id: str,
    tools_list: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call a Judge MCP tool with pre-flight validation."""
    valid, error_msg = validate_tool_call(tool_name, params, tools_list or [])
    if not valid:
        return {"error": error_msg, "validation_failed": True}

    endpoint = judge_mcp_base_url()
    try:
        async with httpx.AsyncClient(timeout=_TOOL_TIMEOUT) as client:
            resp = await client.post(
                endpoint,
                json={
                    "tool": tool_name,
                    "params": params,
                    "session_id": session_id,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.HTTPError, OSError) as exc:
        raise JudgeMcpConnectionError(
            f"Judge MCP tool call failed at {endpoint} for tool '{tool_name}': {exc}. "
            "If this agent is running outside the benchmark Docker network, "
            "set BENCHMARK_JUDGE_MCP_URL to a host-reachable Judge MCP URL or run the agent inside the benchmark network."
        ) from exc
    return payload if isinstance(payload, dict) else {"result": payload}


def _sanitize_model_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value or "JudgeTool").strip("_") or "JudgeTool"


def _python_type_for_schema(prop_schema: dict[str, Any]) -> Any:
    schema_type = str(prop_schema.get("type", "") or "").lower()
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list[Any]
    if schema_type == "object":
        return dict[str, Any]
    return str


def _args_model_for_tool(tool_schema: dict[str, Any]) -> type[BaseModel]:
    input_schema = tool_schema.get("input_schema") or tool_schema.get("inputSchema") or {}
    properties = dict(input_schema.get("properties") or {})
    required = set(input_schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, prop_schema in properties.items():
        prop = dict(prop_schema or {})
        annotation = _python_type_for_schema(prop)
        description = str(prop.get("description", "") or "")
        if field_name in required:
            default = Field(..., description=description)
        else:
            default = Field(prop.get("default", None), description=description)
        fields[str(field_name)] = (annotation, default)

    model_name = _sanitize_model_name(str(tool_schema.get("name", "JudgeTool"))) + "Arguments"
    return create_model(model_name, **fields) if fields else create_model(model_name)


def _tool_description(tool_schema: dict[str, Any]) -> str:
    description = str(tool_schema.get("description", "") or "Judge MCP tool")
    input_schema = tool_schema.get("input_schema") or tool_schema.get("inputSchema") or {}
    required = list(input_schema.get("required", []) or [])
    if required:
        description += f" Required params: {', '.join(required)}."
    return description


def build_judge_tools(
    tools_list: list[dict[str, Any]],
    *,
    session_id: str,
) -> list[Any]:
    """Build LangChain-compatible tools for the discovered Judge schema."""
    wrapped_tools: list[Any] = []
    for tool_schema in tools_list:
        tool_name = str(tool_schema.get("name", "") or "").strip()
        if not tool_name:
            continue
        args_schema = _args_model_for_tool(tool_schema)
        description = _tool_description(tool_schema)

        async def _judge_tool_runner(
            _tool_name: str = tool_name,
            _session_id: str = session_id,
            _tools_list: list[dict[str, Any]] = tools_list,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return await call_judge_tool(
                _tool_name,
                kwargs,
                session_id=_session_id,
                tools_list=_tools_list,
            )

        wrapped_tools.append(
            StructuredTool.from_function(
                coroutine=_judge_tool_runner,
                name=tool_name,
                description=description,
                args_schema=args_schema,
            )
        )
    return wrapped_tools


async def load_judge_tools_for_session(session_id: str = "") -> tuple[list[Any], list[dict[str, Any]]]:
    """Discover Judge tools for a task session and return wrapped tool objects."""
    tools_list = await discover_judge_tools(session_id=session_id)
    return build_judge_tools(tools_list, session_id=session_id), tools_list
