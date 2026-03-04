"""
Built-in Tools: Calculator and Web Search
==========================================
Provides a minimal, always-available fallback tool set for the reasoning engine.
These complement (not replace) the domain-specific MCP server tools.

Tools:
  - CALCULATOR_TOOL : Safe AST-based arithmetic evaluator (no eval, no code injection).
  - SEARCH_TOOL     : Tavily web search (uses TAVILY_API_KEY from .env).
  - get_current_time: (defined in agent.py) UTC timestamp.

Why keep these alongside MCP tools?
  - MCP servers may be unavailable or slow to start (stdio spin-up latency).
  - Calculator and time are instant, dependency-free, and always correct.
  - Tavily search covers general internet queries that no domain MCP handles.
"""

import ast
import operator as _op
import os

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Safe Calculator (no eval, no exec — pure AST walk)
# ---------------------------------------------------------------------------

_SAFE_OPERATORS = {
    ast.Add:      _op.add,
    ast.Sub:      _op.sub,
    ast.Mult:     _op.mul,
    ast.Div:      _op.truediv,
    ast.Pow:      _op.pow,
    ast.USub:     _op.neg,
    ast.UAdd:     _op.pos,
    ast.Mod:      _op.mod,
    ast.FloorDiv: _op.floordiv,
}

_SAFE_FUNCS = {
    "abs": abs, "round": round, "min": min,
    "max": max, "int": int, "float": float,
}


def _safe_eval(node):
    """Recursively evaluate a safe arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"Unsupported function: {func_name}")
        return _SAFE_FUNCS[func_name](*[_safe_eval(a) for a in node.args])
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a single arithmetic expression safely.

    Supports: +, -, *, /, //, %, **, and abs(), round(), min(), max(), int(), float().
    For simple arithmetic that does not need internet data, always prefer this tool.

    Args:
        expression: A math expression string, e.g. '(3 + 5) * 2' or 'round(1/3, 4)'.
    Returns:
        The numeric result as a string, or an error message prefixed with 'Error:'.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


CALCULATOR_TOOL = calculator


# ---------------------------------------------------------------------------
# 2. Tavily Web Search
# ---------------------------------------------------------------------------

@tool
def internet_search(query: str) -> str:
    """Search the internet for real-time information using Tavily.

    Use this for current events, live market data, news, or any fact you don't know.
    Requires TAVILY_API_KEY to be set in the environment.

    Args:
        query: The search query string.
    Returns:
        A formatted list of search result snippets with titles, content, and URLs.
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return "Error: TAVILY_API_KEY is not set in the environment."

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=5)
        results = response.get("results", [])
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            title   = r.get("title", "")
            content = r.get("content", "")
            url     = r.get("url", "")
            lines.append(f"[{i}] {title}\n    {content}\n    URL: {url}")
        return "\n\n".join(lines)
    except ImportError:
        return (
            "Error: 'tavily-python' package is not installed. "
            "Run: uv add tavily-python"
        )
    except Exception as e:
        return f"Error performing Tavily search: {e}"


SEARCH_TOOL = internet_search
