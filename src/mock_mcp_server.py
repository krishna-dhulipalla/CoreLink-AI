"""
Local mock MCP server for end-to-end testing.

Default usage (HTTP):
    uv run src/mock_mcp_server.py --host 127.0.0.1 --port 3001

Stdio usage:
    uv run src/mock_mcp_server.py --transport stdio
"""

import argparse

from mcp.server.fastmcp import FastMCP


def build_server(host: str, port: int) -> FastMCP:
    server = FastMCP(
        name="CoreLink AI Mock MCP",
        instructions=(
            "A local test MCP server that exposes simple deterministic tools "
            "for verifying tool discovery and invocation."
        ),
        host=host,
        port=port,
        streamable_http_path="/mcp",
    )

    @server.tool(
        name="echo_magic",
        description=(
            "Return a deterministic confirmation string. Use this when the "
            "user explicitly asks for the external mock MCP tool."
        ),
    )
    def echo_magic(text: str) -> str:
        return f"MOCK_MCP_OK::{text}"

    @server.tool(
        name="sum_magic",
        description=(
            "Add two integers and return a tagged result string. Useful for "
            "verifying the model selected an external MCP tool."
        ),
    )
    def sum_magic(a: int, b: int) -> str:
        return f"MOCK_SUM::{a + b}"

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local mock MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="streamable-http",
        help="Transport to expose for the mock MCP server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for streamable HTTP mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3001,
        help="Port for streamable HTTP mode.",
    )
    args = parser.parse_args()

    server = build_server(args.host, args.port)
    server.run(args.transport)


if __name__ == "__main__":
    main()
