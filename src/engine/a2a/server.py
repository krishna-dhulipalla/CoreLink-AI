"""CoreLink AI A2A server."""

import argparse
import sys
from pathlib import Path

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR / "src"))

from engine.a2a.executor import Executor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CoreLink AI A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="finance-first-reasoning",
        name="Finance-First Reasoning",
        description=(
            "A modular reasoning engine that builds task-specific context, uses structured "
            "tools when needed, and returns normalized answers for finance, legal, and "
            "document-oriented tasks."
        ),
        tags=["reasoning", "finance", "tool-use", "engine"],
        examples=[
            "Calculate a finance metric from inline table data.",
            "Design an options strategy from volatility inputs.",
            "Explain acquisition structure tradeoffs under regulatory constraints.",
        ],
    )

    agent_card = AgentCard(
        name="CoreLink AI",
        description=(
            "A finance-first reasoning engine built on LangGraph and MCP. "
            "It plans each request, assembles structured evidence, executes safe tool-backed "
            "reasoning, reviews the final artifact, and applies answer normalization."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
