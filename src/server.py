"""
CoreLink AI – A2A Server
=========================
Serves the Purple Agent over the A2A protocol.
"""

import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the CoreLink AI A2A agent.")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9009, help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url", type=str, help="URL to advertise in the agent card"
    )
    args = parser.parse_args()

    # ── Agent Skill ───────────────────────────────────────────────────
    skill = AgentSkill(
        id="generalist-reasoning",
        name="Generalist Reasoning",
        description=(
            "A benchmark-agnostic reasoning engine that uses a Plan-Act-Learn "
            "loop to solve tasks. Capable of multi-step reasoning, tool use, "
            "and self-correction through reflective feedback."
        ),
        tags=["reasoning", "tool-use", "planning", "generalist"],
        examples=[
            "What is the square root of 144?",
            "What time is it right now in UTC?",
            "Break down how to solve this problem step by step.",
        ],
    )

    # ── Agent Card ────────────────────────────────────────────────────
    agent_card = AgentCard(
        name="CoreLink AI",
        description=(
            "A generalist reasoning engine for the AgentX-AgentBeats Competition. "
            "Implements a Plan-Act-Learn architecture powered by LangGraph "
            "for multi-step reasoning, dynamic tool use, and self-evolving feedback."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    # ── Server ────────────────────────────────────────────────────────
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
