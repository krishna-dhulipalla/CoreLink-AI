"""
A2A ↔ LangGraph Bridge (Executor)
==================================
Bridges the A2A protocol with the LangGraph reasoning engine.
Converts A2A RequestContext into LangGraph input, streams status
updates back through the A2A EventQueue.

Architecture Reference: docs/A2A_INTERFACE_SPEC.md
"""

import asyncio
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    get_message_text,
    new_agent_text_message,
    new_task,
)

from agent import build_agent_graph, run_agent
from mcp_client import load_mcp_tools_from_env

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    """A2A executor that delegates reasoning to the LangGraph agent."""

    def __init__(self):
        # Load MCP tools from a synchronous constructor. `asyncio.run()` is the
        # most reliable option here as long as we are not already inside a
        # running event loop.
        try:
            asyncio.get_running_loop()
            self._mcp_tools = []
            self._mcp_loaded = False
            logger.warning(
                "Skipping MCP tool initialization because an event loop is "
                "already running during Executor construction."
            )
        except RuntimeError:
            self._mcp_tools = asyncio.run(load_mcp_tools_from_env())
            self._mcp_loaded = True

        if self._mcp_tools:
            logger.info(
                f"MCP tools loaded: {[t.name for t in self._mcp_tools]}"
            )

        self.graph = build_agent_graph(external_tools=self._mcp_tools)

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # ── Validate request ──────────────────────────────────────────
        msg = context.message
        if not msg:
            raise ServerError(
                error=InvalidRequestError(message="Missing message in request")
            )

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        # ── Create or reuse task ──────────────────────────────────────
        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        # ── Run LangGraph agent ───────────────────────────────────────
        try:
            input_text = get_message_text(msg)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Planning approach..."),
            )

            final_answer, steps = await run_agent(self.graph, input_text)

            # Stream step-level status updates for transparency
            for step in steps:
                action = step.get("action", step["node"])
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(action),
                )

            # ── Publish final result ──────────────────────────────────
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=final_answer))],
                name="Result",
            )
            await updater.complete()

        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {e}",
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
