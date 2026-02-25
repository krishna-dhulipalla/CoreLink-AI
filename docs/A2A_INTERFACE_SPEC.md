# A2A Interface Specification

## Overview

This document details how the LangGraph reasoning engine interfaces with the `RDI-Foundation/agent-template` A2A protocol.

## 1. The `LangGraphAgentExecutor`

The core integration requires extending `a2a.server.agent_execution.AgentExecutor` to handle the A2A `RequestContext` and connect it to LangGraph.

Instead of the default `Executor` that creates a basic `Agent`, we implement a `LangGraphAgentExecutor`:

```python
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_task, new_agent_text_message

class LangGraphAgentExecutor(AgentExecutor):
    def __init__(self, compiled_graph):
        self.graph = compiled_graph
        self.contexts = {} # session memory
```

## 2. Handling `RequestContext` & State Mapping

When `execute(context: RequestContext, event_queue: EventQueue)` is invoked:

1. Extract `context.message` (the user input or Green Agent task).
2. Retrieve or initialize the `Task` and `TaskUpdater`.
3. Load the session state for `context.current_task.context_id` from `self.contexts` (or sqlite checkpointer for LangGraph).

The A2A input is appended to the LangGraph state (e.g., adding to the `messages` list key).

## 3. Streaming Updates to `EventQueue`

Because LangGraph processes in steps (nodes), we can stream updates back to the A2A server by listening to the LangGraph `astream_events` or `astream` API.

**Streaming Loop**:

```python
async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
    task = context.current_task or new_task(context.message)
    updater = TaskUpdater(event_queue, task.id, task.context_id)
    await updater.start_work()

    # Run LangGraph streaming
    async for event in self.graph.astream(
        {"messages": [("user", get_message_text(context.message))]},
        config={"configurable": {"thread_id": task.context_id}}
    ):
        # Example: Log node progression
        for node_name, node_state in event.items():
            if node_name == "tool_execution":
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Using tool: {node_state['tool_name']}")
                )
            elif node_name == "planner":
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message("Drafting plan...")
                )

    # On graph completion
    final_output = get_final_output_from_state(event)
    await updater.add_artifact(parts=[Part(root=TextPart(text=final_output))], name="Result")
    await updater.complete()
```

## 4. MCP Client Integration

The Green Agent's tools are accessed via an MCP client. This client is instantiated per-session or globally and passed into the LangGraph state.

- The `tool_execution` node in LangGraph intercepts tool call requests.
- It translates standard LangChain/LangGraph tool calls into MCP RPC requests.
- The response is parsed and appended to the graph state as a `ToolMessage`, masked/windowed if it exceeds truncation limits.
