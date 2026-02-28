"""
Purple Agent: LangGraph ReAct Reasoning Engine
================================================
Implements a Plan-Act-Learn loop using LangGraph's StateGraph.
The agent uses a ReAct-style cycle:
    Reasoner → Tool Executor → Context Window → (loop or END).

Architecture Reference: docs/DESIGN.md (Brain-to-Arm paradigm)
"""

import logging
import os
import operator
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from context_manager import (
    summarize_and_window,
    truncate_tool_output,
)

logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Agent State
# ---------------------------------------------------------------------------

class ReplaceMessages(list):
    """Sentinel wrapper: tells the reducer to *replace* instead of append."""
    pass


def _messages_reducer(
    existing: list[BaseMessage], update: list[BaseMessage] | ReplaceMessages
) -> list[BaseMessage]:
    """Custom reducer that supports both append and full replace.

    - Normal node output (plain list) → appended to existing.
    - ReplaceMessages wrapper → existing is replaced entirely.
    """
    if isinstance(update, ReplaceMessages):
        return list(update)
    return existing + update


class AgentState(TypedDict):
    """Typed state for the LangGraph reasoning engine.

    - messages: The conversation history (LangChain message format).
                Uses a custom reducer that supports both append and replace.
    """
    messages: Annotated[list[BaseMessage], _messages_reducer]


# ---------------------------------------------------------------------------
# 2. Built-in Tools (starter set – MCP tools added in later sprints)
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Use standard Python math syntax, e.g. '2 + 2' or '(3 ** 2) + 1'.
    """
    try:
        # Only allow safe math expressions
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return f"Error: Expression contains disallowed characters. Only digits and +-*/.() are permitted."
        result = eval(expression)  # noqa: S307 – restricted character set
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


BUILTIN_TOOLS = [calculator, get_current_time]


# ---------------------------------------------------------------------------
# 3. System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are CoreLink AI – a generalist reasoning agent competing in the AgentX-AgentBeats Competition.

Your operating loop is Plan → Act → Learn:
1. **Plan**: Break the task into clear steps. Identify which tools you need.
2. **Act**: Execute one step at a time using the available tools.
3. **Learn**: Review the tool output. If the result is wrong or incomplete, revise your plan and try again.

Rules:
- Always think step-by-step before acting.
- Use tools when they can provide a precise answer (math, time, external data).
- If no tool is needed, answer directly from your knowledge.
- Be concise and accurate in your final response.
"""


# ---------------------------------------------------------------------------
# 4. Graph Nodes
# ---------------------------------------------------------------------------

def _build_model(tools: list):
    """Instantiate the LLM with tool bindings."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return llm.bind_tools(tools)


def _make_reasoner(tools: list):
    """Factory: returns a reasoner node that uses the given tool list."""

    def reasoner(state: AgentState) -> dict:
        """The 'Brain' node – calls the LLM with the current conversation.

        If the LLM decides to use a tool, the response will contain tool_calls,
        which the conditional edge will route to the tool_executor node.
        """
        model = _build_model(tools)
        # Prepend system prompt if this is the first turn
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = model.invoke(messages)
        return {"messages": [response]}

    return reasoner


def should_use_tools(state: AgentState) -> str:
    """Conditional edge: route to tool_executor if the last AI message has tool_calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_executor"
    return END


# ---------------------------------------------------------------------------
# 5. Context Window Node (Observation Masking)
# ---------------------------------------------------------------------------

def _tool_executor_with_truncation(tool_node: ToolNode):
    """Wrap the ToolNode to truncate verbose tool outputs."""

    async def wrapper(state: AgentState) -> dict:
        result = await tool_node.ainvoke(state)
        # result is a dict with "messages" key
        messages = result.get("messages", [])
        truncated_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                original_len = len(msg.content)
                truncated_content = truncate_tool_output(msg.content)
                if len(truncated_content) < original_len:
                    logger.info(
                        f"Truncated tool '{msg.name}' output: "
                        f"{original_len} → {len(truncated_content)} chars"
                    )
                msg = ToolMessage(
                    content=truncated_content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            truncated_messages.append(msg)
        return {"messages": truncated_messages}

    return wrapper


def context_window(state: AgentState) -> dict:
    """Graph node: apply Summarize-and-Forget windowing to message history.

    Fires after tool_executor and before the next reasoner invocation.
    If the conversation is within the token budget, this is a no-op.
    """
    messages = state["messages"]
    compressed = summarize_and_window(messages)

    if len(compressed) < len(messages):
        # Wrap in ReplaceMessages so the custom reducer replaces the list
        return {"messages": ReplaceMessages(compressed)}

    # No compression needed — return empty to keep state as-is
    return {"messages": []}


# ---------------------------------------------------------------------------
# 6. Build the Compiled Graph
# ---------------------------------------------------------------------------


def build_agent_graph(external_tools: list | None = None):
    """Construct and compile the LangGraph StateGraph.

    Args:
        external_tools: Optional list of LangChain tools loaded from MCP
                        servers. These are merged with built-in tools.

    Graph topology:
        reasoner ──(has tool_calls?)──▶ tool_executor ──▶ context_window ──▶ reasoner
                 └─(no tool_calls)───▶ END
    """
    all_tools = BUILTIN_TOOLS + (external_tools or [])
    raw_tool_node = ToolNode(all_tools)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reasoner", _make_reasoner(all_tools))
    graph.add_node("tool_executor", _tool_executor_with_truncation(raw_tool_node))
    graph.add_node("context_window", context_window)

    # Set entry point
    graph.set_entry_point("reasoner")

    # Conditional edge from reasoner
    graph.add_conditional_edges("reasoner", should_use_tools)

    # After tool execution → context window → back to reasoner
    graph.add_edge("tool_executor", "context_window")
    graph.add_edge("context_window", "reasoner")

    return graph.compile()


# ---------------------------------------------------------------------------
# 6. Convenience Runner (used by executor.py)
# ---------------------------------------------------------------------------

async def run_agent(graph, input_text: str) -> tuple[str, list[dict]]:
    """Run the compiled graph with a user message and return (final_answer, steps).

    Returns:
        final_answer: The text content of the last AIMessage.
        steps: A list of dicts describing each node that executed,
               useful for streaming status updates to the A2A EventQueue.
    """
    initial_state = {"messages": [HumanMessage(content=input_text)]}
    steps = []

    # Use astream to capture node-by-node progression
    final_state = None
    async for event in graph.astream(initial_state):
        for node_name, node_output in event.items():
            step_info = {"node": node_name}
            if node_name == "reasoner":
                last_msg = node_output["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    tool_names = [tc["name"] for tc in last_msg.tool_calls]
                    step_info["action"] = f"Calling tools: {', '.join(tool_names)}"
                else:
                    step_info["action"] = "Generating final answer"
            elif node_name == "tool_executor":
                step_info["action"] = "Executing tools"
            steps.append(step_info)
        final_state = event

    # Extract final answer
    if final_state:
        # The last node's output contains the final messages
        for node_output in final_state.values():
            messages = node_output.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    return msg.content, steps

    return "I was unable to generate a response.", steps
