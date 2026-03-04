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
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from context_manager import (
    summarize_and_window,
    truncate_tool_output,
)
from tools import CALCULATOR_TOOL, SEARCH_TOOL

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
    - reflection_count: Number of reflection-revision cycles completed.
    - tool_fail_count: Consecutive tool failures. Triggers forced fallback at threshold.
    - last_tool_signature: Hash of (tool_name + args) to detect duplicate calls.
    """
    messages: Annotated[list[BaseMessage], _messages_reducer]
    reflection_count: int
    tool_fail_count: int
    last_tool_signature: str


# ---------------------------------------------------------------------------
# 2. Built-in Tools (starter set – MCP tools added in later sprints)
# ---------------------------------------------------------------------------

@tool
def get_current_time() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _get_agent_tools() -> list[Any]:
    """Return the set of tools available to the core reasoning agent."""
    return [CALCULATOR_TOOL, SEARCH_TOOL, get_current_time]


# Reflective feedback loop configuration
MAX_REFLECTIONS = int(os.getenv("MAX_REFLECTIONS", "2"))


# ---------------------------------------------------------------------------
# 3. System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a general-purpose reasoning engine designed to solve complex multi-step tasks.
You operate in a Plan -> Act -> Learn loop.

Your core workflow:
1. **Plan**: Analyze the user's request. Determine what steps are needed and what tools to use.
2. **Act**: Execute the plan using the tools available to you.
3. **Learn**: If the tool output answers the question, formulate your final answer. If not, try a different approach.

Tool Usage Rules:
- You have access to a set of domain-specific tools provided by the environment.
- Read their descriptions carefully to understand their purpose, expected inputs, and outputs.
- Choose the most appropriate tool for the task based strictly on its description.
- For simple arithmetic, use the `calculator` tool with a SINGLE expression.
- For real-time facts or external data, use `internet_search`.
- If a tool returns an error, read the error message carefully. Do NOT call the same tool with the same arguments again.
- After a tool error, either fix your input or use a different tool.

Answer Composition Rule:
- When a tool output begins with "STRUCTURED_RESULTS:", copy that STRUCTURED_RESULTS line VERBATIM at the top of your final answer — do NOT rephrase, round, or omit any fields from it. Then add your explanation below.
"""

REFLECTION_PROMPT = """You are a quality reviewer. Examine the assistant's draft answer below and check for:
1. **Completeness** – Does it fully address the original question?
2. **Correctness** – Are all facts, calculations, and logic correct?
3. **Clarity** – Is the answer clear and well-structured?

Respond with EXACTLY one line in one of these two formats:
PASS: <brief justification why the answer is good>
REVISE: <specific issue that must be fixed>

Do NOT rewrite the answer. Only provide your verdict."""


# ---------------------------------------------------------------------------
# 4. Prompt / Reflection Helpers
# ---------------------------------------------------------------------------


def _with_system_prompt(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Ensure the base system prompt is always the first message.

    A windowing summary may also be represented as a ``SystemMessage``. That
    should not suppress the agent's real operating instructions.
    """
    if (
        messages
        and isinstance(messages[0], SystemMessage)
        and messages[0].content == SYSTEM_PROMPT
    ):
        return messages
    return [SystemMessage(content=SYSTEM_PROMPT)] + messages


def _is_reflection_message(msg: BaseMessage) -> bool:
    return (
        isinstance(msg, AIMessage)
        and bool(msg.content)
        and msg.content.startswith("[Reflection]")
    )


def _build_reflection_context(
    messages: list[BaseMessage],
    keep_last: int = 6,
) -> list[BaseMessage]:
    """Build a protocol-safe context slice for the reflection LLM call.

    We deliberately exclude:
    - ``AIMessage`` entries that contain ``tool_calls`` (they require matching
      ``ToolMessage`` entries in OpenAI chat-completions)
    - ``ToolMessage`` entries (not needed for the reflection verdict)
    - internal ``[Reflection]`` messages from previous passes
    """
    filtered: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            filtered.append(msg)
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                continue
            if _is_reflection_message(msg):
                continue
            filtered.append(msg)
    return filtered[-keep_last:]


# ---------------------------------------------------------------------------
# 5. Graph Nodes
# ---------------------------------------------------------------------------

def _build_model(tools: list):
    """Instantiate the LLM with tool bindings."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=1000,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return llm.bind_tools(tools)


# Step counter for logging (reset per run_agent call)
_step_counter = 0


def _make_reasoner(tools: list):
    """Factory: returns a reasoner node that uses the given tool list."""

    def reasoner(state: AgentState) -> dict:
        """The 'Brain' node – calls the LLM with the current conversation."""
        global _step_counter
        _step_counter += 1
        model = _build_model(tools)
        messages = _with_system_prompt(state["messages"])
        response = model.invoke(messages)

        # ── Step logging ──
        if isinstance(response, AIMessage) and response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info(f"[Step {_step_counter}] reasoner → tool_call: {', '.join(tool_names)}")
        else:
            preview = (response.content or "")[:100]
            logger.info(f"[Step {_step_counter}] reasoner → final answer: {preview}...")
        return {"messages": [response]}

    return reasoner


MAX_TOOL_FAILURES = 2  # consecutive failures before forcing fallback


def should_use_tools(state: AgentState) -> str:
    """Conditional edge: route to tool_executor or reflector.

    Before allowing another tool call, check if the agent is stuck in a
    repeated-failure loop. If tool_fail_count >= MAX_TOOL_FAILURES, force
    the agent to the reflector so it must produce a final answer.
    """
    # Failure gate: prevent repeated identical failed tool calls
    if state.get("tool_fail_count", 0) >= MAX_TOOL_FAILURES:
        logger.warning(
            f"[FailureGate] Tool failure limit ({MAX_TOOL_FAILURES}) reached. "
            "Forcing agent to produce final answer."
        )
        return "reflector"

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_executor"
    return "reflector"


# ---------------------------------------------------------------------------
# 6. Context Window Node (Observation Masking)
# ---------------------------------------------------------------------------

def _make_tool_signature(state: AgentState) -> str:
    """Build a string signature from the last AIMessage's tool calls (name + args).
    Used to detect duplicate identical tool calls.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            parts = []
            for tc in msg.tool_calls:
                args_str = str(sorted(tc.get("args", {}).items()))
                parts.append(f"{tc['name']}:{args_str}")
            return "|".join(parts)
    return ""


def _tool_executor_with_truncation(tool_node: ToolNode):
    """Wrap the ToolNode to truncate verbose tool outputs and track failures."""

    async def wrapper(state: AgentState) -> dict:
        global _step_counter
        _step_counter += 1
        result = await tool_node.ainvoke(state)
        messages = result.get("messages", [])
        truncated_messages = []

        # Compute signature of the call we just made (before results)
        call_signature = _make_tool_signature(state)
        previous_signature = state.get("last_tool_signature", "")

        any_error = False
        is_duplicate = (call_signature == previous_signature and call_signature != "")

        for msg in messages:
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                original_len = len(msg.content)
                truncated_content = truncate_tool_output(msg.content)
                if len(truncated_content) < original_len:
                    logger.info(
                        f"Truncated tool '{msg.name}' output: "
                        f"{original_len} → {len(truncated_content)} chars"
                    )
                # Detect error output from tool
                if truncated_content.startswith("Error"):
                    any_error = True
                    logger.warning(
                        f"[Step {_step_counter}] tool_executor → {msg.name} ERROR: "
                        f"{truncated_content[:120]}"
                    )
                    # Augment the error message to guide the model
                    if is_duplicate:
                        truncated_content += (
                            "\n\n[SYSTEM NOTE: You have called this tool with the same arguments before and it failed. "
                            "Do NOT repeat this call. Use a different tool or answer directly.]"
                        )
                else:
                    preview = truncated_content[:120].replace('\n', ' ')
                    logger.info(f"[Step {_step_counter}] tool_executor → {msg.name}: {preview}...")
                msg = ToolMessage(
                    content=truncated_content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            truncated_messages.append(msg)

        # Update failure tracking state
        current_fail_count = state.get("tool_fail_count", 0)
        new_fail_count = (current_fail_count + 1) if (any_error or is_duplicate) else 0

        return {
            "messages": truncated_messages,
            "tool_fail_count": new_fail_count,
            "last_tool_signature": call_signature,
        }

    return wrapper


def context_window(state: AgentState) -> dict:
    """Graph node: apply Summarize-and-Forget windowing to message history."""
    global _step_counter
    _step_counter += 1
    messages = state["messages"]
    compressed = summarize_and_window(messages)

    if len(compressed) < len(messages):
        logger.info(f"[Step {_step_counter}] context_window → compressed {len(messages)} → {len(compressed)} msgs")
        return {"messages": ReplaceMessages(compressed)}

    logger.info(f"[Step {_step_counter}] context_window → no compression needed")
    return {"messages": []}


# ---------------------------------------------------------------------------
# 7. Reflective Feedback Node
# ---------------------------------------------------------------------------

def reflector(state: AgentState) -> dict:
    """Graph node: critique the draft answer before submission.

    Uses a separate LLM call (no tools) with REFLECTION_PROMPT to evaluate
    the agent's last answer. Appends the critique to the message history.
    """
    messages = state["messages"]
    count = state.get("reflection_count", 0)

    # Skip reflection if we've hit the retry limit
    if count >= MAX_REFLECTIONS:
        logger.info(
            f"Reflection limit reached ({MAX_REFLECTIONS}). "
            "Submitting answer as-is."
        )
        return {"reflection_count": count}

    # Build reflection input: system prompt + a protocol-safe conversation slice.
    # Do not forward tool-call messages here without their matching ToolMessages.
    reflection_messages = [SystemMessage(content=REFLECTION_PROMPT)]
    reflection_messages.extend(_build_reflection_context(messages))

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=500,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    verdict = llm.invoke(reflection_messages)
    verdict_text = verdict.content.strip() if verdict.content else "PASS: no verdict"

    logger.info(f"Reflection #{count + 1}: {verdict_text}")

    return {
        "messages": [AIMessage(content=f"[Reflection]: {verdict_text}")],
        "reflection_count": count + 1,
    }


def should_revise(state: AgentState) -> str:
    """Conditional edge after reflector: REVISE loops back, PASS goes to END."""
    count = state.get("reflection_count", 0)

    # If we hit the limit, go to END regardless
    if count >= MAX_REFLECTIONS:
        return END

    # Check the last message for the reflection verdict
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.content:
        text = last_msg.content.upper()
        if "REVISE:" in text:
            logger.info("Reflector requested revision → looping back to reasoner")
            return "reasoner"

    return END


# ---------------------------------------------------------------------------
# 8. Build the Compiled Graph
# ---------------------------------------------------------------------------


def build_agent_graph(external_tools: list | None = None):
    """Construct and compile the LangGraph StateGraph.

    Args:
        external_tools: Optional list of LangChain tools loaded from MCP
                        servers. These are merged with built-in tools.

    Graph topology:
        reasoner ──(has tool_calls?)──▶ tool_executor ──▶ context_window ──▶ reasoner
                 └─(no tool_calls)───▶ reflector ──(PASS)──▶ END
                                                    └─(REVISE)─▶ reasoner
    """
    all_tools = BUILTIN_TOOLS + (external_tools or [])
    raw_tool_node = ToolNode(all_tools)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("reasoner", _make_reasoner(all_tools))
    graph.add_node("tool_executor", _tool_executor_with_truncation(raw_tool_node))
    graph.add_node("context_window", context_window)
    graph.add_node("reflector", reflector)

    # Set entry point
    graph.set_entry_point("reasoner")

    # Conditional edge from reasoner
    graph.add_conditional_edges("reasoner", should_use_tools)

    # After tool execution → context window → back to reasoner
    graph.add_edge("tool_executor", "context_window")
    graph.add_edge("context_window", "reasoner")

    # Reflector decision: PASS → END, REVISE → reasoner
    graph.add_conditional_edges("reflector", should_revise)

    return graph.compile()


# ---------------------------------------------------------------------------
# 9. Convenience Runner (used by executor.py)
# ---------------------------------------------------------------------------

async def run_agent(
    graph,
    input_text: str,
    history: list[BaseMessage] | None = None,
) -> tuple[str, list[dict], list[BaseMessage]]:
    """Run the compiled graph and return (final_answer, steps, updated_history).

    Args:
        graph: The compiled LangGraph.
        input_text: The new user message.
        history: Optional prior conversation messages for multi-turn support.

    Returns:
        final_answer: The text content of the last AIMessage (excluding reflections).
        steps: A list of dicts describing each node that executed.
        updated_history: The cleaned message list after execution, for persistence.
    """
    messages = list(history) if history else []
    messages.append(HumanMessage(content=input_text))

    # Reset step counter for this run
    global _step_counter
    _step_counter = 0

    # ── Fix #2: Front-gate pruning ────────────────────────────────────
    # Apply context windowing BEFORE graph entry so the first reasoner
    # call never exceeds the token budget (context_window node only fires
    # after tool_executor, not before the first reasoner call).
    messages = summarize_and_window(messages)

    initial_state = {
        "messages": messages,
        "reflection_count": 0,
        "tool_fail_count": 0,
        "last_tool_signature": "",
    }

    # ── Fix #1: Use ainvoke for correct post-windowing state ──────────
    # ainvoke returns the final accumulated state after the graph completes.
    # This state has been through the custom _messages_reducer, so it
    # correctly reflects any ReplaceMessages compressions from context_window.
    #
    # ── Safety: recursion_limit prevents runaway tool loops ────────────
    # LangGraph counts each node invocation as one step. With limit=25,
    # the agent can make roughly 7-8 tool calls before being stopped.
    try:
        final_state = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": 25},
        )
    except GraphRecursionError as e:
        # Recursion limit hit — extract the best partial answer from input state
        logger.warning(f"[Safety] Recursion limit hit: {e}")
        # Try to find the last non-reflection AI response so far
        partial_answer = None
        for msg in reversed(initial_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not _is_reflection_message(msg) and not msg.tool_calls:
                partial_answer = msg.content
                break
        return (
            partial_answer or "I was unable to complete this task within the step limit.",
            [{"node": "safety", "action": "Recursion limit reached"}],
            initial_state["messages"],
        )
    # All other exceptions propagate loudly — do not mask real bugs

    # Build step list from the final state (lightweight summary)
    steps = []
    all_messages = final_state.get("messages", [])
    for msg in all_messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_names = [tc["name"] for tc in msg.tool_calls]
                steps.append({
                    "node": "reasoner",
                    "action": f"Calling tools: {', '.join(tool_names)}",
                })
            elif _is_reflection_message(msg):
                steps.append({
                    "node": "reflector",
                    "action": f"Self-review: {msg.content[:80]}",
                })
            else:
                steps.append({
                    "node": "reasoner",
                    "action": "Generating draft answer",
                })
        elif isinstance(msg, ToolMessage):
            steps.append({
                "node": "tool_executor",
                "action": f"Tool result: {msg.name}",
            })

    # ── Fix #3: Strip reflection messages from persisted history ──────
    # [Reflection] messages are internal quality-control artifacts; they
    # should NOT leak into the conversation store for future turns.
    updated_history = [
        msg for msg in all_messages
            if not _is_reflection_message(msg)
        ]

    # Extract final answer: last AIMessage that is NOT a reflection
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and msg.content:
            if not _is_reflection_message(msg):
                return msg.content, steps, updated_history

    return "I was unable to generate a response.", steps, updated_history
