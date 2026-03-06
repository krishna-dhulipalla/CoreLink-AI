"""
Step-Level Verifier Node
========================
Evaluates the Executor's latest step (tool call or final answer) and emits a structured
verdict: PASS, REVISE, or BACKTRACK. Reverts state via checkpoint_stack on BACKTRACK.
"""

import json
import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.cost import CostTracker
from agent.prompts import VERIFIER_PROMPT, MODEL_NAME, VerdictDecision
from context_manager import count_tokens

logger = logging.getLogger(__name__)

# System message to inject on backtrack
BACKTRACK_WARNING = (
    "SYSTEM WARNING: Your previous approach was evaluated as fundamentally flawed or "
    "a hallucination (BACKTRACK verdict). The state has been reverted to this verified "
    "checkpoint. Do NOT carefully repeat the same mistake. Try a completely different "
    "strategy or tool."
)

def _get_last_checkpoint_messages(stack: list[dict]) -> list:
    """Helper to retrieve the messages from the last checkpoint."""
    if not stack:
        return []
    return stack[-1].get("messages", [])


def verifier(state: AgentState) -> dict:
    """
    Evaluates the most recent action and emits a verdict.
    If PASS: pushes current state to checkpoint_stack.
    If REVISE: appends a SystemMessage pointing out the error.
    If BACKTRACK: reverts `messages` to the last checkpoint and appends a warning.
    """
    selected_layers = state.get("selected_layers", [])
    if "verifier_check" not in selected_layers:
        # Skip verification if not explicitly activated in the policy
        return {}
        
    tracker: CostTracker = state.get("cost_tracker")
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0).with_structured_output(VerdictDecision)
    
    # Exclude system/internal warnings from the verifier's context to keep it focused
    # on the actual task flow
    history = [m for m in state["messages"] if not getattr(m, "additional_kwargs", {}).get("is_warning", False)]
    
    system_msg = SystemMessage(content=VERIFIER_PROMPT)
    messages = [system_msg] + history
    
    t0 = time.monotonic()
    try:
        verdict: VerdictDecision = llm.invoke(messages)
        success = True
    except Exception as e:
        logger.warning(f"Verifier LLM failed: {e}. Defaulting to PASS.")
        verdict = VerdictDecision(verdict="PASS", reasoning=f"Fallback due to error: {e}")
        success = False
        
    latency = (time.monotonic() - t0) * 1000
    
    if tracker:
        tracker.record(
            operator="verifier_check",
            tokens_in=count_tokens(messages),
            tokens_out=count_tokens([AIMessage(content=verdict.model_dump_json())]),
            latency_ms=latency,
            success=success
        )
        
    logger.info(f"[Verifier] Verdict: {verdict.verdict} | Reason: {verdict.reasoning[:120]}...")
    
    stack = list(state.get("checkpoint_stack", []))
    
    if verdict.verdict == "PASS":
        # Save a serialized form of messages to the stack
        from langchain_core.messages import messages_to_dict
        current_msgs_dump = messages_to_dict(state["messages"])
        stack.append({"messages": current_msgs_dump})
        return {"checkpoint_stack": stack}
        
    elif verdict.verdict == "REVISE":
        # Inject the critique directly to the executor to fix it
        warning_msg = SystemMessage(
            content=f"VERIFIER REVISION REQUIRED:\n{verdict.reasoning}",
            additional_kwargs={"is_warning": True}
        )
        return {"messages": [warning_msg]}
        
    elif verdict.verdict == "BACKTRACK":
        if not stack:
            # If no checkpoint exists, we can't really backtrack. Treat like REVISE.
            warning_msg = SystemMessage(
                content=f"VERIFIER BACKTRACK (No checkpoints to revert to):\n{verdict.reasoning}",
                additional_kwargs={"is_warning": True}
            )
            return {"messages": [warning_msg]}
            
        from langchain_core.messages import messages_from_dict
        
        # Pop the current (failed) trajectory up to the last checkpoint
        checkpoint_dict = stack.pop() # Wait, the last item in stack is the PREVIOUS pass?
        # Actually, if the step before this was a PASS, it's the top of the stack.
        # But wait, we want to REVERT to that top element, not pop it, or pop it if we consider it bad?
        # No, the top of the stack is the LAST VERIFIED GOOD state. We want to revert TO it.
        # We don't pop it unless we want to roll back even further. Let's just keep it on top.
        stack.append(checkpoint_dict) # put it back
        
        last_good_messages = messages_from_dict(checkpoint_dict["messages"])
        
        # We need to use the ReplaceMessages wrapper, but since we are returning a dict,
        # we actually just return the replacing payload. But our reducer handles `ReplaceMessages` type.
        from agent.state import ReplaceMessages
        
        warning_msg = SystemMessage(
            content=f"{BACKTRACK_WARNING}\n\nReason for backtrack: {verdict.reasoning}",
            additional_kwargs={"is_warning": True}
        )
        
        new_messages = last_good_messages + [warning_msg]
        
        return {
            "messages": ReplaceMessages(new_messages),
            "checkpoint_stack": stack[:-1] # Wait, if we keep the stack, actually let's just leave the stack as is.
            # Next time it passes, it will push on top of it.
        }
        
    return {}

def verify_routing(state: AgentState) -> str:
    """
    Conditional edge coming out of the verifier.
    If PASS:
        Has the agent produced a final answer (no tool calls)? If so, go to format_normalizer.
        If it was a tool call, route to reasoner to continue.
    If REVISE:
        Route back to reasoner.
    If BACKTRACK:
        Route back to reasoner (with reverted state).
    """
    # Look at the last message.
    # If there's a recent warning (from REVISE or BACKTRACK), we route to reasoner.
    last_msg = state["messages"][-1]
    if getattr(last_msg, "additional_kwargs", {}).get("is_warning", False):
        return "reasoner"
        
    # It must be a PASS. Look at the message before the (potential) verifier passes.
    # Actually, the verifier doesn't append a message on PASS.
    # So `last_msg` is the Executor's AIMessage or the ToolMessage.
    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
        # Final answer achieved and verified.
        return "format_normalizer"
    
    # Either a tool call just got verified, or tool results just arrived and got verified? 
    # Wait, when does verifier run?
    # reasoner -> should_use_tools -> (tool_executor -> context_window -> verifier -> verify_routing -> reasoner).
    # What if reasoner -> should_use_tools -> verifier -> verify_routing -> format_normalizer?
    return "reasoner"

