"""
Operator Abstraction: First-Class Runtime Units
=================================================
Defines operators as self-describing building blocks that the
controller can compose into layered execution plans.

Each operator maps to a LangGraph node and carries metadata about
its cost characteristics, dependencies, and failure behavior.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Operator:
    """A single unit of work the controller can select."""

    name: str               # unique identifier: "direct_answer", "react_reason", etc.
    description: str        # human-readable purpose
    cost_class: str         # "cheap" | "moderate" | "expensive"
    requires_tools: bool    # does this operator need tool bindings?
    failure_policy: str     # "retry_once" | "fallback_to_reflector" | "abort"
    node_name: str          # corresponding LangGraph node name


# ---------------------------------------------------------------------------
# Operator Registry
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY: dict[str, Operator] = {
    "direct_answer": Operator(
        name="direct_answer",
        description="Fast LLM response for simple factual/conversational queries. No tools.",
        cost_class="cheap",
        requires_tools=False,
        failure_policy="abort",
        node_name="direct_responder",
    ),
    "react_reason": Operator(
        name="react_reason",
        description="Full ReAct reasoning loop with tool access. For multi-step tasks.",
        cost_class="expensive",
        requires_tools=True,
        failure_policy="fallback_to_reflector",
        node_name="reasoner",
    ),
    "search_retrieve": Operator(
        name="search_retrieve",
        description="Internet search or document retrieval via MCP tools.",
        cost_class="moderate",
        requires_tools=True,
        failure_policy="retry_once",
        node_name="tool_executor",
    ),
    "calculator_exec": Operator(
        name="calculator_exec",
        description="Safe arithmetic evaluation via the calculator tool.",
        cost_class="cheap",
        requires_tools=True,
        failure_policy="retry_once",
        node_name="tool_executor",
    ),
    "reflection_review": Operator(
        name="reflection_review",
        description="Self-critique pass that evaluates draft answer quality.",
        cost_class="moderate",
        requires_tools=False,
        failure_policy="abort",
        node_name="reflector",
    ),
    "format_normalize": Operator(
        name="format_normalize",
        description="Strict JSON/XML formatting pass for benchmark compliance.",
        cost_class="cheap",
        requires_tools=False,
        failure_policy="abort",
        node_name="format_normalizer",
    ),
    "verifier_check": Operator(
        name="verifier_check",
        description="Strict step-level verification emitting PASS, REVISE, or BACKTRACK.",
        cost_class="moderate",
        requires_tools=False,
        failure_policy="abort",
        node_name="verifier",
    ),
}


def validate_layers(layers: list[str]) -> list[str]:
    """Validate operator names against the registry.

    Returns only valid operator names. If none are valid,
    returns the default heavy-research plan.
    """
    valid = [l for l in layers if l in OPERATOR_REGISTRY]
    if not valid:
        return ["react_reason", "reflection_review"]
    return valid


# Default execution plans for common routes
DEFAULT_PLANS = {
    "direct": ["direct_answer"],
    "heavy_research": ["react_reason", "reflection_review", "format_normalize"],
}
