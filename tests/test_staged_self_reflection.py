from langchain_core.messages import AIMessage

from agent.nodes.reviewer import route_from_reviewer
from agent.nodes.self_reflection import ReflectionResult, self_reflection
from staged_test_utils import make_state


def test_route_from_reviewer_sends_eligible_finals_to_self_reflection_in_benchmark_mode(monkeypatch):
    monkeypatch.setenv("BENCHMARK_STATELESS", "1")
    state = make_state(
        "Need to move quickly here.",
        task_profile="legal_transactional",
        execution_template={
            "template_id": "legal_reasoning_only",
            "allowed_stages": ["SYNTHESIZE", "REVISE", "COMPLETE"],
        },
        solver_stage="COMPLETE",
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "self_reflection_attempts": 0},
    )

    assert route_from_reviewer(state) == "self_reflection"


def test_route_from_reviewer_sends_complex_qualitative_legal_final_to_self_reflection_without_env(monkeypatch):
    monkeypatch.delenv("BENCHMARK_STATELESS", raising=False)
    monkeypatch.delenv("ENABLE_FINAL_SELF_REFLECTION", raising=False)
    monkeypatch.delenv("DISABLE_FINAL_SELF_REFLECTION", raising=False)
    state = make_state(
        "Need to move quickly here.",
        task_profile="legal_transactional",
        execution_template={
            "template_id": "legal_reasoning_only",
            "allowed_stages": ["SYNTHESIZE", "REVISE", "COMPLETE"],
        },
        solver_stage="COMPLETE",
        workpad={
            "events": [],
            "stage_outputs": {},
            "tool_results": [],
            "self_reflection_attempts": 0,
            "task_complexity_tier": "complex_qualitative",
        },
    )

    assert route_from_reviewer(state) == "self_reflection"


def test_route_from_reviewer_skips_self_reflection_after_one_attempt(monkeypatch):
    monkeypatch.setenv("BENCHMARK_STATELESS", "1")
    state = make_state(
        "Need to move quickly here.",
        task_profile="legal_transactional",
        execution_template={
            "template_id": "legal_reasoning_only",
            "allowed_stages": ["SYNTHESIZE", "REVISE", "COMPLETE"],
        },
        solver_stage="COMPLETE",
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "self_reflection_attempts": 1},
    )

    assert route_from_reviewer(state) == "reflect"


def test_self_reflection_requests_one_targeted_final_revision(monkeypatch):
    monkeypatch.setenv("BENCHMARK_STATELESS", "1")
    monkeypatch.setattr(
        "agent.nodes.self_reflection.invoke_structured_output",
        lambda role, schema, messages, **kwargs: (
            ReflectionResult(
                score=0.61,
                complete=False,
                missing_dimensions=["liability allocation detail", "execution timing detail"],
                improve_prompt="Add concrete indemnity or escrow mechanics and sharper signing or closing timing tradeoffs.",
            ),
            "Qwen/Qwen3-32B-fast",
        ),
    )
    state = make_state(
        "Need to move quickly here.",
        task_profile="legal_transactional",
        execution_template={
            "template_id": "legal_reasoning_only",
            "allowed_stages": ["SYNTHESIZE", "REVISE", "COMPLETE"],
        },
        solver_stage="COMPLETE",
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "self_reflection_attempts": 0},
    )
    state["messages"].append(
        AIMessage(
            content=(
                "Recommendation: pursue an asset deal.\n"
                "Tax: likely cleaner than a stock deal.\n"
                "Liability: lower inherited exposure.\n"
                "Next steps: confirm diligence."
            )
        )
    )

    result = self_reflection(state)

    assert result["solver_stage"] == "REVISE"
    assert result["review_feedback"]["repair_target"] == "final"
    assert "indemnity" in result["review_feedback"]["reasoning"].lower()
    assert result["workpad"]["self_reflection_attempts"] == 1


def test_self_reflection_passes_strong_qualitative_final(monkeypatch):
    monkeypatch.setenv("BENCHMARK_STATELESS", "1")
    state = make_state(
        "Advise on acquisition structure.",
        task_profile="legal_transactional",
        execution_template={
            "template_id": "legal_reasoning_only",
            "allowed_stages": ["SYNTHESIZE", "REVISE", "COMPLETE"],
        },
        solver_stage="COMPLETE",
        workpad={"events": [], "stage_outputs": {}, "tool_results": [], "self_reflection_attempts": 0},
    )
    state["messages"].append(
        AIMessage(
            content=(
                "Recommendation: sign an asset acquisition with a narrow stock fallback only if assignments fail.\n"
                "Liability allocation: use specific indemnities, escrow holdback, caps and baskets, survival periods, "
                "and buyer-favorable reps and warranties backed by disclosure schedules.\n"
                "Execution timing: use a short signing-to-closing window with consent and condition-precedent tracking, "
                "plus interim covenants and a rapid diligence plan.\n"
                "Next steps: prioritize consent mapping, IP chain-of-title diligence, and escrow sizing."
            )
        )
    )

    result = self_reflection(state)

    assert result["solver_stage"] == "COMPLETE"
    assert result["workpad"]["self_reflection_attempts"] == 1
    assert result["reflection_feedback"]["complete"] is True
