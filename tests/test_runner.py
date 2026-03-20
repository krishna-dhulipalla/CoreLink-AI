import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

import agent.runner as runner_module


class _StaticGraph:
    def __init__(self, final_messages):
        self.final_messages = final_messages
        self.initial_state = None

    async def astream(self, initial_state, config=None, stream_mode=None):
        self.initial_state = initial_state
        state = dict(initial_state)
        state["messages"] = list(self.final_messages)
        yield state


class _RecursingGraph:
    def __init__(self):
        self.initial_state = None

    async def astream(self, initial_state, config=None, stream_mode=None):
        self.initial_state = initial_state
        state = dict(initial_state)
        state["messages"] = list(initial_state["messages"]) + [AIMessage(content="0.9273")]
        yield state
        raise GraphRecursionError("step limit")


def test_run_agent_trace_stateless_mode_ignores_history_and_dedupes(monkeypatch):
    monkeypatch.setenv("BENCHMARK_STATELESS", "1")
    monkeypatch.setattr(runner_module, "_get_memory_store", lambda: None)
    monkeypatch.setattr(runner_module, "summarize_and_window", lambda messages: messages)

    graph = _StaticGraph([HumanMessage(content="What is ROE?"), AIMessage(content="0.12")])
    trace = asyncio.run(
        runner_module.run_agent_trace(
            graph,
            "What is ROE?",
            history=[HumanMessage(content="old task"), AIMessage(content="old answer")],
        )
    )

    assert [msg.content for msg in graph.initial_state["messages"]] == ["What is ROE?"]
    assert [msg.content for msg in trace["updated_history"]] == ["What is ROE?", "0.12"]


def test_run_agent_trace_recursion_preserves_partial_answer_and_history(monkeypatch):
    monkeypatch.delenv("BENCHMARK_STATELESS", raising=False)
    monkeypatch.setattr(runner_module, "_get_memory_store", lambda: None)
    monkeypatch.setattr(runner_module, "summarize_and_window", lambda messages: messages)

    graph = _RecursingGraph()
    trace = asyncio.run(
        runner_module.run_agent_trace(
            graph,
            "Compute the leverage effect.",
            history=[HumanMessage(content="Compute the leverage effect.")],
        )
    )

    assert trace["answer"] == "0.9273"
    assert [msg.content for msg in trace["updated_history"]] == ["Compute the leverage effect.", "0.9273"]
    budget_step = next(step for step in trace["steps"] if step["node"] == "budget_summary")
    assert budget_step["budget_exits"][0]["category"] == "recursion_limit"
