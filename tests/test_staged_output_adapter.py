from langchain_core.messages import AIMessage

from agent.nodes.output_adapter import output_adapter
from staged_test_utils import make_state


def test_output_adapter_wraps_json_answer():
    state = make_state(
        "Return JSON.",
        answer_contract={"format": "json", "requires_adapter": True, "wrapper_key": "answer"},
    )
    state["messages"].append(AIMessage(content="0.9274"))

    result = output_adapter(state)

    assert result["messages"][0].content == '{"answer": 0.9274}'


def test_output_adapter_does_not_double_wrap_existing_json_answer():
    state = make_state(
        "Return JSON.",
        answer_contract={"format": "json", "requires_adapter": True, "wrapper_key": "answer"},
    )
    state["messages"].append(AIMessage(content='{"answer": 0.9274}'))

    result = output_adapter(state)

    assert result["messages"] == []


def test_output_adapter_wraps_xml_answer():
    state = make_state(
        "Return XML.",
        answer_contract={"format": "xml", "requires_adapter": True, "xml_root_tag": "answer"},
    )
    state["messages"].append(AIMessage(content="net seller"))

    result = output_adapter(state)

    assert result["messages"][0].content == "<answer>net seller</answer>"
