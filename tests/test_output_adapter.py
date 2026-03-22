from langchain_core.messages import AIMessage

from agent.nodes.output_adapter import output_adapter
from agent.runtime_support import extract_answer_contract
from test_utils import make_state


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


def test_extract_answer_contract_enables_officeqa_xml(monkeypatch):
    monkeypatch.setenv("OFFICEQA_FINAL_ANSWER_TAGS", "1")

    contract = extract_answer_contract("What were the total expenditures for U.S. national defense in 1940?")

    assert contract.format == "xml"
    assert contract.requires_adapter is True
    assert contract.xml_root_tag == "FINAL_ANSWER"
    assert contract.value_rules["reasoning_tag"] == "REASONING"


def test_output_adapter_wraps_officeqa_reasoning_and_final_answer_tags():
    state = make_state(
        "OfficeQA numeric answer.",
        answer_contract={
            "format": "xml",
            "requires_adapter": True,
            "xml_root_tag": "FINAL_ANSWER",
            "value_rules": {
                "reasoning_tag": "REASONING",
                "final_answer_tag": "FINAL_ANSWER",
                "final_answer_only": True,
            },
        },
    )
    state["messages"].append(
        AIMessage(content="The total expenditures in 1940 were 2,602 million nominal dollars.")
    )

    result = output_adapter(state)
    content = result["messages"][0].content

    assert "<REASONING>" in content
    assert "<FINAL_ANSWER>" in content
    assert "2,602" in content
    assert "million nominal dollars" not in content.split("<FINAL_ANSWER>", 1)[1]
