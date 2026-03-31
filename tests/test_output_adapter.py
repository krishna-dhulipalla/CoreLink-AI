from langchain_core.messages import AIMessage

from agent.nodes.output_adapter import output_adapter
from agent.runtime_support import extract_answer_contract, infer_benchmark_overrides
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


def test_infer_benchmark_overrides_explicit_benchmark_activates_officeqa_runtime(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")

    overrides = infer_benchmark_overrides("What was AAPL's EBITDA in fiscal year 2024?")

    assert overrides["benchmark_name"] == "officeqa"
    assert overrides["benchmark_adapter"] == "officeqa"
    assert overrides["officeqa_mode"] is True
    assert overrides["officeqa_xml_contract"] is True
    assert "document_retrieval" in overrides["benchmark_policy"]["allowed_families"]
    assert "source family grounding" in overrides["benchmark_policy"]["validation_dimensions"]
    assert overrides["benchmark_policy"]["output_normalization"]["final_answer_tag"] == "FINAL_ANSWER"


def test_extract_answer_contract_uses_explicit_officeqa_benchmark(monkeypatch):
    monkeypatch.setenv("BENCHMARK_NAME", "officeqa")

    contract = extract_answer_contract("Use the provided files to compute the exact answer.")

    assert contract.format == "xml"
    assert contract.requires_adapter is True
    assert contract.xml_root_tag == "FINAL_ANSWER"


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


def test_output_adapter_normalizes_overlong_existing_final_answer_block():
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
        AIMessage(
            content=(
                "<REASONING>Work shown here.</REASONING>\n"
                "<FINAL_ANSWER>Thus, the answer is 40.90.\n"
                "&lt;FINAL_ANSWER&gt;40.90&lt;/FINAL_ANSWER&gt;</FINAL_ANSWER>"
            )
        )
    )

    result = output_adapter(state)
    content = result["messages"][0].content
    final_block = content.split("<FINAL_ANSWER>", 1)[1].split("</FINAL_ANSWER>", 1)[0].strip()

    assert final_block == "40.90"


def test_output_adapter_does_not_extract_year_from_insufficient_officeqa_reasoning():
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
        AIMessage(
            content=(
                "<REASONING>The monthly values for 1953 are not present in the provided evidence. "
                "Insufficient data in provided evidence.</REASONING>\n"
                "<FINAL_ANSWER>\n\n</FINAL_ANSWER>"
            )
        )
    )

    result = output_adapter(state)
    content = result["messages"][0].content
    final_block = content.split("<FINAL_ANSWER>", 1)[1].split("</FINAL_ANSWER>", 1)[0].strip()

    assert "1953" != final_block
    assert "Insufficient data" in final_block
