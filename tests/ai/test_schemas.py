from src.ai.schemas import ToolSelection, AssistantResult


def test_tool_selection_holds_name_and_arguments():
    sel = ToolSelection(name="targeted_attack", arguments={"metric": "degree", "k": 10})
    assert sel.name == "targeted_attack"
    assert sel.arguments["k"] == 10


def test_assistant_result_roundtrips_to_dict():
    res = AssistantResult(
        query="what if we lose 5 hubs?",
        tool_name="targeted_attack",
        arguments={"metric": "degree", "k": 5},
        metrics={"baseline": {}, "after": {}},
        explanation="Connectivity dropped.",
    )
    d = res.model_dump()
    assert d["tool_name"] == "targeted_attack"
    assert d["explanation"] == "Connectivity dropped."
