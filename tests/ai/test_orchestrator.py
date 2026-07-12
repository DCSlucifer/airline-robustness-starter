import pytest

from src.ai.guardrails import GuardrailError
from src.ai.llm_client import FakeLLMClient
from src.ai.orchestrator import run_whatif
from src.ai.schemas import AssistantResult, ToolSelection


def test_run_whatif_end_to_end_grounded(small_graph):
    fake = FakeLLMClient(
        selection=ToolSelection(name="targeted_attack", arguments={"metric": "degree", "k": 2}),
        explanation="Two hubs removed; connectivity decreased.",
    )
    result = run_whatif("what if we lose the 2 biggest hubs?", small_graph, fake)
    assert isinstance(result, AssistantResult)
    assert result.tool_name == "targeted_attack"
    assert result.arguments["k"] == 2
    assert result.metrics["after"]["n_nodes"] < result.metrics["baseline"]["n_nodes"]
    assert result.explanation == "Two hubs removed; connectivity decreased."


def test_run_whatif_clamps_unsafe_arguments(small_graph):
    fake = FakeLLMClient(
        selection=ToolSelection(name="targeted_attack", arguments={"metric": "degree", "k": 9999}),
        explanation="ok",
    )
    result = run_whatif("destroy everything", small_graph, fake)
    assert result.arguments["k"] == small_graph.number_of_nodes()


def test_run_whatif_rejects_unknown_tool(small_graph):
    fake = FakeLLMClient(
        selection=ToolSelection(name="hack_the_mainframe", arguments={}),
        explanation="ok",
    )
    with pytest.raises(GuardrailError):
        run_whatif("do something bad", small_graph, fake)
