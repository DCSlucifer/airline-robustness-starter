from src.ai.eval.golden_set import GOLDEN_SET
from src.ai.eval.runner import evaluate, format_report
from src.ai.schemas import ToolSelection


class ScriptedClient:
    """Returns a preset ToolSelection per query; explain unused in eval."""

    def __init__(self, mapping):
        self._mapping = mapping

    def select_tool(self, query, tools):
        return self._mapping[query]

    def explain(self, query, tool_name, result):
        return ""


def _perfect_mapping(dataset):
    return {
        c["query"]: ToolSelection(
            name=c["expected_tool"], arguments=dict(c.get("expected_args", {}))
        )
        for c in dataset
    }


def test_perfect_client_scores_100_percent():
    report = evaluate(ScriptedClient(_perfect_mapping(GOLDEN_SET)), GOLDEN_SET)
    assert report.tool_accuracy == 1.0
    assert report.arg_accuracy == 1.0
    assert report.n_cases == len(GOLDEN_SET)


def test_wrong_tool_lowers_accuracy():
    dataset = [
        {"query": "q1", "expected_tool": "defend", "expected_args": {"budget": 3}},
        {"query": "q2", "expected_tool": "edge_attack", "expected_args": {"m": 5}},
    ]
    mapping = {
        "q1": ToolSelection(name="defend", arguments={"budget": 3}),
        "q2": ToolSelection(name="targeted_attack", arguments={}),  # wrong tool
    }
    report = evaluate(ScriptedClient(mapping), dataset)
    assert report.tool_accuracy == 0.5
    assert report.arg_accuracy == 0.5


def test_right_tool_wrong_args_counts_as_arg_miss():
    dataset = [{"query": "q1", "expected_tool": "defend", "expected_args": {"budget": 3}}]
    mapping = {"q1": ToolSelection(name="defend", arguments={"budget": 99})}
    report = evaluate(ScriptedClient(mapping), dataset)
    assert report.tool_accuracy == 1.0
    assert report.arg_accuracy == 0.0


def test_golden_set_well_formed():
    from src.ai.tools import TOOL_NAMES

    assert len(GOLDEN_SET) >= 10
    for c in GOLDEN_SET:
        assert c["expected_tool"] in TOOL_NAMES
        assert c["query"]


def test_format_report_contains_accuracy():
    text = format_report(evaluate(ScriptedClient(_perfect_mapping(GOLDEN_SET)), GOLDEN_SET))
    assert "Tool-selection accuracy" in text
    assert "100.0%" in text
