import networkx as nx

from src.ai.tools import TOOL_SPECS, TOOL_NAMES, run_tool


def test_tool_specs_are_well_formed():
    assert TOOL_NAMES  # non-empty
    for spec in TOOL_SPECS:
        assert spec["name"] in TOOL_NAMES
        assert "description" in spec and spec["description"]
        schema = spec["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert spec["strict"] is True


def test_run_targeted_attack_returns_grounded_metrics(small_graph):
    result = run_tool("targeted_attack", {"metric": "degree", "k": 2}, small_graph)
    assert set(result) >= {"baseline", "after", "removed_nodes"}
    assert len(result["removed_nodes"]) == 2
    assert result["after"]["n_nodes"] < result["baseline"]["n_nodes"]


def test_run_geographic_attack_removes_nearby_nodes(small_graph):
    result = run_tool(
        "geographic_attack", {"lat": 40.5, "lon": -73.5, "radius_km": 300}, small_graph
    )
    assert "AAA" in result["removed_nodes"]
    assert "BBB" in result["removed_nodes"]
    assert "FFF" not in result["removed_nodes"]


def test_run_defend_adds_edges(small_graph):
    result = run_tool("defend", {"budget": 1, "max_distance_km": 20000}, small_graph)
    assert "added_edges" in result
    assert "after" in result


def test_run_tool_rejects_unknown_tool(small_graph):
    import pytest
    with pytest.raises(KeyError):
        run_tool("nonexistent", {}, small_graph)
