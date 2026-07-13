import pytest

from src.ai.guardrails import GuardrailError, validate_and_clamp


def test_caps_k_at_node_count(small_graph):
    out = validate_and_clamp("targeted_attack", {"metric": "degree", "k": 9999}, small_graph)
    assert out["k"] == small_graph.number_of_nodes()


def test_floors_k_at_one(small_graph):
    out = validate_and_clamp("targeted_attack", {"metric": "degree", "k": 0}, small_graph)
    assert out["k"] == 1


def test_caps_budget_at_ten(small_graph):
    out = validate_and_clamp("defend", {"budget": 50, "max_distance_km": 3000}, small_graph)
    assert out["budget"] == 10


def test_rejects_unknown_tool(small_graph):
    with pytest.raises(GuardrailError):
        validate_and_clamp("rm_rf", {}, small_graph)


def test_rejects_bad_metric(small_graph):
    with pytest.raises(GuardrailError):
        validate_and_clamp("targeted_attack", {"metric": "evil", "k": 3}, small_graph)


def test_rejects_missing_required_arg(small_graph):
    with pytest.raises(GuardrailError):
        validate_and_clamp("targeted_attack", {"metric": "degree"}, small_graph)


def test_clamps_radius_to_positive(small_graph):
    out = validate_and_clamp(
        "geographic_attack", {"lat": 40.0, "lon": -74.0, "radius_km": -5}, small_graph
    )
    assert out["radius_km"] >= 1.0


def test_clamps_max_distance_to_floor(small_graph):
    out = validate_and_clamp("defend", {"budget": 2, "max_distance_km": 10}, small_graph)
    assert out["max_distance_km"] == 100.0


def test_does_not_mutate_caller_args(small_graph):
    args = {"metric": "degree", "k": 9999}
    validate_and_clamp("targeted_attack", args, small_graph)
    assert args["k"] == 9999  # caller's dict is untouched
