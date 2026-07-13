"""
Unit tests for defense strategy functions.

Tests cover greedy edge addition and node hardening with input validation.
"""

import warnings

import networkx as nx
import pytest

import src.defenses as defenses
from src.defenses import _candidate_pairs, greedy_edge_addition, node_hardening_list


class TestGreedyEdgeAddition:
    """Tests for greedy_edge_addition function."""

    def test_adds_edges_up_to_budget(self, simple_digraph):
        """Should add up to 'budget' bidirectional edges."""
        budget = 2
        H, log = greedy_edge_addition(simple_digraph, budget=budget, max_distance_km=500)
        # Each step adds 2 edges (bidirectional)
        assert len(log) <= budget

    def test_each_log_entry_has_expected_fields(self, simple_digraph):
        """Each log entry should have step, added_edges, report_after."""
        H, log = greedy_edge_addition(simple_digraph, budget=1, max_distance_km=500)
        if log:  # May be empty if no candidates found
            assert "step" in log[0]
            assert "added_edges" in log[0]
            assert "report_after" in log[0]

    def test_budget_zero_returns_unchanged(self, simple_digraph):
        """Should return unchanged graph for budget=0."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            H, log = greedy_edge_addition(simple_digraph, budget=0)
            assert any("budget must be positive" in str(warning.message) for warning in w)
        assert H.number_of_edges() == simple_digraph.number_of_edges()
        assert log == []

    @pytest.mark.parametrize("distance", [0.0, -100.0, float("nan"), float("inf")])
    def test_invalid_max_distance_raises_error(self, simple_digraph, distance):
        """Should reject non-positive and non-finite distance limits."""
        with pytest.raises(ValueError, match="max_distance_km must be positive"):
            greedy_edge_addition(simple_digraph, budget=1, max_distance_km=distance)

    def test_small_graph_warning(self, single_node_digraph):
        """Should warn for graph with fewer than 2 nodes."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            H, log = greedy_edge_addition(single_node_digraph, budget=1)
            assert any("fewer than 2 nodes" in str(warning.message) for warning in w)
        assert log == []


class TestNodeHardeningList:
    """Tests for node_hardening_list function."""

    def test_returns_top_n_nodes(self, simple_digraph):
        """Should return top_n nodes sorted by metric."""
        result = node_hardening_list(simple_digraph, top_n=3, metric="degree")
        assert len(result) == 3

    def test_various_metrics(self, simple_digraph):
        """Should work with different centrality metrics."""
        for metric in ["degree", "betweenness", "pagerank"]:
            result = node_hardening_list(simple_digraph, top_n=2, metric=metric)
            assert len(result) == 2

    def test_invalid_metric_raises_error(self, simple_digraph):
        """Should raise ValueError for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            node_hardening_list(simple_digraph, top_n=3, metric="invalid_metric")

    def test_top_n_exceeds_nodes(self, simple_digraph):
        """Should return all nodes if top_n exceeds node count."""
        result = node_hardening_list(simple_digraph, top_n=100, metric="degree")
        assert len(result) == simple_digraph.number_of_nodes()

    def test_large_graph_betweenness_uses_deterministic_seed(self, monkeypatch):
        graph = nx.DiGraph()
        graph.add_nodes_from(range(501))
        recorded = {}

        def fake_betweenness(graph, *, k, seed):
            recorded.update(k=k, seed=seed)
            return {node: float(node) for node in graph}

        monkeypatch.setattr(defenses.nx, "betweenness_centrality", fake_betweenness)

        result = node_hardening_list(graph, top_n=2, metric="betweenness")

        assert recorded == {"k": 200, "seed": 42}
        assert result == [500, 499]

    def test_equal_scores_use_stable_node_order(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(["B", "A"])

        assert node_hardening_list(graph, top_n=2, metric="degree") == ["A", "B"]


def test_candidate_pairs_are_deterministic(monkeypatch):
    graph = nx.DiGraph()
    for node, lat, lon in [
        ("D", 3.0, 3.0),
        ("B", 1.0, 1.0),
        ("C", 2.0, 2.0),
        ("A", 0.0, 0.0),
    ]:
        graph.add_node(node, lat=lat, lon=lon)

    monkeypatch.setattr(
        nx.algorithms.community,
        "label_propagation_communities",
        lambda _graph: [{"D", "C"}, {"B", "A"}],
    )

    assert _candidate_pairs(graph, max_distance_km=20000) == [
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
    ]


def test_candidate_pairs_skip_nonfinite_coordinates(monkeypatch):
    graph = nx.DiGraph()
    graph.add_node("A", lat=0.0, lon=0.0)
    graph.add_node("B", lat=1.0, lon=1.0)
    graph.add_node("C", lat=float("nan"), lon=2.0)
    graph.add_node("D", lat=100.0, lon=2.0)
    monkeypatch.setattr(
        nx.algorithms.community,
        "label_propagation_communities",
        lambda _graph: [{"A"}, {"B"}, {"C"}, {"D"}],
    )

    assert _candidate_pairs(graph, max_distance_km=20000) == [("A", "B")]


def test_greedy_edge_addition_breaks_candidate_ties_deterministically(monkeypatch):
    graph = nx.DiGraph()
    for node, lat, lon in [
        ("D", 3.0, 3.0),
        ("B", 1.0, 1.0),
        ("C", 2.0, 2.0),
        ("A", 0.0, 0.0),
    ]:
        graph.add_node(node, lat=lat, lon=lon)
    monkeypatch.setattr(
        nx.algorithms.community,
        "label_propagation_communities",
        lambda _graph: [{"D", "C"}, {"B", "A"}],
    )
    monkeypatch.setattr(
        defenses,
        "topological_report",
        lambda _graph, *, fast_mode: {"gwcc_frac": 1.0, "aspl_gwcc": 1.0},
    )

    result, log = greedy_edge_addition(graph, budget=1, max_distance_km=20000)

    assert log[0]["added_edges"] == [("A", "C"), ("C", "A")]
    assert result.has_edge("A", "C") and result.has_edge("C", "A")
