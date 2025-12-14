"""
Unit tests for defense strategy functions.

Tests cover greedy edge addition and node hardening with input validation.
"""
import pytest
import warnings
import networkx as nx
from src.defenses import greedy_edge_addition, node_hardening_list


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

    def test_negative_max_distance_raises_error(self, simple_digraph):
        """Should raise ValueError for negative max_distance_km."""
        with pytest.raises(ValueError, match="max_distance_km must be positive"):
            greedy_edge_addition(simple_digraph, budget=1, max_distance_km=-100)

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
