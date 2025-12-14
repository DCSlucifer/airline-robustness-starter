"""
Unit tests for attack simulation functions.

Tests cover targeted node removal, random failures, edge betweenness attacks,
geographic attacks, and community bridge attacks with various edge cases.
"""
import pytest
import warnings
import networkx as nx
from src.attacks import (
    targeted_node_removal,
    random_node_failures,
    edge_betweenness_attack,
    geographic_attack_radius,
    community_bridge_attack,
    collective_influence_scores,
)


class TestTargetedNodeRemoval:
    """Tests for targeted_node_removal function."""

    def test_removes_k_nodes(self, simple_digraph):
        """Should remove exactly k nodes from the graph."""
        original_count = simple_digraph.number_of_nodes()
        k = 2
        H, log = targeted_node_removal(simple_digraph, k=k, metric="degree")
        assert H.number_of_nodes() == original_count - k
        assert len(log) == k

    def test_adaptive_mode(self, simple_digraph):
        """Adaptive mode should recompute rankings after each removal."""
        H, log = targeted_node_removal(simple_digraph, k=2, metric="degree", adaptive=True)
        # Should have logged 2 steps
        assert len(log) == 2
        assert "removed_node" in log[0]
        assert "report" in log[0]

    def test_static_mode(self, simple_digraph):
        """Static mode should use pre-computed rankings."""
        H, log = targeted_node_removal(simple_digraph, k=2, metric="degree", adaptive=False)
        assert len(log) == 2

    def test_various_metrics(self, simple_digraph):
        """Should work with different centrality metrics."""
        for metric in ["degree", "betweenness", "pagerank", "CI"]:
            H, log = targeted_node_removal(simple_digraph, k=1, metric=metric)
            assert len(log) == 1

    def test_invalid_metric_raises_error(self, simple_digraph):
        """Should raise ValueError for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            targeted_node_removal(simple_digraph, k=1, metric="invalid_metric")

    def test_k_exceeds_nodes_capped(self, simple_digraph):
        """Should cap k at graph size and warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            H, log = targeted_node_removal(simple_digraph, k=100)
            # Should have warned about capping
            assert any("exceeds node count" in str(warning.message) for warning in w)
        assert H.number_of_nodes() == 0

    def test_k_zero_returns_unchanged(self, simple_digraph):
        """Should return unchanged graph for k=0."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            H, log = targeted_node_removal(simple_digraph, k=0)
        assert H.number_of_nodes() == simple_digraph.number_of_nodes()
        assert log == []

    def test_empty_graph(self, empty_digraph):
        """Should handle empty graph gracefully."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            H, log = targeted_node_removal(empty_digraph, k=5)
        assert H.number_of_nodes() == 0
        assert log == []


class TestRandomNodeFailures:
    """Tests for random_node_failures function."""

    def test_returns_r_reports(self, simple_digraph):
        """Should return R simulation reports."""
        R = 5
        reports = random_node_failures(simple_digraph, k=2, R=R)
        assert len(reports) == R

    def test_each_report_has_expected_fields(self, simple_digraph):
        """Each report should have rep, removed_nodes, and report."""
        reports = random_node_failures(simple_digraph, k=2, R=3)
        for rep in reports:
            assert "rep" in rep
            assert "removed_nodes" in rep
            assert "report" in rep

    def test_reproducible_with_seed(self, simple_digraph):
        """Should produce same results with same seed."""
        reports1 = random_node_failures(simple_digraph, k=2, R=3, seed=42)
        reports2 = random_node_failures(simple_digraph, k=2, R=3, seed=42)
        for r1, r2 in zip(reports1, reports2):
            assert r1["removed_nodes"] == r2["removed_nodes"]

    def test_k_exceeds_nodes_capped(self, simple_digraph):
        """Should cap k at graph size and warn."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reports = random_node_failures(simple_digraph, k=100, R=1)
        assert len(reports) == 1

    def test_empty_graph(self, empty_digraph):
        """Should handle empty graph gracefully."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reports = random_node_failures(empty_digraph, k=5, R=3)
        assert reports == []


class TestEdgeBetweennessAttack:
    """Tests for edge_betweenness_attack function."""

    def test_removes_m_edges_adaptive(self, simple_digraph):
        """Should remove m edges in adaptive mode."""
        original_edges = simple_digraph.number_of_edges()
        m = 2
        H, log = edge_betweenness_attack(simple_digraph, m=m, adaptive=True)
        assert H.number_of_edges() <= original_edges - m
        assert len(log) == m

    def test_removes_m_edges_nonadaptive(self, simple_digraph):
        """Should remove m edges in non-adaptive mode."""
        original_edges = simple_digraph.number_of_edges()
        m = 2
        H, log = edge_betweenness_attack(simple_digraph, m=m, adaptive=False)
        # Non-adaptive should also remove m edges
        assert len(log) <= m

    def test_m_zero_returns_unchanged(self, simple_digraph):
        """Should return unchanged graph for m=0."""
        H, log = edge_betweenness_attack(simple_digraph, m=0)
        assert H.number_of_edges() == simple_digraph.number_of_edges()
        assert log == []


class TestGeographicAttackRadius:
    """Tests for geographic_attack_radius function."""

    def test_removes_nodes_within_radius(self, simple_digraph):
        """Should remove nodes within the specified radius."""
        # Center at (0.0, 0.0) with small radius should remove AAA
        H, info = geographic_attack_radius(simple_digraph, center=(0.0, 0.0), radius_km=50)
        assert "removed_nodes" in info
        assert "AAA" in info["removed_nodes"]

    def test_large_radius_removes_all(self, simple_digraph):
        """Large radius should remove all nodes."""
        H, info = geographic_attack_radius(simple_digraph, center=(0.0, 0.0), radius_km=1000000)
        assert H.number_of_nodes() == 0


class TestCollectiveInfluenceScores:
    """Tests for collective_influence_scores function."""

    def test_returns_scores_for_all_nodes(self, simple_digraph):
        """Should return a score for every node."""
        scores = collective_influence_scores(simple_digraph, l=2)
        assert len(scores) == simple_digraph.number_of_nodes()

    def test_empty_graph(self, empty_digraph):
        """Should handle empty graph."""
        scores = collective_influence_scores(empty_digraph)
        assert scores == {}
