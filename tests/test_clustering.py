"""
Unit tests for clustering module.

Tests cover community and geographic clustering algorithms.
"""

import networkx as nx
import pytest

from src.clustering import (
    cluster_aggregates,
    community_clustering,
    geographic_clustering,
    get_unclustered_nodes,
)


class TestCommunityClustering:
    """Tests for community_clustering function."""

    def test_returns_dict_for_all_nodes(self, simple_digraph):
        """Should return a cluster ID for every node."""
        clusters = community_clustering(simple_digraph)
        assert isinstance(clusters, dict)
        assert len(clusters) == simple_digraph.number_of_nodes()

    def test_empty_graph(self, empty_digraph):
        """Should return empty dict for empty graph."""
        clusters = community_clustering(empty_digraph)
        assert clusters == {}

    def test_cluster_ids_are_integers(self, simple_digraph):
        """Cluster IDs should be integers."""
        clusters = community_clustering(simple_digraph)
        for cluster_id in clusters.values():
            assert isinstance(cluster_id, int)

    def test_cluster_ids_are_stable_across_node_insertion_order(self):
        first = nx.DiGraph([("A", "B"), ("C", "D")])
        second = nx.DiGraph()
        second.add_nodes_from(["D", "C", "B", "A"])
        second.add_edges_from([("C", "D"), ("A", "B")])

        assert community_clustering(first) == community_clustering(second)


class TestGeographicClustering:
    """Tests for geographic_clustering function."""

    def test_returns_dict_for_all_nodes(self, simple_digraph):
        """Should return a cluster ID for every node."""
        clusters = geographic_clustering(simple_digraph, grid_size_deg=5.0)
        assert len(clusters) == simple_digraph.number_of_nodes()

    def test_nodes_in_same_grid_cell(self):
        """Nodes in same grid cell should have same cluster ID."""
        G = nx.DiGraph()
        G.add_node("A", lat=0.5, lon=0.5)
        G.add_node("B", lat=0.6, lon=0.6)  # Same 5° cell
        G.add_node("C", lat=10.0, lon=10.0)  # Different cell

        clusters = geographic_clustering(G, grid_size_deg=5.0)
        assert clusters["A"] == clusters["B"]
        assert clusters["A"] != clusters["C"]

    def test_empty_graph(self, empty_digraph):
        """Should return empty dict for empty graph."""
        clusters = geographic_clustering(empty_digraph)
        assert clusters == {}

    @pytest.mark.parametrize("grid_size", [0.0, -1.0, float("nan"), float("inf")])
    def test_grid_size_must_be_positive_and_finite(self, simple_digraph, grid_size):
        with pytest.raises(ValueError, match="grid_size_deg must be positive and finite"):
            geographic_clustering(simple_digraph, grid_size_deg=grid_size)

    def test_invalid_coordinates_are_left_unclustered(self):
        graph = nx.DiGraph()
        graph.add_node("valid", lat=1.0, lon=2.0)
        graph.add_node("missing", lat=None, lon=None)
        graph.add_node("nonfinite", lat=float("nan"), lon=0.0)
        graph.add_node("out_of_range", lat=91.0, lon=0.0)

        clusters = geographic_clustering(graph, grid_size_deg=5.0)

        assert set(clusters) == {"valid"}
        assert get_unclustered_nodes(graph, clusters) == [
            "valid",
            "missing",
            "nonfinite",
            "out_of_range",
        ]

    def test_grid_cluster_ids_are_stable_across_node_order(self):
        first = nx.DiGraph()
        first.add_node("west", lat=0.0, lon=-10.0)
        first.add_node("east", lat=0.0, lon=10.0)
        second = nx.DiGraph()
        second.add_node("east", lat=0.0, lon=10.0)
        second.add_node("west", lat=0.0, lon=-10.0)

        assert geographic_clustering(first, 5.0) == geographic_clustering(second, 5.0)


class TestClusterAggregates:
    """Tests for cluster_aggregates function."""

    def test_skips_small_clusters(self):
        """Clusters smaller than MIN_CLUSTER_SIZE should be skipped."""
        G = nx.DiGraph()
        G.add_node("A", lat=0.0, lon=0.0)
        G.add_node("B", lat=0.0, lon=0.0)  # Only 2 nodes
        clusters = {"A": 0, "B": 0}

        # MIN_CLUSTER_SIZE=3, so this should return empty
        aggs = cluster_aggregates(G, clusters)
        assert aggs == []

    def test_returns_list_of_dicts(self, simple_digraph):
        """Should return list of aggregate dicts for valid clusters."""
        # Create a cluster with enough nodes
        clusters = {node: 0 for node in simple_digraph.nodes()}
        aggs = cluster_aggregates(simple_digraph, clusters)

        if aggs:  # Only check if cluster meets MIN_CLUSTER_SIZE
            assert isinstance(aggs[0], dict)
            assert "cluster_id" in aggs[0]
            assert "centroid_lat" in aggs[0]
            assert "node_count" in aggs[0]

    def test_cluster_without_finite_coordinates_is_not_rendered_at_origin(self):
        graph = nx.DiGraph()
        for node in ["A", "B", "C"]:
            graph.add_node(node, lat=None, lon=None)

        assert cluster_aggregates(graph, {"A": 0, "B": 0, "C": 0}) == []

    def test_aggregates_ignore_stale_nodes_and_have_stable_order(self):
        graph = nx.DiGraph()
        for node, lat in [("C", 2.0), ("B", 1.0), ("A", 0.0)]:
            graph.add_node(node, lat=lat, lon=0.0)
        first = {"STALE": 0, "C": 0, "A": 0, "B": 0}
        second = {"B": 0, "A": 0, "C": 0, "STALE": 0}

        first_result = cluster_aggregates(graph, first)
        second_result = cluster_aggregates(graph, second)

        assert first_result == second_result
        assert first_result[0]["member_nodes"] == ["A", "B", "C"]
        assert first_result[0]["node_count"] == 3


def test_get_unclustered_nodes_includes_nodes_missing_from_partial_mapping():
    graph = nx.DiGraph()
    graph.add_nodes_from(["A", "B", "C", "D"])

    assert get_unclustered_nodes(graph, {"A": 0, "B": 0, "C": 0, "STALE": 0}) == ["D"]
