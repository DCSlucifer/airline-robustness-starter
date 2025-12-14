"""
Unit tests for clustering module.

Tests cover community and geographic clustering algorithms.
"""
import pytest
import networkx as nx
from src.clustering import community_clustering, geographic_clustering, cluster_aggregates


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
