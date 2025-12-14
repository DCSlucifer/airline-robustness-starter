"""
Unit tests for visualization utilities.

Tests cover node emphasis computation and layer building.
"""
import pytest
import networkx as nx
from src.viz import compute_node_emphasis, build_node_layer
from src.constants import ATTACK_NODE_COLOR, HARDENED_NODE_COLOR, EMPHASIZED_NODE_COLOR


class TestComputeNodeEmphasis:
    """Tests for compute_node_emphasis function."""

    def test_returns_dict_for_all_nodes(self, simple_digraph):
        """Should return emphasis dict for every node."""
        emphasis = compute_node_emphasis(simple_digraph, top_n=2, metric="degree")
        assert len(emphasis) == simple_digraph.number_of_nodes()

    def test_top_n_are_emphasized(self, simple_digraph):
        """Top-N nodes should have is_emphasized=True."""
        emphasis = compute_node_emphasis(simple_digraph, top_n=2, metric="degree")
        emphasized_count = sum(1 for e in emphasis.values() if e.get("is_emphasized"))
        assert emphasized_count >= 2

    def test_removed_nodes_are_attack_colored(self, simple_digraph):
        """Removed nodes should have attack color."""
        emphasis = compute_node_emphasis(
            simple_digraph, top_n=2, metric="degree",
            removed_nodes={"AAA"}
        )
        assert emphasis["AAA"]["color"] == list(ATTACK_NODE_COLOR)

    def test_hardened_nodes_are_hardened_colored(self, simple_digraph):
        """Hardened nodes should have hardened color."""
        emphasis = compute_node_emphasis(
            simple_digraph, top_n=2, metric="degree",
            hardened_nodes={"BBB"}
        )
        assert emphasis["BBB"]["color"] == list(HARDENED_NODE_COLOR)

    def test_various_metrics(self, simple_digraph):
        """Should work with different centrality metrics."""
        for metric in ["degree", "betweenness", "pagerank"]:
            emphasis = compute_node_emphasis(simple_digraph, top_n=2, metric=metric)
            assert len(emphasis) == simple_digraph.number_of_nodes()


class TestBuildNodeLayer:
    """Tests for build_node_layer function."""

    def test_handles_missing_coords_gracefully(self):
        """Nodes without lat/lon should be skipped without error."""
        G = nx.DiGraph()
        G.add_node("A", lat=0.0, lon=0.0, name="Alpha")
        G.add_node("B", name="Beta")  # No coords!

        emphasis = {"A": {"size": 80000, "color": [255, 0, 0], "is_emphasized": True}}
        layer, df = build_node_layer(G, emphasis)

        # Should only include node A
        assert len(df) == 1
        assert df.iloc[0]["iata"] == "A"

    def test_returns_none_for_empty(self):
        """Should return None layer for empty dataframe."""
        G = nx.DiGraph()
        G.add_node("A")  # No coords

        emphasis = {}
        layer, df = build_node_layer(G, emphasis)

        assert layer is None or df.empty

    def test_includes_all_valid_nodes(self, simple_digraph):
        """Should include all nodes with valid coordinates."""
        emphasis = {n: {"size": 50000, "color": [100, 100, 100], "is_emphasized": False}
                   for n in simple_digraph.nodes()}
        layer, df = build_node_layer(simple_digraph, emphasis)

        # All simple_digraph nodes have coords (from conftest)
        assert len(df) == simple_digraph.number_of_nodes()
