"""
Unit tests for network metrics functions.

Tests cover GWCC, GSCC, ASPL, diameter, and OD reachability calculations.
"""
import pytest
import networkx as nx
from src.metrics import (
    gwcc,
    gscc,
    aspl_and_diameter_on_gwcc,
    percent_od_within_hops,
    topological_report,
)


class TestGWCC:
    """Tests for Giant Weakly Connected Component."""

    def test_returns_set(self, simple_digraph):
        """Should return a set of nodes."""
        result = gwcc(simple_digraph)
        assert isinstance(result, set)

    def test_all_nodes_in_connected_graph(self, simple_digraph):
        """All nodes should be in GWCC for a connected graph."""
        result = gwcc(simple_digraph)
        assert len(result) == simple_digraph.number_of_nodes()

    def test_empty_graph(self, empty_digraph):
        """Should return empty set for empty graph."""
        result = gwcc(empty_digraph)
        assert result == set()

    def test_disconnected_graph_largest_component(self, disconnected_digraph):
        """Should return the largest component for disconnected graph."""
        result = gwcc(disconnected_digraph)
        # Both components have 2 nodes, either could be returned
        assert len(result) == 2


class TestGSCC:
    """Tests for Giant Strongly Connected Component."""

    def test_returns_set(self, simple_digraph):
        """Should return a set of nodes."""
        result = gscc(simple_digraph)
        assert isinstance(result, set)

    def test_empty_graph(self, empty_digraph):
        """Should return empty set for empty graph."""
        result = gscc(empty_digraph)
        assert result == set()


class TestASPLAndDiameter:
    """Tests for Average Shortest Path Length and Diameter."""

    def test_returns_tuple(self, simple_digraph):
        """Should return a tuple of (aspl, diameter)."""
        result = aspl_and_diameter_on_gwcc(simple_digraph)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_finite_values_for_connected_graph(self, simple_digraph):
        """Connected graph should have finite ASPL."""
        aspl, diameter = aspl_and_diameter_on_gwcc(simple_digraph)
        assert aspl != float("inf")
        assert diameter > 0

    def test_empty_graph_returns_inf(self, empty_digraph):
        """Empty graph should return (inf, 0)."""
        aspl, diameter = aspl_and_diameter_on_gwcc(empty_digraph)
        assert aspl == float("inf")
        assert diameter == 0

    def test_single_node_returns_inf(self, single_node_digraph):
        """Single node should return (inf, 0)."""
        aspl, diameter = aspl_and_diameter_on_gwcc(single_node_digraph)
        assert aspl == float("inf")
        assert diameter == 0


class TestPercentODWithinHops:
    """Tests for OD reachability within H hops."""

    def test_returns_fraction(self, simple_digraph):
        """Should return a value between 0 and 1."""
        result = percent_od_within_hops(simple_digraph, H=4)
        assert 0.0 <= result <= 1.0

    def test_small_hop_limit_reduces_reachability(self, simple_digraph):
        """Smaller H should result in lower or equal reachability."""
        result_h1 = percent_od_within_hops(simple_digraph, H=1)
        result_h4 = percent_od_within_hops(simple_digraph, H=4)
        assert result_h1 <= result_h4

    def test_empty_graph_returns_zero(self, empty_digraph):
        """Empty graph should return 0."""
        result = percent_od_within_hops(empty_digraph, H=4)
        assert result == 0.0


class TestTopologicalReport:
    """Tests for comprehensive topological report."""

    def test_returns_dict_with_expected_keys(self, simple_digraph):
        """Should return dict with all expected metric keys."""
        report = topological_report(simple_digraph)
        expected_keys = [
            "n_nodes", "n_edges", "gwcc_frac", "gscc_frac",
            "n_components", "aspl_gwcc", "diameter_gwcc", "pct_od_within_H"
        ]
        for key in expected_keys:
            assert key in report

    def test_node_and_edge_counts(self, simple_digraph):
        """Should correctly report node and edge counts."""
        report = topological_report(simple_digraph)
        assert report["n_nodes"] == simple_digraph.number_of_nodes()
        assert report["n_edges"] == simple_digraph.number_of_edges()

    def test_empty_graph(self, empty_digraph):
        """Should handle empty graph gracefully."""
        report = topological_report(empty_digraph)
        assert report["n_nodes"] == 0
        assert report["n_edges"] == 0
        assert report["gwcc_frac"] == 0
