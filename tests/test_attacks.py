"""
Unit tests for attack simulation functions.

Tests cover targeted node removal, random failures, edge betweenness attacks,
geographic attacks, and community bridge attacks with various edge cases.
"""

import random
import warnings

import networkx as nx
import pytest

import src.attacks as attacks
from src.attacks import (
    collective_influence_scores,
    community_bridge_attack,
    edge_betweenness_attack,
    geographic_attack_radius,
    random_node_failures,
    targeted_node_removal,
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

    @pytest.mark.parametrize("interval", [0, -1])
    def test_report_interval_must_be_positive(self, simple_digraph, interval):
        with pytest.raises(ValueError, match="report_every_n must be positive"):
            targeted_node_removal(simple_digraph, k=1, report_every_n=interval)


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
        for r1, r2 in zip(reports1, reports2, strict=True):
            assert r1["removed_nodes"] == r2["removed_nodes"]

    def test_does_not_mutate_process_random_state(self, simple_digraph):
        random.seed(12345)
        state = random.getstate()

        random_node_failures(simple_digraph, k=2, R=2, seed=7)

        assert random.getstate() == state

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
        assert H.number_of_edges() == original_edges - m
        assert len(log) == m

    def test_nonadaptive_counts_reciprocal_arcs_separately(self):
        graph = nx.DiGraph([("A", "B"), ("B", "A")])

        result, log = edge_betweenness_attack(graph, m=2, adaptive=False, fast_mode=True)

        assert result.number_of_edges() == 0
        assert [entry["removed_edge"] for entry in log] == [("A", "B"), ("B", "A")]
        assert log[-1]["report"] is not None

    def test_nonadaptive_caps_at_directed_edge_count_and_reports_final_step(self):
        graph = nx.DiGraph([("B", "A"), ("A", "B")])

        result, log = edge_betweenness_attack(
            graph, m=99, adaptive=False, fast_mode=True, report_every_n=10
        )

        assert result.number_of_edges() == 0
        assert [entry["removed_edge"] for entry in log] == [("A", "B"), ("B", "A")]
        assert log[0]["report"] is None
        assert log[-1]["report"] is not None

    def test_m_zero_returns_unchanged(self, simple_digraph):
        """Should return unchanged graph for m=0."""
        H, log = edge_betweenness_attack(simple_digraph, m=0)
        assert H.number_of_edges() == simple_digraph.number_of_edges()
        assert log == []

    def test_negative_m_raises(self, simple_digraph):
        with pytest.raises(ValueError, match="m must be non-negative"):
            edge_betweenness_attack(simple_digraph, m=-1)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"report_every_n": 0}, "report_every_n must be positive"),
            ({"report_every_n": -1}, "report_every_n must be positive"),
            ({"recompute_every": 0}, "recompute_every must be positive"),
            ({"recompute_every": -1}, "recompute_every must be positive"),
            ({"k_samples": 0}, "k_samples must be positive"),
            ({"k_samples": -1}, "k_samples must be positive"),
        ],
    )
    def test_sampling_and_recompute_controls_must_be_positive(
        self, simple_digraph, kwargs, message
    ):
        with pytest.raises(ValueError, match=message):
            edge_betweenness_attack(simple_digraph, m=1, **kwargs)


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

    def test_zero_radius_removes_node_at_exact_center(self, simple_digraph):
        result, info = geographic_attack_radius(simple_digraph, center=(0.0, 0.0), radius_km=0.0)

        assert "AAA" in info["removed_nodes"]
        assert "AAA" not in result

    def test_fast_mode_is_forwarded_to_report(self, monkeypatch, simple_digraph):
        recorded = []
        monkeypatch.setattr(
            attacks,
            "topological_report",
            lambda _graph, *, fast_mode: recorded.append(fast_mode) or {"fast": fast_mode},
        )

        _result, info = geographic_attack_radius(
            simple_digraph, center=(0.0, 0.0), radius_km=1.0, fast_mode=True
        )

        assert recorded == [True]
        assert info["report"] == {"fast": True}

    @pytest.mark.parametrize("radius", [-1.0, float("nan"), float("inf")])
    def test_invalid_radius_raises(self, simple_digraph, radius):
        with pytest.raises(ValueError, match="radius_km must be a finite non-negative value"):
            geographic_attack_radius(simple_digraph, center=(0.0, 0.0), radius_km=radius)

    @pytest.mark.parametrize(
        "center", [(float("nan"), 0.0), (float("inf"), 0.0), (91.0, 0.0), (0.0, 181.0), (0.0,)]
    )
    def test_invalid_center_raises(self, simple_digraph, center):
        with pytest.raises(ValueError, match="center"):
            geographic_attack_radius(simple_digraph, center=center, radius_km=10.0)


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

    def test_negative_radius_raises(self, simple_digraph):
        with pytest.raises(ValueError, match="l must be non-negative"):
            collective_influence_scores(simple_digraph, l=-1)


def test_large_graph_betweenness_ranking_uses_deterministic_seed(monkeypatch):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(501))
    recorded = {}

    def fake_betweenness(graph, *, k, seed):
        recorded.update(k=k, seed=seed)
        return {node: float(node) for node in graph}

    monkeypatch.setattr(attacks.nx, "betweenness_centrality", fake_betweenness)

    ranking = attacks._rank_nodes(graph, metric="betweenness")

    assert recorded == {"k": 200, "seed": 42}
    assert ranking[0] == 500


def test_rank_nodes_breaks_equal_scores_stably():
    graph = nx.DiGraph()
    graph.add_nodes_from(["B", "A"])

    assert attacks._rank_nodes(graph, metric="degree") == ["A", "B"]


def test_large_edge_attack_uses_deterministic_sample(monkeypatch):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(501))
    graph.add_edge(0, 500)
    recorded = {}

    def fake_edge_betweenness(graph, *, k, normalized, seed):
        recorded.update(k=k, normalized=normalized, seed=seed)
        return {(0, 500): 1.0}

    monkeypatch.setattr(attacks.nx, "edge_betweenness_centrality", fake_edge_betweenness)

    result, log = edge_betweenness_attack(graph, m=1, adaptive=False, fast_mode=True)

    assert recorded == {"k": 200, "normalized": True, "seed": 42}
    assert log[0]["removed_edge"] == (0, 500)
    assert not result.has_edge(0, 500)


def test_community_bridge_attack_validates_negative_m(simple_digraph):
    with pytest.raises(ValueError, match="m must be non-negative"):
        community_bridge_attack(simple_digraph, m=-1)


@pytest.mark.parametrize("sample_count", [0, -1])
def test_community_bridge_attack_validates_sample_count(simple_digraph, sample_count):
    with pytest.raises(ValueError, match="k_samples must be positive"):
        community_bridge_attack(simple_digraph, m=1, k_samples=sample_count)


def test_community_bridge_zero_is_a_noop_without_detection(monkeypatch, simple_digraph):
    report_modes = []

    def fail_if_called(_graph):
        raise AssertionError("community detection should not run")

    monkeypatch.setattr(nx.algorithms.community, "label_propagation_communities", fail_if_called)
    monkeypatch.setattr(
        attacks,
        "topological_report",
        lambda _graph, *, fast_mode: report_modes.append(fast_mode) or {"fast": fast_mode},
    )

    result, info = community_bridge_attack(simple_digraph, m=0, fast_mode=False)

    assert nx.utils.graphs_equal(result, simple_digraph)
    assert info["removed_edges"] == []
    assert info["report"] == {"fast": False}
    assert report_modes == [False]


def test_community_bridge_attack_samples_large_graph_deterministically(monkeypatch):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(501))
    graph.add_edge(0, 500)
    recorded = {}

    monkeypatch.setattr(
        nx.algorithms.community,
        "label_propagation_communities",
        lambda _graph: [set(range(500)), {500}],
    )

    def fake_edge_betweenness(graph, *, k, normalized, seed):
        recorded.update(k=k, normalized=normalized, seed=seed)
        return {(0, 500): 1.0}

    monkeypatch.setattr(attacks.nx, "edge_betweenness_centrality", fake_edge_betweenness)

    result, info = community_bridge_attack(graph, m=1)

    assert recorded == {"k": 200, "normalized": True, "seed": 42}
    assert info["removed_edges"] == [(0, 500)]
    assert not result.has_edge(0, 500)


def test_community_bridge_attack_breaks_score_ties_deterministically(monkeypatch):
    graph = nx.DiGraph()
    graph.add_edges_from([("B", "D"), ("A", "C")])
    monkeypatch.setattr(
        nx.algorithms.community,
        "label_propagation_communities",
        lambda _graph: [{"D", "C"}, {"B", "A"}],
    )
    monkeypatch.setattr(
        attacks.nx,
        "edge_betweenness_centrality",
        lambda _graph, *, normalized: {("B", "D"): 1.0, ("A", "C"): 1.0},
    )

    _result, info = community_bridge_attack(graph, m=1)

    assert info["removed_edges"] == [("A", "C")]
