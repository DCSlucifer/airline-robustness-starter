"""Centrality reports are part of the public analysis API."""

from __future__ import annotations

import networkx as nx
import pytest

from src.centrality import node_centralities


def test_node_centralities_returns_ranked_metrics() -> None:
    graph = nx.DiGraph([("A", "B"), ("A", "C"), ("A", "D"), ("B", "C")])

    report = node_centralities(graph)

    assert report.columns.tolist() == [
        "node",
        "deg_in",
        "deg_out",
        "betweenness",
        "pagerank",
        "deg_total",
    ]
    assert report["deg_total"].is_monotonic_decreasing
    assert report.iloc[0]["node"] == "A"
    assert report.set_index("node").loc["A", "deg_out"] == 3
    assert report["pagerank"].sum() == pytest.approx(1.0)


def test_node_centralities_handles_an_empty_graph() -> None:
    report = node_centralities(nx.DiGraph())

    assert report.empty
    assert "deg_total" in report.columns
