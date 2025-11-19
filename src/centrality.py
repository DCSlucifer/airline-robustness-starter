"""
Centrality metric computations for network analysis.

This module provides functions to calculate and aggregate various centrality
measures (degree, betweenness, PageRank) for nodes in the graph.
"""
from __future__ import annotations
import networkx as nx
import pandas as pd
from typing import Dict

def node_centralities(G: nx.DiGraph) -> pd.DataFrame:
    """
    Calculates multiple centrality metrics for all nodes in the graph.

    Metrics computed:
    - In-Degree: Number of incoming edges.
    - Out-Degree: Number of outgoing edges.
    - Total Degree: Sum of in-degree and out-degree.
    - Betweenness Centrality: Fraction of shortest paths passing through the node.
    - PageRank: Measure of node importance based on link structure.

    Args:
        G: The input directed graph.

    Returns:
        A pandas DataFrame containing centrality scores for each node,
        sorted by total degree in descending order.
    """
    # Calculate degree centralities
    deg_in = dict(G.in_degree())
    deg_out = dict(G.out_degree())

    # Calculate Betweenness Centrality
    # Note: This can be computationally expensive for large graphs (O(V*E)).
    # normalized=True ensures values are comparable across graphs of different sizes.
    btw = nx.betweenness_centrality(G, normalized=True)

    # Calculate PageRank
    # alpha=0.85 is the standard damping factor.
    pr = nx.pagerank(G, alpha=0.85)

    # Aggregate results into a DataFrame
    df = pd.DataFrame({
        "node": list(G.nodes()),
        "deg_in": [deg_in.get(n, 0) for n in G.nodes()],
        "deg_out": [deg_out.get(n, 0) for n in G.nodes()],
        "betweenness": [btw.get(n, 0.0) for n in G.nodes()],
        "pagerank": [pr.get(n, 0.0) for n in G.nodes()],
    })

    # Compute total degree as a summary metric
    df["deg_total"] = df["deg_in"] + df["deg_out"]

    # Return sorted DataFrame for easier analysis
    return df.sort_values("deg_total", ascending=False).reset_index(drop=True)
