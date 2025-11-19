"""
Network robustness metrics and topological analysis.

This module implements functions to compute key metrics for analyzing network
robustness, including Giant Connected Components (Weakly/Strongly), Average
Shortest Path Length (ASPL), Diameter, and reachability within a specific
number of hops.
"""
from __future__ import annotations
import networkx as nx
import pandas as pd
from typing import Dict, Tuple, Set

def gwcc(G: nx.DiGraph) -> Set[any]:
    """
    Identifies the Giant Weakly Connected Component (GWCC) of the graph.

    A weakly connected component is a set of nodes where every node is reachable
    from every other node if edge directions are ignored.

    Args:
        G: The input directed graph.

    Returns:
        A set of nodes belonging to the largest weakly connected component.
    """
    if G.number_of_nodes() == 0:
        return set()

    # Get all weakly connected components
    comps = list(nx.weakly_connected_components(G))

    # Return the largest one by node count
    return max(comps, key=len) if comps else set()

def gscc(G: nx.DiGraph) -> Set[any]:
    """
    Identifies the Giant Strongly Connected Component (GSCC) of the graph.

    A strongly connected component is a set of nodes where every node is reachable
    from every other node following the edge directions.

    Args:
        G: The input directed graph.

    Returns:
        A set of nodes belonging to the largest strongly connected component.
    """
    if G.number_of_nodes() == 0:
        return set()

    # Get all strongly connected components
    comps = list(nx.strongly_connected_components(G))

    # Return the largest one by node count
    return max(comps, key=len) if comps else set()

def aspl_and_diameter_on_gwcc(G: nx.DiGraph) -> Tuple[float, int]:
    """
    Calculates the Average Shortest Path Length (ASPL) and Diameter on the GWCC.

    These metrics are computed on the undirected view of the GWCC to represent
    underlying structural efficiency.

    Args:
        G: The input directed graph.

    Returns:
        A tuple containing (ASPL, Diameter). Returns (inf, 0) if the component is trivial.
    """
    # Extract the subgraph for the GWCC and convert to undirected for metric calculation
    W = G.subgraph(gwcc(G)).to_undirected()

    if W.number_of_nodes() <= 1:
        return float("inf"), 0

    try:
        aspl = nx.average_shortest_path_length(W)
    except nx.NetworkXError:
        # Should not happen if W is connected, but safe fallback
        aspl = float("inf")

    try:
        diameter = nx.diameter(W)
    except nx.NetworkXError:
        diameter = 0

    return aspl, diameter

def percent_od_within_hops(G: nx.DiGraph, H: int = 4) -> float:
    """
    Calculates the percentage of Origin-Destination (OD) pairs reachable within H hops.

    This metric indicates how well-connected the network is for practical travel,
    assuming a maximum acceptable number of transfers. Computed on the GWCC.

    Args:
        G: The input directed graph.
        H: The maximum number of hops allowed.

    Returns:
        The fraction of reachable pairs (0.0 to 1.0).
    """
    W = G.subgraph(gwcc(G)).to_undirected()
    n = W.number_of_nodes()

    if n <= 1:
        return 0.0

    # Compute all-pairs shortest paths within H cutoff
    count = 0
    total = n * (n - 1)  # Total possible directed pairs in the component (excluding self-loops)

    for s in W.nodes():
        # Get lengths of paths from source 's' up to length H
        sp = nx.single_source_shortest_path_length(W, s, cutoff=H)

        # Count reachable nodes, excluding the source itself
        count += sum(1 for t, d in sp.items() if t != s and d <= H)

    return count / total

def topological_report(G: nx.DiGraph, H: int = 4) -> Dict[str, any]:
    """
    Generates a comprehensive report of topological metrics for the graph.

    Args:
        G: The input directed graph.
        H: The hop limit for the reachability metric.

    Returns:
        A dictionary containing key metrics:
        - n_nodes, n_edges
        - gwcc_frac: Fraction of nodes in the Giant Weakly Connected Component.
        - gscc_frac: Fraction of nodes in the Giant Strongly Connected Component.
        - n_components: Number of weakly connected components.
        - aspl_gwcc: Average Shortest Path Length of the GWCC.
        - diameter_gwcc: Diameter of the GWCC.
        - pct_od_within_H: Percentage of pairs reachable within H hops.
    """
    n = G.number_of_nodes()
    gw = gwcc(G)
    gs = gscc(G)
    aspl, diam = aspl_and_diameter_on_gwcc(G)
    pctH = percent_od_within_hops(G, H=H)

    return {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "gwcc_frac": len(gw) / n if n else 0,
        "gscc_frac": len(gs) / n if n else 0,
        "n_components": nx.number_weakly_connected_components(G),
        "aspl_gwcc": aspl,
        "diameter_gwcc": diam,
        "pct_od_within_H": pctH,
    }
