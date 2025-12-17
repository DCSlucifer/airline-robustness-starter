"""
Network robustness metrics and topological analysis.

This module implements functions to compute key metrics for analyzing network
robustness, including Giant Connected Components (Weakly/Strongly), Average
Shortest Path Length (ASPL), Diameter, and reachability within a specific
number of hops.
"""
from __future__ import annotations
from typing import Any, Dict, Set, Tuple

import random
import networkx as nx
import pandas as pd


__all__ = [
    "gwcc",
    "gscc",
    "aspl_and_diameter_on_gwcc",
    "percent_od_within_hops",
    "topological_report",
]

def gwcc(G: nx.DiGraph) -> Set[Any]:
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

def gscc(G: nx.DiGraph) -> Set[Any]:
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

def _gwcc_undirected(G: nx.DiGraph) -> nx.Graph:
    """Undirected view of the GWCC (used for efficiency metrics)."""
    return G.subgraph(gwcc(G)).to_undirected()


def _sample_nodes(W: nx.Graph, k: int, seed: int = 42):
    nodes = list(W.nodes())
    if not nodes:
        return []
    rng = random.Random(seed)
    if k >= len(nodes):
        return nodes
    return rng.sample(nodes, k)


def aspl_sampled_on_gwcc(G: nx.DiGraph, samples: int = 200, seed: int = 42) -> float:
    """
    Approximate ASPL on GWCC using BFS from sampled sources.
    """
    W = _gwcc_undirected(G)
    n = W.number_of_nodes()
    if n <= 1:
        return float("inf")

    sources = _sample_nodes(W, min(samples, n), seed=seed)
    total_dist = 0
    total_cnt = 0

    for s in sources:
        sp = nx.single_source_shortest_path_length(W, s)
        for t, d in sp.items():
            if t != s:
                total_dist += d
                total_cnt += 1

    return (total_dist / total_cnt) if total_cnt else float("inf")


def diameter_two_sweep_on_gwcc(G: nx.DiGraph, sweeps: int = 4, seed: int = 42) -> int:
    """
    Approximate diameter on GWCC using repeated BFS “two-sweep” heuristic.
    """
    W = _gwcc_undirected(G)
    n = W.number_of_nodes()
    if n <= 1:
        return 0

    rng = random.Random(seed)
    cur = rng.choice(list(W.nodes()))
    best = 0

    for _ in range(max(1, sweeps)):
        dist = nx.single_source_shortest_path_length(W, cur)
        if not dist:
            break
        far, dmax = max(dist.items(), key=lambda kv: kv[1])
        best = max(best, dmax)
        cur = far

    return int(best)


def percent_od_within_hops_sampled(G: nx.DiGraph, H: int = 4, samples: int = 200, seed: int = 42) -> float:
    """
    Approximate percent OD pairs within H hops using sampled sources on GWCC.
    """
    W = _gwcc_undirected(G)
    n = W.number_of_nodes()
    if n <= 1:
        return 0.0

    sources = _sample_nodes(W, min(samples, n), seed=seed)
    count = 0
    total = len(sources) * (n - 1)

    for s in sources:
        sp = nx.single_source_shortest_path_length(W, s, cutoff=H)
        count += sum(1 for t, d in sp.items() if t != s and d <= H)

    return (count / total) if total else 0.0


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

def topological_report(G: nx.DiGraph, H: int = 4, fast_mode: bool = False) -> Dict[str, Any]:
    """
    Generates a comprehensive report of topological metrics for the graph.

    Args:
        G: The input directed graph.
        H: The hop limit for the reachability metric.
        fast_mode: If True, skip expensive ASPL/diameter/OD computations for UI responsiveness.

    Returns:
        A dictionary containing key metrics:
        - n_nodes, n_edges
        - gwcc_frac: Fraction of nodes in the Giant Weakly Connected Component.
        - gscc_frac: Fraction of nodes in the Giant Strongly Connected Component.
        - n_components: Number of weakly connected components.
        - aspl_gwcc: Average Shortest Path Length of the GWCC (inf if fast_mode).
        - diameter_gwcc: Diameter of the GWCC (0 if fast_mode).
        - pct_od_within_H: Percentage of pairs reachable within H hops (0 if fast_mode).
    """
    gwcc_nodes = gwcc(G)
    gscc_nodes = gscc(G)
    n = G.number_of_nodes()

    if fast_mode:
        # Approximate metrics for UI responsiveness on large graphs
        aspl = aspl_sampled_on_gwcc(G, samples=200)
        diam = diameter_two_sweep_on_gwcc(G, sweeps=4)
        pctH = percent_od_within_hops_sampled(G, H=H, samples=200)
    else:
        aspl, diam = aspl_and_diameter_on_gwcc(G)
        pctH = percent_od_within_hops(G, H=H)


    return {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "gwcc_n": len(gwcc_nodes),
        "gscc_n": len(gscc_nodes),
        "gwcc_frac": len(gwcc_nodes) / n if n else 0,
        "gscc_frac": len(gscc_nodes) / n if n else 0,
        "n_components": nx.number_weakly_connected_components(G),
        "aspl_gwcc": aspl,
        "diameter_gwcc": diam,
        "pct_od_within_H": pctH,
    }
