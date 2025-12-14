"""
Defense strategies for enhancing network robustness.

This module implements strategies to improve network resilience, such as adding
strategic edges (redundancy) to connect communities or hardening critical nodes.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Optional
import warnings
import itertools
import networkx as nx
import pandas as pd

__all__ = [
    "greedy_edge_addition",
    "node_hardening_list",
]

from .metrics import topological_report
from .geo import haversine_km

def _candidate_pairs(
    G: nx.DiGraph,
    max_distance_km: float = 3000,
    across_communities: bool = True,
    top_n_per_comm: int = 10
) -> List[Tuple[str, str]]:
    """
    Generates a list of candidate node pairs for edge addition.

    To avoid checking all O(N^2) pairs, this heuristic focuses on connecting
    high-degree nodes from different communities, which is often effective for
    improving global connectivity.

    Args:
        G: The input directed graph.
        max_distance_km: The maximum geographic distance allowed between nodes.
        across_communities: If True, only considers pairs where nodes are in different communities.
        top_n_per_comm: The number of top-degree nodes to consider from each community.

    Returns:
        A list of tuples representing candidate node pairs (u, v).
    """
    # Use the undirected view for community detection as it captures structural clusters better.
    U = G.to_undirected()
    from networkx.algorithms.community import label_propagation_communities
    comms = list(label_propagation_communities(U))

    # Map each node to its community ID for quick lookup
    comm_id = {}
    for cid, c in enumerate(comms):
        for n in c:
            comm_id[n] = cid

    # Select top-N nodes by degree from each community to form a candidate pool.
    # High-degree nodes are good candidates for "hubs" to connect communities.
    candidates = []
    for cid, c in enumerate(comms):
        nodes = sorted(list(c), key=lambda n: U.degree(n), reverse=True)[:top_n_per_comm]
        candidates.extend(nodes)

    cand_set = set(candidates)

    # Form pairs from the candidate pool that are not already connected.
    pairs = []
    for u, v in itertools.combinations(cand_set, 2):
        if U.has_edge(u, v):
            continue

        # Filter for cross-community edges if requested
        if across_communities and comm_id.get(u) == comm_id.get(v):
            continue

        # Check geographic constraints
        a, b = G.nodes[u], G.nodes[v]
        lat1, lon1 = a.get("lat"), a.get("lon")
        lat2, lon2 = b.get("lat"), b.get("lon")

        # Skip if location data is missing
        if None in (lat1, lon1, lat2, lon2):
            continue

        if haversine_km(lat1, lon1, lat2, lon2) <= max_distance_km:
            pairs.append((u, v))

    return pairs

def greedy_edge_addition(
    G: nx.DiGraph,
    budget: int = 5,
    max_distance_km: float = 3000
) -> Tuple[nx.DiGraph, List[Dict]]:
    """
    Greedily adds edges to the graph to maximize robustness metrics.

    The algorithm iteratively adds the 'best' edge from a candidate set.
    Optimization: Uses a two-stage 'funnel' approach for speed:
      1. Fast Filter: Score all candidates by simple heuristics (degree sum).
      2. Accurate Check: Run expensive ASPL checks only on the top-k candidates.

    Args:
        G: The input directed graph.
        budget: The number of edges (bidirectional) to add.
        max_distance_km: The maximum allowed length for a new edge.

    Returns:
        A tuple containing the modified graph and a log of the additions.
    """
    H = G.copy()
    log = []

    # Input validation
    if budget <= 0:
        warnings.warn("budget must be positive, returning unchanged graph", UserWarning)
        return H, log

    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be positive")

    if G.number_of_nodes() < 2:
        warnings.warn("Graph has fewer than 2 nodes, cannot add edges", UserWarning)
        return H, log

    for b in range(budget):
        # --- STAGE 1: Fast Filter ---
        # Generate candidates
        candidates = _candidate_pairs(H, max_distance_km=max_distance_km)

        # Heuristic scoring: Prefer connecting high-degree nodes (hubs)
        # We use the undirected view for degree to capture overall importance
        U = H.to_undirected()
        scored_candidates = []

        for u, v in candidates:
            # Simple heuristic score: Sum of degrees
            # Logic: Connecting two big hubs is likely to improve global flow
            score = U.degree(u) + U.degree(v)
            scored_candidates.append(((u, v), score))

        # Sort by score descending and take top 5
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [pair for pair, score in scored_candidates[:5]]

        # --- STAGE 2: Accurate Check ---
        best_pair = None
        best_score = float("-inf")
        best_report = None

        # If no candidates found (rare), stop
        if not top_candidates:
            break

        for (u, v) in top_candidates:
            # Temporarily add bidirectional edge u <-> v
            H.add_edge(u, v)
            H.add_edge(v, u)

            rep = topological_report(H)

            # Scoring function:
            # Primary objective: Maximize GWCC fraction (connectivity).
            # Secondary objective: Minimize ASPL (efficiency).
            # We use a weighted sum where ASPL impact is scaled down to act as a tie-breaker.
            # Subtracting ASPL because we want to minimize it.
            score = rep["gwcc_frac"] - rep["aspl_gwcc"] * 1e-3

            # Revert changes
            H.remove_edge(u, v)
            H.remove_edge(v, u)

            if score > best_score:
                best_score = score
                best_pair = (u, v)
                best_report = rep

        if best_pair is None:
            break

        # Permanently add the best edge found in this iteration
        u, v = best_pair
        H.add_edge(u, v)
        H.add_edge(v, u)

        # Ensure we have a report for the final state if it wasn't the last checked
        if best_report is None:
             best_report = topological_report(H)

        log.append({
            "step": b + 1,
            "added_edges": [(u, v), (v, u)],
            "report_after": best_report
        })

    return H, log

def node_hardening_list(G: nx.DiGraph, top_n: int = 10, metric: str = "betweenness") -> List[str]:
    """
    Identifies a list of critical nodes to 'harden' (protect) against attacks.

    Hardening implies these nodes would be immune to random failures or targeted attacks
    in a simulation that supports such a mechanism.

    Args:
        G: The input graph.
        top_n: The number of nodes to identify.
        metric: The centrality metric used to identify critical nodes.

    Returns:
        A list of node IDs sorted by criticality.
    """
    if metric == "betweenness":
        scores = nx.betweenness_centrality(G)
    elif metric == "degree":
        scores = dict(G.degree())
    elif metric == "pagerank":
        scores = nx.pagerank(G, alpha=0.85)
    else:
        raise ValueError(
            f"Unknown metric: '{metric}'. Valid options: 'betweenness', 'degree', 'pagerank'"
        )

    return [n for n, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
