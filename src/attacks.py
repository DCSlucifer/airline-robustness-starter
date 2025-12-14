"""
Attack strategies for network robustness analysis.

This module implements various strategies to simulate attacks on a network,
including targeted node removal based on centrality metrics, random node failures,
edge removal based on betweenness, and geographic attacks.
"""
from __future__ import annotations
from typing import Any, List, Tuple, Dict, Iterable, Optional, Set
import warnings

__all__ = [
    "collective_influence_scores",
    "targeted_node_removal",
    "random_node_failures",
    "edge_betweenness_attack",
    "geographic_attack_radius",
    "community_bridge_attack",
]
import random
import networkx as nx
import pandas as pd
from .metrics import topological_report
from .geo import nodes_within_radius_km

def collective_influence_scores(G: nx.Graph, l: int = 2) -> Dict[Any, float]:
    """
    Calculates the Collective Influence (CI) score for each node in the graph.

    The CI score measures the influence of a node based on its degree and the
    degrees of nodes at a distance `l` from it. It is defined as:
    CI_l(i) = (k_i - 1) * sum_{j in boundary at distance l} (k_j - 1)

    Args:
        G: The input graph (will be converted to undirected for this calculation).
        l: The radius of the ball (distance) to consider for influence. Defaults to 2.

    Returns:
        A dictionary mapping node IDs to their CI scores.
    """
    # We operate on the undirected view of the graph because robustness metrics
    # like Giant Connected Component (GCC) are typically defined for undirected connectivity.
    U = G.to_undirected()
    deg = dict(U.degree())
    scores = {i: 0.0 for i in U.nodes()}

    for i in U.nodes():
        k_i = deg.get(i, 0)
        # Nodes with degree <= 1 have no influence in this metric (cannot bridge components)
        if k_i <= 1:
            scores[i] = 0.0
            continue

        # Perform a BFS to find the set of nodes exactly at distance `l`.
        # We maintain the current frontier and the set of all visited nodes to avoid cycles.
        frontier = {i}
        visited = {i}

        for _ in range(l):
            nxt = set()
            for u in frontier:
                # Expand to unvisited neighbors
                nxt.update(set(U.neighbors(u)) - visited)
            visited.update(nxt)
            frontier = nxt
            # If the frontier is empty before reaching distance l, stop early.
            if not frontier:
                break

        boundary = frontier

        # Calculate CI score: (degree - 1) * sum(degree - 1 for boundary nodes)
        # We use max(d-1, 0) to handle edge cases safely, though d>=1 is implied by traversal.
        scores[i] = (k_i - 1) * sum(max(deg.get(j, 0) - 1, 0) for j in boundary)

    return scores

def _rank_nodes(G: nx.DiGraph, metric: str = "degree", l_ci: int = 2) -> List[Any]:
    """
    Ranks nodes in the graph based on a specified centrality metric.

    Args:
        G: The input directed graph.
        metric: The metric to use for ranking ("degree", "betweenness", "pagerank", "CI").
        l_ci: The distance parameter for Collective Influence (only used if metric="CI").

    Returns:
        A list of node IDs sorted by their score in descending order.
    """
    if metric == "degree":
        scores = dict(G.degree())
    elif metric == "betweenness":
        # Use approximate betweenness for large graphs (>500 nodes)
        n_nodes = G.number_of_nodes()
        if n_nodes > 500:
            k_samples = min(200, n_nodes)
            scores = nx.betweenness_centrality(G, k=k_samples)
        else:
            scores = nx.betweenness_centrality(G)
    elif metric == "pagerank":
        scores = nx.pagerank(G, alpha=0.85)
    elif metric == "CI":
        scores = collective_influence_scores(G, l=l_ci)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Sort nodes by score descending
    return [n for n, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

def targeted_node_removal(
    G: nx.DiGraph,
    k: int,
    metric: str = "degree",
    adaptive: bool = True,
    l_ci: int = 2,
    fast_mode: bool = False,
    report_every_n: int = 1,
) -> Tuple[nx.DiGraph, List[Dict]]:
    """
    Simulates a targeted attack by removing `k` nodes based on a centrality metric.

    Args:
        G: The initial directed graph.
        k: The number of nodes to remove.
        metric: The centrality metric to target ("degree", "betweenness", "pagerank", "CI").
        adaptive: If True, recomputes centrality scores after each removal.
                  If False, computes scores once at the beginning.
        l_ci: Distance parameter for Collective Influence metric.
        fast_mode: If True, use lightweight topological reports (skip ASPL/diameter).
        report_every_n: Compute full report every N steps (1 = every step).

    Returns:
        A tuple containing:
            - The modified graph H after node removal.
            - A log (list of dicts) recording each step's removed node and topological report.
    """
    H = G.copy()
    log = []

    # Input validation
    if k <= 0:
        warnings.warn("k must be positive, returning unchanged graph", UserWarning)
        return H, log

    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        warnings.warn("Graph is empty, nothing to remove", UserWarning)
        return H, log

    if k > n_nodes:
        warnings.warn(f"k={k} exceeds node count ({n_nodes}), capping at {n_nodes}", UserWarning)
        k = n_nodes

    # Pre-calculate order if not adaptive
    initial_order = []
    if not adaptive:
        initial_order = _rank_nodes(G, metric=metric, l_ci=l_ci)

    for step in range(k):
        if adaptive:
            # Re-evaluate the most central node in the current graph state
            order = _rank_nodes(H, metric=metric, l_ci=l_ci)
        else:
            # Filter the pre-calculated order to skip nodes that might have been removed
            # (though in simple node removal, this is just the next in list,
            # checks are for safety if logic changes)
            order = [n for n in initial_order if n in H]

        if not order:
            break

        node = order[0]
        H.remove_node(node)

        # Only compute expensive report every N steps or on final step
        if (step + 1) % report_every_n == 0 or step == k - 1:
            log.append({
                "step": step + 1,
                "removed_node": node,
                "report": topological_report(H, fast_mode=fast_mode)
            })
        else:
            # Lightweight log entry without full report
            log.append({
                "step": step + 1,
                "removed_node": node,
                "report": None
            })

    return H, log

def random_node_failures(G: nx.DiGraph, k: int, R: int = 10, seed: int = 42) -> List[Dict]:
    """
    Simulates random node failures (errors) multiple times to estimate average impact.

    Args:
        G: The input graph.
        k: The number of nodes to remove in each simulation.
        R: The number of repetitions (simulations) to run.
        seed: Random seed for reproducibility.

    Returns:
        A list of dictionaries, each containing the report for a single simulation run.
    """
    # Input validation
    if k <= 0:
        warnings.warn("k must be positive, returning empty results", UserWarning)
        return []

    if R <= 0:
        warnings.warn("R must be positive, returning empty results", UserWarning)
        return []

    nodes = list(G.nodes())
    if not nodes:
        warnings.warn("Graph is empty, nothing to remove", UserWarning)
        return []

    if k > len(nodes):
        warnings.warn(f"k={k} exceeds node count ({len(nodes)}), capping at {len(nodes)}", UserWarning)
        k = len(nodes)

    random.seed(seed)
    reports = []

    for r in range(R):
        # Randomly sample k nodes to fail
        sample = random.sample(nodes, min(k, len(nodes)))
        H = G.copy()
        H.remove_nodes_from(sample)
        reports.append({
            "rep": r + 1,
            "removed_nodes": sample,
            "report": topological_report(H)
        })

    return reports

def edge_betweenness_attack(G: nx.DiGraph, m: int, adaptive: bool = True) -> Tuple[nx.DiGraph, List[Dict]]:
    """
    Simulates an attack targeting edges with the highest betweenness centrality.

    This attack often disconnects communities by removing "bridge" edges.

    Args:
        G: The input graph.
        m: The number of edges to remove.
        adaptive: If True, recomputes edge betweenness after each removal.
                  If False, computes scores once at the beginning (faster but less accurate).

    Returns:
        A tuple containing the modified graph and the attack log.
    """
    H = G.copy()
    log = []

    # Input validation
    if m <= 0:
        return H, log

    m = min(m, H.number_of_edges())  # Cap at available edges

    if not adaptive:
        # NON-ADAPTIVE MODE: Pre-compute edge betweenness ranking once
        eb = nx.edge_betweenness_centrality(G.to_undirected())
        if not eb:
            return H, log

        # Sort all edges by betweenness score descending
        ranked_edges = sorted(eb.items(), key=lambda kv: kv[1], reverse=True)

        for step, (edge, _score) in enumerate(ranked_edges[:m]):
            # Remove the edge from the directed graph (check both directions)
            if H.has_edge(*edge):
                H.remove_edge(*edge)
            elif H.has_edge(edge[1], edge[0]):
                H.remove_edge(edge[1], edge[0])
            else:
                # Edge already removed (can happen with undirected view)
                continue

            log.append({
                "step": step + 1,
                "removed_edge": edge,
                "report": topological_report(H)
            })
    else:
        # ADAPTIVE MODE: Recompute edge betweenness after each removal
        for step in range(m):
            eb = nx.edge_betweenness_centrality(H.to_undirected())

            if not eb:
                break

            # Find the edge with the maximum score
            emax = max(eb, key=eb.get)

            # Remove the edge from the directed graph (check both directions)
            if H.has_edge(*emax):
                H.remove_edge(*emax)
            elif H.has_edge(emax[1], emax[0]):
                H.remove_edge(emax[1], emax[0])

            log.append({
                "step": step + 1,
                "removed_edge": emax,
                "report": topological_report(H)
            })

    return H, log

def geographic_attack_radius(
    G: nx.DiGraph,
    center: Tuple[float, float],
    radius_km: float
) -> Tuple[nx.DiGraph, Dict]:
    """
    Simulates a localized failure where all nodes within a geographic radius are removed.

    Args:
        G: The input graph.
        center: A tuple of (latitude, longitude) for the center of the attack.
        radius_km: The radius of the attack in kilometers.

    Returns:
        A tuple containing the modified graph and a report dictionary.
    """
    victims = list(nodes_within_radius_km(G, center, radius_km))
    H = G.copy()
    H.remove_nodes_from(victims)
    return H, {"removed_nodes": victims, "report": topological_report(H)}

def community_bridge_attack(G: nx.DiGraph, m: int) -> Tuple[nx.DiGraph, Dict]:
    """
    Targets edges that connect different communities (bridges).

    This strategy first detects communities using Label Propagation, then identifies
    edges spanning across communities and removes those with the highest betweenness.

    Args:
        G: The input graph.
        m: The number of bridge edges to remove.

    Returns:
        A tuple containing the modified graph and a report dictionary.
    """
    # Use Label Propagation to detect communities on the undirected structure
    from networkx.algorithms.community import label_propagation_communities
    comms = list(label_propagation_communities(G.to_undirected()))

    # Map each node to its community ID
    comm_id = {}
    for cid, c in enumerate(comms):
        for n in c:
            comm_id[n] = cid

    # Identify edges that connect nodes from different communities
    inter_edges = [(u, v) for u, v in G.edges() if comm_id.get(u) != comm_id.get(v)]

    if not inter_edges:
        return G.copy(), {"removed_edges": [], "report": topological_report(G)}

    H = G.copy()

    # Rank these inter-community edges by their edge betweenness centrality.
    # We calculate centrality on the full undirected graph to capture global flow importance.
    eb = nx.edge_betweenness_centrality(G.to_undirected())

    # Sort inter-edges by their centrality score.
    # We check both (u,v) and (v,u) in the centrality dict since it's undirected.
    ranked = sorted(
        [e for e in inter_edges if e in eb or (e[::-1] in eb)],
        key=lambda e: eb.get(e, eb.get((e[1], e[0]), 0)),
        reverse=True
    )

    removed = ranked[:m]
    H.remove_edges_from(removed)

    return H, {"removed_edges": removed, "report": topological_report(H)}
