"""
Attack strategies (node/edge removal).
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Optional
import random
import networkx as nx
import pandas as pd
from .metrics import topological_report
from .geo import nodes_within_radius_km

def collective_influence_scores(G: nx.Graph, l: int=2) -> Dict:
    # Simple CI_l(i) = (k_i - 1) * sum_{j in boundary at distance l} (k_j - 1)
    # on undirected view for robustness targeting
    U = G.to_undirected()
    deg = dict(U.degree())
    scores = {i: 0.0 for i in U.nodes()}
    for i in U.nodes():
        k_i = deg.get(i, 0)
        if k_i <= 1:
            scores[i] = 0.0
            continue
        # BFS frontier at exact distance l
        frontier = {i}
        visited = {i}
        for _ in range(l):
            nxt = set()
            for u in frontier:
                nxt.update(set(U.neighbors(u)) - visited)
            visited.update(nxt)
            frontier = nxt
            if not frontier:
                break
        boundary = frontier
        scores[i] = (k_i - 1) * sum(max(deg.get(j,0)-1, 0) for j in boundary)
    return scores

def _rank_nodes(G: nx.DiGraph, metric: str="degree", l_ci: int=2) -> List[str]:
    if metric == "degree":
        scores = dict(G.degree())
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(G)
    elif metric == "pagerank":
        scores = nx.pagerank(G, alpha=0.85)
    elif metric == "CI":
        scores = collective_influence_scores(G, l=l_ci)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return [n for n,_ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

def targeted_node_removal(G: nx.DiGraph, k: int, metric: str="degree", adaptive: bool=True, l_ci: int=2) -> Tuple[nx.DiGraph, List[Dict]]:
    """Remove k nodes by a chosen metric; adaptive recomputes after each removal."""
    H = G.copy()
    log = []
    for step in range(k):
        order = _rank_nodes(H, metric=metric, l_ci=l_ci)
        if not order:
            break
        node = order[0]
        H.remove_node(node)
        log.append({"step": step+1, "removed_node": node, "report": topological_report(H)})
        if not adaptive:
            # If not adaptive, remove the next top-k from the starting scores
            if step == 0:
                initial_order = _rank_nodes(G, metric=metric, l_ci=l_ci)
            # Remove next according to the initial order
            if step+1 < k:
                for nxt in initial_order:
                    if nxt in H:
                        node = nxt
                        H.remove_node(node)
                        log.append({"step": step+2, "removed_node": node, "report": topological_report(H)})
                        break
                break
    return H, log

def random_node_failures(G: nx.DiGraph, k: int, R: int=10, seed: int=42) -> List[Dict]:
    random.seed(seed)
    nodes = list(G.nodes())
    reports = []
    for r in range(R):
        sample = random.sample(nodes, min(k, len(nodes)))
        H = G.copy()
        H.remove_nodes_from(sample)
        reports.append({"rep": r+1, "removed_nodes": sample, "report": topological_report(H)})
    return reports

def edge_betweenness_attack(G: nx.DiGraph, m: int, adaptive: bool=True) -> Tuple[nx.DiGraph, List[Dict]]:
    H = G.copy()
    log = []
    for step in range(m):
        # compute edge betweenness on undirected view for robustness
        eb = nx.edge_betweenness_centrality(H.to_undirected())
        if not eb:
            break
        emax = max(eb, key=eb.get)
        if H.has_edge(*emax):
            H.remove_edge(*emax)
        elif H.has_edge(emax[1], emax[0]):
            H.remove_edge(emax[1], emax[0])
        log.append({"step": step+1, "removed_edge": emax, "report": topological_report(H)})
        if not adaptive:
            break
    return H, log

def geographic_attack_radius(G: nx.DiGraph, center: Tuple[float,float], radius_km: float) -> Tuple[nx.DiGraph, Dict]:
    victims = list(nodes_within_radius_km(G, center, radius_km))
    H = G.copy()
    H.remove_nodes_from(victims)
    return H, {"removed_nodes": victims, "report": topological_report(H)}

def community_bridge_attack(G: nx.DiGraph, m: int) -> Tuple[nx.DiGraph, Dict]:
    # Label propagation communities, then remove top-m inter-community edges by betweenness
    from networkx.algorithms.community import label_propagation_communities
    comms = list(label_propagation_communities(G.to_undirected()))
    comm_id = {}
    for cid, c in enumerate(comms):
        for n in c:
            comm_id[n] = cid
    inter_edges = [(u,v) for u,v in G.edges() if comm_id.get(u) != comm_id.get(v)]
    if not inter_edges:
        return G.copy(), {"removed_edges": [], "report": topological_report(G)}
    H = G.copy()
    # Rank inter-community edges by edge betweenness on undirected view
    eb = nx.edge_betweenness_centrality(G.to_undirected())
    ranked = sorted([e for e in inter_edges if e in eb or (e[::-1] in eb)], key=lambda e: eb.get(e, eb.get((e[1],e[0]),0)), reverse=True)
    removed = ranked[:m]
    H.remove_edges_from(removed)
    return H, {"removed_edges": removed, "report": topological_report(H)}
