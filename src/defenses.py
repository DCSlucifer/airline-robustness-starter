"""
Defense strategies: edge addition heuristics, node hardening.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Optional
import itertools
import networkx as nx
import pandas as pd
from .metrics import topological_report
from .geo import haversine_km

def _candidate_pairs(G: nx.DiGraph, max_distance_km: float=3000, across_communities: bool=True, top_n_per_comm: int=10) -> List[Tuple[str,str]]:
    # Small candidate set: pick top-degree nodes; if across_communities, ensure endpoints are in different communities
    U = G.to_undirected()
    from networkx.algorithms.community import label_propagation_communities
    comms = list(label_propagation_communities(U))
    comm_id = {}
    for cid, c in enumerate(comms):
        for n in c:
            comm_id[n] = cid
    # top-N per community by degree
    candidates = []
    for cid, c in enumerate(comms):
        nodes = sorted(list(c), key=lambda n: U.degree(n), reverse=True)[:top_n_per_comm]
        candidates.extend(nodes)
    cand_set = set(candidates)
    # Form pairs not already connected
    pairs = []
    for u, v in itertools.combinations(cand_set, 2):
        if U.has_edge(u, v):
            continue
        if across_communities and comm_id.get(u) == comm_id.get(v):
            continue
        a, b = G.nodes[u], G.nodes[v]
        lat1, lon1, lat2, lon2 = a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon")
        if None in (lat1, lon1, lat2, lon2):
            continue
        if haversine_km(lat1, lon1, lat2, lon2) <= max_distance_km:
            pairs.append((u, v))
    return pairs

def greedy_edge_addition(G: nx.DiGraph, budget: int=5, max_distance_km: float=3000) -> Tuple[nx.DiGraph, List[Dict]]:
    """Greedy add up to 'budget' bidirectional edges (u->v and v->u) subject to distance cap, maximizing GWCC and ASPL improvements."""
    H = G.copy()
    log = []
    for b in range(budget):
        best_pair, best_score, best_report = None, float("-inf"), None
        for (u, v) in _candidate_pairs(H, max_distance_km=max_distance_km):
            # try adding u<->v
            H.add_edge(u, v)
            H.add_edge(v, u)
            rep = topological_report(H)
            score = rep["gwcc_frac"] - rep["aspl_gwcc"]*1e-3  # lexicographic-ish: prioritize GWCC, lightly prefer lower ASPL
            # revert
            H.remove_edge(u, v)
            H.remove_edge(v, u)
            if score > best_score:
                best_score, best_pair, best_report = score, (u, v), rep
        if best_pair is None:
            break
        # commit
        u, v = best_pair
        H.add_edge(u, v)
        H.add_edge(v, u)
        rep_after = topological_report(H)
        log.append({"step": b+1, "added_edges": [(u,v),(v,u)], "report_after": rep_after})
    return H, log

def node_hardening_list(G: nx.DiGraph, top_n: int=10, metric: str="betweenness") -> List[str]:
    if metric == "betweenness":
        scores = nx.betweenness_centrality(G)
    elif metric == "degree":
        scores = dict(G.degree())
    elif metric == "pagerank":
        scores = nx.pagerank(G, alpha=0.85)
    else:
        scores = dict(G.degree())
    return [n for n,_ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
