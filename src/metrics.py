"""
Robustness metrics.
"""
from __future__ import annotations
import networkx as nx
import pandas as pd
from typing import Dict, Tuple

def gwcc(G: nx.DiGraph):
    if G.number_of_nodes() == 0:
        return set()
    # Weakly connected components on DiGraph
    comps = list(nx.weakly_connected_components(G))
    return max(comps, key=len) if comps else set()

def gscc(G: nx.DiGraph):
    if G.number_of_nodes() == 0:
        return set()
    comps = list(nx.strongly_connected_components(G))
    return max(comps, key=len) if comps else set()

def aspl_and_diameter_on_gwcc(G: nx.DiGraph) -> Tuple[float, int]:
    W = G.subgraph(gwcc(G)).to_undirected()
    if W.number_of_nodes() <= 1:
        return float("inf"), 0
    try:
        aspl = nx.average_shortest_path_length(W)
    except nx.NetworkXError:
        aspl = float("inf")
    try:
        diameter = nx.diameter(W)
    except nx.NetworkXError:
        diameter = 0
    return aspl, diameter

def percent_od_within_hops(G: nx.DiGraph, H: int=4) -> float:
    W = G.subgraph(gwcc(G)).to_undirected()
    n = W.number_of_nodes()
    if n <= 1:
        return 0.0
    # Compute all-pairs shortest paths within H cutoff
    count = 0
    total = n*(n-1)
    for s in W.nodes():
        sp = nx.single_source_shortest_path_length(W, s, cutoff=H)
        # exclude self (distance 0)
        count += sum(1 for t,d in sp.items() if t != s and d <= H)
    return count / total

def topological_report(G: nx.DiGraph, H: int=4) -> Dict:
    n = G.number_of_nodes()
    gw = gwcc(G)
    gs = gscc(G)
    aspl, diam = aspl_and_diameter_on_gwcc(G)
    pctH = percent_od_within_hops(G, H=H)
    return {
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "gwcc_frac": len(gw)/n if n else 0,
        "gscc_frac": len(gs)/n if n else 0,
        "n_components": nx.number_weakly_connected_components(G),
        "aspl_gwcc": aspl,
        "diameter_gwcc": diam,
        "pct_od_within_H": pctH,
    }
