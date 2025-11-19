"""
Centrality computations.
"""
from __future__ import annotations
import networkx as nx
import pandas as pd
from typing import Dict

def node_centralities(G: nx.DiGraph) -> pd.DataFrame:
    # degree
    deg_in = dict(G.in_degree())
    deg_out = dict(G.out_degree())
    # betweenness on directed graph; may be heavy
    btw = nx.betweenness_centrality(G, normalized=True)
    pr = nx.pagerank(G, alpha=0.85)
    df = pd.DataFrame({
        "node": list(G.nodes()),
        "deg_in": [deg_in.get(n,0) for n in G.nodes()],
        "deg_out": [deg_out.get(n,0) for n in G.nodes()],
        "betweenness": [btw.get(n,0.0) for n in G.nodes()],
        "pagerank": [pr.get(n,0.0) for n in G.nodes()],
    })
    df["deg_total"] = df["deg_in"] + df["deg_out"]
    return df.sort_values("deg_total", ascending=False).reset_index(drop=True)
