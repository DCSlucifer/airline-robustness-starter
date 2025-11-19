"""
Graph construction helpers.
"""
from __future__ import annotations
import math
import networkx as nx
import pandas as pd
from typing import Tuple, Optional

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def build_digraph(airports: pd.DataFrame, routes: pd.DataFrame, add_distance: bool = True) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in airports.iterrows():
        G.add_node(row["iata"], **row.to_dict())
    for _, row in routes.iterrows():
        u, v = row["source_iata"], row["dest_iata"]
        if u == v:
            continue
        attrs = row.to_dict()
        if add_distance and u in G and v in G:
            a, b = G.nodes[u], G.nodes[v]
            attrs["distance_km"] = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        G.add_edge(u, v, **attrs)
    return G
