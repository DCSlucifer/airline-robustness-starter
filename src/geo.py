"""
Geographic helpers.
"""
from __future__ import annotations
import math
from typing import Iterable, Tuple, Dict
import networkx as nx

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def nodes_within_radius_km(G: nx.Graph, center: Tuple[float,float], radius_km: float):
    lat0, lon0 = center
    for n, data in G.nodes(data=True):
        lat, lon = data.get("lat"), data.get("lon")
        if lat is None or lon is None:
            continue
        if haversine_km(lat0, lon0, lat, lon) <= radius_km:
            yield n
