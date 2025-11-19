"""
Graph construction utilities for the airline network.

This module provides functions to build a NetworkX directed graph from airport
and route data, including the calculation of edge weights (distances).
"""
from __future__ import annotations
import math
import networkx as nx
import pandas as pd
from typing import Tuple, Optional

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points on Earth.

    Args:
        lat1: Latitude of the first point.
        lon1: Longitude of the first point.
        lat2: Latitude of the second point.
        lon2: Longitude of the second point.

    Returns:
        The distance in kilometers.
    """
    R = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def build_digraph(airports: pd.DataFrame, routes: pd.DataFrame, add_distance: bool = True) -> nx.DiGraph:
    """
    Builds a directed graph (DiGraph) from airport and route data.

    Nodes represent airports and edges represent routes. Node attributes include
    metadata like city, country, and coordinates. Edge attributes can include
    geographic distance.

    Args:
        airports: DataFrame containing airport data.
        routes: DataFrame containing route data.
        add_distance: If True, calculates and adds 'distance_km' as an edge attribute.

    Returns:
        A NetworkX DiGraph representing the airline network.
    """
    G = nx.DiGraph()

    # Add nodes with all available attributes from the airports DataFrame
    for _, row in airports.iterrows():
        G.add_node(row["iata"], **row.to_dict())

    # Add edges (routes)
    for _, row in routes.iterrows():
        u, v = row["source_iata"], row["dest_iata"]

        # Skip self-loops if any exist in the data
        if u == v:
            continue

        attrs = row.to_dict()

        # Calculate and add distance attribute if requested and coordinates are available
        if add_distance and u in G and v in G:
            a, b = G.nodes[u], G.nodes[v]
            # Ensure coordinates exist before calculation
            if "lat" in a and "lon" in a and "lat" in b and "lon" in b:
                attrs["distance_km"] = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])

        G.add_edge(u, v, **attrs)

    return G
