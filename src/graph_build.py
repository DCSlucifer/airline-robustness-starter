"""
Graph construction utilities for the airline network.

This module provides functions to build a NetworkX directed graph from airport
and route data, including the calculation of edge weights (distances).
"""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd

from .geo import haversine_km  # Import from geo.py to avoid DRY violation


def build_digraph(
    airports: pd.DataFrame, routes: pd.DataFrame, add_distance: bool = True
) -> nx.DiGraph:
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

    # Vectorized node addition (faster than iterrows)
    # Drop duplicate IATA codes (keep first occurrence) to avoid index error
    # This handles data quality issues where the same IATA appears multiple times
    deduped_airports = airports.drop_duplicates(subset="iata", keep="first")
    node_attrs = deduped_airports.set_index("iata").to_dict("index")
    for iata, attrs in node_attrs.items():
        try:
            lat = float(attrs["lat"])
            lon = float(attrs["lon"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid coordinates for airport {iata!r}") from exc
        if (
            not math.isfinite(lat)
            or not math.isfinite(lon)
            or not -90.0 <= lat <= 90.0
            or not -180.0 <= lon <= 180.0
        ):
            raise ValueError(f"Invalid coordinates for airport {iata!r}")
        attrs["lat"] = lat
        attrs["lon"] = lon
        G.add_node(iata, **attrs)

    # Use itertuples for edges (3x faster than iterrows)
    for row in routes.itertuples(index=False):
        u, v = row.source_iata, row.dest_iata

        # NetworkX creates missing endpoint nodes implicitly. Skip dangling routes
        # so every graph node retains validated airport metadata.
        if u not in G or v not in G:
            continue

        # Skip self-loops if any exist in the data
        if u == v:
            continue

        attrs = row._asdict()

        # Calculate and add distance attribute if requested and coordinates are available
        if add_distance:
            a, b = G.nodes[u], G.nodes[v]
            # Ensure coordinates exist before calculation
            if all(k in a and k in b for k in ("lat", "lon")):
                attrs["distance_km"] = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])

        G.add_edge(u, v, **attrs)

    return G
