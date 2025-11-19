"""
Geographic utility functions for spatial analysis.

This module provides tools for calculating distances and identifying nodes within
specific geographic regions using the Haversine formula.
"""
from __future__ import annotations
import math
from typing import Iterable, Tuple, Dict, Generator
import networkx as nx

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

def nodes_within_radius_km(
    G: nx.Graph,
    center: Tuple[float, float],
    radius_km: float
) -> Generator[str, None, None]:
    """
    Identifies nodes within a specified geographic radius from a center point.

    Args:
        G: The input graph containing nodes with 'lat' and 'lon' attributes.
        center: A tuple (latitude, longitude) representing the center point.
        radius_km: The radius in kilometers.

    Yields:
        The IDs of nodes that fall within the specified radius.
    """
    lat0, lon0 = center

    for n, data in G.nodes(data=True):
        lat, lon = data.get("lat"), data.get("lon")

        # Skip nodes with missing geographic data
        if lat is None or lon is None:
            continue

        if haversine_km(lat0, lon0, lat, lon) <= radius_km:
            yield n
