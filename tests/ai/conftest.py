"""Shared fixtures for AI assistant tests."""
import networkx as nx
import pytest


@pytest.fixture
def small_graph() -> nx.DiGraph:
    """A tiny airline-like directed graph with coordinates."""
    G = nx.DiGraph()
    coords = {
        "AAA": (40.0, -74.0),
        "BBB": (41.0, -73.0),
        "CCC": (34.0, -118.0),
        "DDD": (51.5, -0.1),
        "EEE": (48.8, 2.3),
        "FFF": (35.7, 139.7),
    }
    for iata, (lat, lon) in coords.items():
        G.add_node(iata, lat=lat, lon=lon, name=iata)
    edges = [
        ("AAA", "BBB"), ("BBB", "AAA"), ("AAA", "CCC"), ("CCC", "AAA"),
        ("CCC", "FFF"), ("FFF", "CCC"), ("DDD", "EEE"), ("EEE", "DDD"),
        ("BBB", "DDD"), ("DDD", "BBB"), ("AAA", "DDD"), ("DDD", "AAA"),
    ]
    G.add_edges_from(edges)
    return G
