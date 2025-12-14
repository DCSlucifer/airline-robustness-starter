"""
Pytest configuration and shared fixtures for airline robustness tests.

This module provides reusable test fixtures for building graphs of various
sizes and configurations to be used across test modules.
"""
import pytest
import networkx as nx
import pandas as pd


@pytest.fixture
def sample_airports():
    """Returns a DataFrame with 5 sample airports."""
    return pd.DataFrame([
        {"airport_id": 1, "name": "Alpha", "city": "A-City", "country": "Land", "iata": "AAA", "icao": "AAAA", "lat": 0.0, "lon": 0.0},
        {"airport_id": 2, "name": "Beta", "city": "B-City", "country": "Land", "iata": "BBB", "icao": "BBBB", "lat": 1.0, "lon": 0.0},
        {"airport_id": 3, "name": "Gamma", "city": "C-City", "country": "Land", "iata": "CCC", "icao": "CCCC", "lat": 0.0, "lon": 1.0},
        {"airport_id": 4, "name": "Delta", "city": "D-City", "country": "Land", "iata": "DDD", "icao": "DDDD", "lat": 1.0, "lon": 1.0},
        {"airport_id": 5, "name": "Epsilon", "city": "E-City", "country": "Land", "iata": "EEE", "icao": "EEEE", "lat": 2.0, "lon": 2.0},
    ])


@pytest.fixture
def sample_routes():
    """Returns a DataFrame with sample routes forming a connected graph."""
    return pd.DataFrame([
        {"source_iata": "AAA", "dest_iata": "BBB"},
        {"source_iata": "BBB", "dest_iata": "CCC"},
        {"source_iata": "CCC", "dest_iata": "DDD"},
        {"source_iata": "DDD", "dest_iata": "EEE"},
        {"source_iata": "AAA", "dest_iata": "DDD"},  # Shortcut edge
    ])


@pytest.fixture
def simple_digraph(sample_airports, sample_routes):
    """Builds a simple DiGraph from sample data."""
    from src.graph_build import build_digraph
    return build_digraph(sample_airports, sample_routes, add_distance=True)


@pytest.fixture
def empty_digraph():
    """Returns an empty directed graph."""
    return nx.DiGraph()


@pytest.fixture
def single_node_digraph():
    """Returns a DiGraph with a single node."""
    G = nx.DiGraph()
    G.add_node("AAA", lat=0.0, lon=0.0, name="Alpha")
    return G


@pytest.fixture
def disconnected_digraph():
    """Returns a DiGraph with two disconnected components."""
    G = nx.DiGraph()
    # Component 1
    G.add_node("AAA", lat=0.0, lon=0.0, name="Alpha")
    G.add_node("BBB", lat=1.0, lon=0.0, name="Beta")
    G.add_edge("AAA", "BBB")
    # Component 2 (isolated)
    G.add_node("CCC", lat=10.0, lon=10.0, name="Gamma")
    G.add_node("DDD", lat=11.0, lon=10.0, name="Delta")
    G.add_edge("CCC", "DDD")
    return G
