"""
Unit tests for geographic utilities.

Tests cover Haversine distance calculations and radius queries.
"""
import pytest
import networkx as nx
from src.geo import haversine_km, nodes_within_radius_km


class TestHaversineKm:
    """Tests for haversine_km function."""

    def test_same_point_returns_zero(self):
        """Distance from point to itself should be 0."""
        assert haversine_km(0, 0, 0, 0) == 0.0

    def test_known_distance_london_paris(self):
        """Known distance: London to Paris ~344 km."""
        # London: 51.5074, -0.1278
        # Paris: 48.8566, 2.3522
        dist = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
        assert 340 < dist < 350  # Allow small variance

    def test_antipodal_points(self):
        """Antipodal points should be ~20,000 km apart (half Earth circumference)."""
        dist = haversine_km(0, 0, 0, 180)
        assert 20000 < dist < 20100

    def test_symmetric(self):
        """Distance should be symmetric: d(A,B) == d(B,A)."""
        d1 = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)  # NYC to LA
        d2 = haversine_km(34.0522, -118.2437, 40.7128, -74.0060)  # LA to NYC
        assert abs(d1 - d2) < 0.01


class TestNodesWithinRadius:
    """Tests for nodes_within_radius_km function."""

    def test_finds_nearby_nodes(self):
        """Should find nodes within the specified radius."""
        G = nx.DiGraph()
        G.add_node("A", lat=0.0, lon=0.0)
        G.add_node("B", lat=0.01, lon=0.01)  # ~1.5 km away
        G.add_node("C", lat=10.0, lon=10.0)  # ~1500 km away

        nearby = list(nodes_within_radius_km(G, (0, 0), radius_km=10))
        assert "A" in nearby
        assert "B" in nearby
        assert "C" not in nearby

    def test_skips_nodes_without_coords(self):
        """Nodes without lat/lon should be skipped without error."""
        G = nx.DiGraph()
        G.add_node("A", lat=0.0, lon=0.0)
        G.add_node("B")  # No coords
        G.add_node("C", lat=None, lon=None)  # Explicit None

        nearby = list(nodes_within_radius_km(G, (0, 0), radius_km=1000))
        assert "A" in nearby
        assert "B" not in nearby
        assert "C" not in nearby

    def test_empty_graph(self):
        """Should return empty for empty graph."""
        G = nx.DiGraph()
        nearby = list(nodes_within_radius_km(G, (0, 0), radius_km=1000))
        assert nearby == []
