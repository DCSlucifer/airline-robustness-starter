"""Regression tests for data normalization and graph-boundary validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_io import load_airports, load_routes
from src.graph_build import build_digraph


def _airport(iata: str, *, lat: object = 1.0, lon: object = 2.0) -> dict[str, object]:
    return {
        "airport_id": 1,
        "name": f"Airport {iata}",
        "city": "City",
        "country": "Country",
        "iata": iata,
        "icao": "TEST",
        "lat": lat,
        "lon": lon,
    }


def test_load_airports_filters_missing_marker_and_preserves_nan_iata(tmp_path):
    path = tmp_path / "airports.csv"
    pd.DataFrame(
        [
            _airport(r"\N", lat=float("inf")),
            _airport(""),
            _airport("   "),
            _airport("NULL"),
            _airport("none"),
            _airport("NAN"),
            _airport("du9"),
        ]
    ).to_csv(path, index=False)

    airports = load_airports(path)

    assert airports["iata"].tolist() == ["NAN", "DU9"]
    assert airports["lat"].tolist() == [1.0, 1.0]
    assert airports["lon"].tolist() == [2.0, 2.0]


def test_load_routes_normalizes_codes_and_filters_missing_endpoints(tmp_path):
    path = tmp_path / "routes.csv"
    pd.DataFrame(
        [
            {"source_iata": " aaa ", "dest_iata": "NAN"},
            {"source_iata": r"\N", "dest_iata": "AAA"},
            {"source_iata": "AAA", "dest_iata": ""},
        ]
    ).to_csv(path, index=False)

    routes = load_routes(path)

    assert routes[["source_iata", "dest_iata"]].to_dict("records") == [
        {"source_iata": "AAA", "dest_iata": "NAN"}
    ]


def test_load_routes_rejects_non_missing_invalid_iata(tmp_path):
    path = tmp_path / "routes.csv"
    pd.DataFrame([{"source_iata": "AA", "dest_iata": "BBB"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Invalid IATA code"):
        load_routes(path)


@pytest.mark.parametrize("iata", ["AA", "A-1", "TOOLONG"])
def test_load_airports_rejects_non_missing_invalid_iata(tmp_path, iata):
    path = tmp_path / "airports.csv"
    pd.DataFrame([_airport(iata)]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Invalid IATA code"):
        load_airports(path)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        ("not-a-number", 0.0),
        (91.0, 0.0),
        (-91.0, 0.0),
        (0.0, -181.0),
        (0.0, 181.0),
    ],
)
def test_load_airports_rejects_invalid_coordinates(tmp_path, lat, lon):
    path = tmp_path / "airports.csv"
    pd.DataFrame([_airport("AAA", lat=lat, lon=lon)]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid coordinates"):
        load_airports(path)


def test_load_airports_accepts_coordinate_boundaries(tmp_path):
    path = tmp_path / "airports.csv"
    pd.DataFrame(
        [
            _airport("AAA", lat="-90", lon="-180"),
            _airport("BBB", lat="90", lon="180"),
        ]
    ).to_csv(path, index=False)

    airports = load_airports(path)

    assert airports[["lat", "lon"]].to_dict("records") == [
        {"lat": -90.0, "lon": -180.0},
        {"lat": 90.0, "lon": 180.0},
    ]


def test_build_digraph_skips_routes_with_unknown_endpoints(sample_airports):
    routes = pd.DataFrame(
        [
            {"source_iata": "AAA", "dest_iata": "BBB"},
            {"source_iata": "AAA", "dest_iata": "ZZZ"},
            {"source_iata": "ZZZ", "dest_iata": "AAA"},
            {"source_iata": "ZZZ", "dest_iata": "YYY"},
            {"source_iata": "AAA", "dest_iata": "AAA"},
        ]
    )

    graph = build_digraph(sample_airports, routes)

    assert graph.has_edge("AAA", "BBB")
    assert "ZZZ" not in graph
    assert "YYY" not in graph
    assert graph.number_of_edges() == 1


def test_build_digraph_normalizes_finite_coordinates(sample_airports, sample_routes):
    airports = sample_airports.copy()
    airports["lat"] = airports["lat"].astype(str)
    airports["lon"] = airports["lon"].astype(str)

    graph = build_digraph(airports, sample_routes)

    assert isinstance(graph.nodes["AAA"]["lat"], float)
    assert isinstance(graph.nodes["AAA"]["lon"], float)
    assert all(pd.notna(data["distance_km"]) for _, _, data in graph.edges(data=True))


def test_build_digraph_rejects_nonfinite_coordinates(sample_airports, sample_routes):
    airports = sample_airports.copy()
    airports.loc[airports["iata"] == "AAA", "lat"] = float("nan")

    with pytest.raises(ValueError, match="Invalid coordinates for airport 'AAA'"):
        build_digraph(airports, sample_routes)
