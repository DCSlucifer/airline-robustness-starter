"""
Data loading utilities for OpenFlights-like CSV datasets.

This module provides functions to load and validate airport and route data from CSV files.
It ensures that the necessary columns are present and filters routes to ensure consistency
with the available airport data.

Expected CSV schemas:
- airports.csv: Must contain [airport_id, name, city, country, iata, icao, lat, lon]
- routes.csv: Must contain [source_iata, dest_iata] representing directed edges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MISSING_IATA_MARKERS = frozenset({"", r"\N", "NULL", "NONE"})


def _normalize_iata(series: pd.Series, column: str) -> tuple[pd.Series, pd.Series]:
    """Normalize IATA values and return them with a mask for explicit missing markers."""
    normalized = series.astype("string").fillna("").str.strip().str.upper()
    missing = normalized.isin(_MISSING_IATA_MARKERS)
    # OpenFlights contains a small number of three-character alphanumeric
    # location identifiers (for example ``DU9``), so validation must not be
    # narrower than the source data's actual identifier domain.
    invalid = ~missing & ~normalized.str.fullmatch(r"[A-Z0-9]{3}", na=False)
    if invalid.any():
        examples = sorted(normalized.loc[invalid].unique().tolist())[:5]
        raise ValueError(f"Invalid IATA code(s) in '{column}': {examples}")
    return normalized, missing


def _validate_coordinates(df: pd.DataFrame) -> None:
    """Convert airport coordinates to floats and reject non-finite or out-of-range values."""
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    invalid = (
        ~np.isfinite(df["lat"])
        | ~np.isfinite(df["lon"])
        | ~df["lat"].between(-90.0, 90.0)
        | ~df["lon"].between(-180.0, 180.0)
    )
    if invalid.any():
        examples = df.loc[invalid, "iata"].astype(str).head(5).tolist()
        raise ValueError(f"Airports CSV contains invalid coordinates for IATA code(s): {examples}")


def load_airports(path: str) -> pd.DataFrame:
    """
    Loads airport data from a CSV file.

    Args:
        path: The file path to the airports CSV.

    Returns:
        A pandas DataFrame containing airport information.

    Raises:
        ValueError: If required columns are missing from the CSV.
    """
    # Preserve valid codes such as IATA ``NAN`` instead of letting pandas treat
    # them as generic missing-value tokens. Missing markers are handled explicitly below.
    df = pd.read_csv(path, keep_default_na=False)

    # Define the set of required columns for downstream processing
    needed = {"airport_id", "name", "city", "country", "iata", "icao", "lat", "lon"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Airports CSV missing columns: {missing}")

    df["iata"], missing_iata = _normalize_iata(df["iata"], "iata")
    df = df.loc[~missing_iata].copy()
    _validate_coordinates(df)
    return df.reset_index(drop=True)


def load_routes(path: str) -> pd.DataFrame:
    """
    Loads route data from a CSV file.

    Args:
        path: The file path to the routes CSV.

    Returns:
        A pandas DataFrame containing route information (edges).

    Raises:
        ValueError: If required columns are missing from the CSV.
    """
    df = pd.read_csv(path, keep_default_na=False)

    # Define required columns for edge construction
    needed = {"source_iata", "dest_iata"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Routes CSV missing columns: {missing}")

    df["source_iata"], missing_source = _normalize_iata(df["source_iata"], "source_iata")
    df["dest_iata"], missing_dest = _normalize_iata(df["dest_iata"], "dest_iata")
    df = df.loc[~(missing_source | missing_dest)].copy()
    return df.reset_index(drop=True)


def merge_airports_routes(
    airports: pd.DataFrame, routes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters routes to ensure both endpoints exist in the airports dataset.

    This step is crucial to avoid 'dangling edges' in the graph where a route
    points to a non-existent airport.

    Args:
        airports: DataFrame of airports.
        routes: DataFrame of routes.

    Returns:
        A tuple containing:
            - A copy of the airports DataFrame.
            - A filtered copy of the routes DataFrame.
    """
    # Create a set of valid IATA codes for O(1) lookup
    valid_iata = set(airports["iata"].unique())

    # Filter routes: keep only those where both source and destination are in the valid set
    mask = routes["source_iata"].isin(valid_iata) & routes["dest_iata"].isin(valid_iata)

    return airports.copy(), routes.loc[mask].copy()
