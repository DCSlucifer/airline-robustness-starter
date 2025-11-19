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
import pandas as pd
from typing import Tuple

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
    df = pd.read_csv(path)

    # Define the set of required columns for downstream processing
    needed = {"airport_id", "name", "city", "country", "iata", "icao", "lat", "lon"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Airports CSV missing columns: {missing}")

    # Ensure IATA codes are treated as strings (e.g., to avoid issues with 'NaN' or numeric-like codes)
    df["iata"] = df["iata"].astype(str)
    return df

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
    df = pd.read_csv(path)

    # Define required columns for edge construction
    needed = {"source_iata", "dest_iata"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Routes CSV missing columns: {missing}")

    # Ensure source and destination IATA codes are strings
    df["source_iata"] = df["source_iata"].astype(str)
    df["dest_iata"] = df["dest_iata"].astype(str)
    return df

def merge_airports_routes(airports: pd.DataFrame, routes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
