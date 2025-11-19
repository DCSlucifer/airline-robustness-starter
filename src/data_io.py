"""
Data loading utilities for OpenFlights-like CSVs.
Expected columns:
- airports: airport_id, name, city, country, iata, icao, lat, lon
- routes: source_iata, dest_iata  (directed edge)
"""
from __future__ import annotations
import pandas as pd
from typing import Tuple

def load_airports(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"airport_id","name","city","country","iata","icao","lat","lon"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Airports CSV missing columns: {missing}")
    df["iata"] = df["iata"].astype(str)
    return df

def load_routes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"source_iata","dest_iata"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Routes CSV missing columns: {missing}")
    df["source_iata"] = df["source_iata"].astype(str)
    df["dest_iata"] = df["dest_iata"].astype(str)
    return df

def merge_airports_routes(airports: pd.DataFrame, routes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Keep only routes whose endpoints exist in airports
    IATA = set(airports["iata"].unique())
    mask = routes["source_iata"].isin(IATA) & routes["dest_iata"].isin(IATA)
    return airports.copy(), routes.loc[mask].copy()
