"""
Sanity check script for data loading and graph construction.

This script verifies that the configuration file can be read, data can be loaded
from the specified CSVs, and a graph can be successfully built. It prints summary
statistics to the console.
"""
import yaml
import pandas as pd
from src.data_io import load_airports, load_routes, merge_airports_routes
from src.graph_build import build_digraph

def main():
    """
    Main execution function for the sanity check.
    """
    # Load configuration
    try:
        with open("config/default.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        print("Configuration loaded successfully:")
        print(cfg)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Load raw data
    print("\nLoading data...")
    try:
        A = load_airports(cfg["airports_csv"])
        R = load_routes(cfg["routes_csv"])
        print(f"Airports loaded: {A.shape[0]} rows, Columns: {list(A.columns)}")
        print(f"Routes loaded:   {R.shape[0]} rows, Columns: {list(R.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Process data and build graph
    print("\nBuilding graph...")
    try:
        A, R = merge_airports_routes(A, R)
        G = build_digraph(A, R, add_distance=True)
        print(f"Graph built successfully:")
        print(f"  Nodes (|V|): {G.number_of_nodes()}")
        print(f"  Edges (|E|): {G.number_of_edges()}")
    except Exception as e:
        print(f"Error building graph: {e}")

if __name__ == "__main__":
    main()
