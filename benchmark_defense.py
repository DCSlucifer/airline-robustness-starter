"""
Benchmark script for defense algorithm performance testing.

Measures execution time and results of greedy_edge_addition on the full dataset.
"""
import time
import networkx as nx
from src.defenses import greedy_edge_addition
from src.graph_build import build_digraph
from src.data_io import load_airports, load_routes, merge_airports_routes

def benchmark_defense():
    print("Loading data...")
    airports = load_airports("data/airports.csv")
    routes = load_routes("data/routes.csv")
    airports, routes = merge_airports_routes(airports, routes)
    G = build_digraph(airports, routes, add_distance=True)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    print("Running defense benchmark (budget=5)...")
    start_time = time.time()
    H, log = greedy_edge_addition(G, budget=5, max_distance_km=3000)
    end_time = time.time()

    print(f"Defense finished in {end_time - start_time:.2f} seconds.")
    print(f"Added {len(log)} edges.")
    for entry in log:
        print(f"Step {entry['step']}: Added {entry['added_edges'][0]}")

if __name__ == "__main__":
    benchmark_defense()
