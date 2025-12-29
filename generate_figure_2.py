
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import yaml
from pathlib import Path
import sys

# Setup path
sys.path.append(os.getcwd())

from src.data_io import load_airports, load_routes, merge_airports_routes
from src.graph_build import build_digraph
from src.attacks import _rank_nodes # Helper to get top nodes

# --- CONFIGURATION ---
CONFIG_PATH = "config/default.yaml"
OUTPUT_DIR = "outputs"
IMG_NAME = "Figure_2_Geospatial_Targeted.png"
NUM_TARGETS = 50  # Number of nodes to disable (Top 50 hubs)

def build_graph():
    print("Loading graph data...", flush=True)
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    airports = load_airports(cfg["airports_csv"])
    routes = load_routes(cfg["routes_csv"])
    airports, routes = merge_airports_routes(airports, routes)
    G = build_digraph(airports, routes, add_distance=True)
    return G

def plot_geospatial_attack(G, disabled_nodes, out_path):
    print("Generating map...", flush=True)
    plt.figure(figsize=(12, 8))

    # Position based on lon/lat
    pos = {n: (G.nodes[n]['lon'], G.nodes[n]['lat']) for n in G.nodes() if 'lon' in G.nodes[n]}

    # Subgraph of remaining nodes
    active_nodes = set(G.nodes()) - set(disabled_nodes)

    # Draw edges first (very thin, faint)
    # To save time/memory, we might only draw a subset of edges or alpha them heavily
    # Drawing all edges can be slow for 30k edges on matplotlib
    # Let's draw all but with very low alpha
    print("Drawing edges...", flush=True)
    nx.draw_networkx_edges(G, pos, alpha=0.03, width=0.3, edge_color="gray", arrows=False)

    # Draw active nodes
    print("Drawing active nodes...", flush=True)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(active_nodes),
        node_size=10,
        node_color='steelblue',
        alpha=0.6,
        label="Active Airports"
    )

    # Draw disabled nodes (RED)
    print("Drawing disabled nodes...", flush=True)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(disabled_nodes),
        node_size=60,
        node_color='red',
        node_shape='^',
        alpha=1.0,
        label="Disabled (Targeted)"
    )

    plt.title(f"Figure 2: Geospatial state after targeted attack (Top {len(disabled_nodes)} removed)", fontsize=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.3)

    # Add a note about connectivity
    plt.figtext(0.5, 0.01,
                "Red markers indicate disabled hubs. Note the loss of central connectors.",
                wrap=True, horizontalalignment='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Map saved to {out_path}", flush=True)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    G = build_graph()

    # Identify targets (Degree based)
    # Using internal helper from attacks logic or just sorting degree manually
    print(f"identifying top {NUM_TARGETS} targets...", flush=True)
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
    targets = [node for node, degree in sorted_nodes[:NUM_TARGETS]]

    save_path = os.path.join(OUTPUT_DIR, IMG_NAME)
    plot_geospatial_attack(G, set(targets), save_path)

if __name__ == "__main__":
    main()
