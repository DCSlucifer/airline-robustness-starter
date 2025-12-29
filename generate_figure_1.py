
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import random
import yaml
from pathlib import Path
import numpy as np

import sys
sys.path.append(os.getcwd())

from src.data_io import load_airports, load_routes, merge_airports_routes
from src.graph_build import build_digraph
# topological_report is fast enough if fast_mode=True
from src.metrics import topological_report
from src.attacks import targeted_node_removal

# --- CONFIGURATION ---
CONFIG_PATH = "config/default.yaml"
OUTPUT_DIR = "outputs"
IMG_NAME = "Figure_1_GWCC_Robustness_HighRes.png"
REMOVAL_FRACTION = 0.20  # Increased to 20% to show full collapse
STEPS = 20  # Increased resolution for smoother curve

def build_graph():
    print("Loading graph data...", flush=True)
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    airports = load_airports(cfg["airports_csv"])
    routes = load_routes(cfg["routes_csv"])
    airports, routes = merge_airports_routes(airports, routes)
    G = build_digraph(airports, routes, add_distance=True)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)
    return G, cfg

def get_gwcc_frac(G):
    return topological_report(G, fast_mode=True)["gwcc_frac"]

def run_random_attack(G, k, num_points):
    print("Simulating Random Failures...", flush=True)
    # Increase runs for smoother random baseline
    runs = 3

    report_interval = max(1, k // num_points)

    aggregated_results = {}

    for r in range(runs):
        H = G.copy()
        nodes = list(H.nodes())
        random.shuffle(nodes)
        to_remove = nodes[:k]

        current_gwcc = get_gwcc_frac(H)
        aggregated_results.setdefault(0, []).append(current_gwcc)

        for i, node in enumerate(to_remove):
            H.remove_node(node)
            step = i + 1
            if step % report_interval == 0 or step == k:
                frac = get_gwcc_frac(H)
                aggregated_results.setdefault(step, []).append(frac)

    steps = sorted(aggregated_results.keys())
    y_values = [np.mean(aggregated_results[s]) for s in steps]
    x_values = [s / G.number_of_nodes() for s in steps]

    return x_values, y_values

def run_targeted_attack(G, k, metric, name, num_points, adaptive=True):
    print(f"Simulating Targeted Attack ({name}, adaptive={adaptive})...", flush=True)

    report_interval = max(1, k // num_points)

    H, log = targeted_node_removal(
        G,
        k=k,
        metric=metric,
        adaptive=adaptive,
        fast_mode=True,
        report_every_n=report_interval
    )

    x_values = [0.0]
    y_values = [get_gwcc_frac(G)]

    initial_n = G.number_of_nodes()

    for entry in log:
        if entry["report"] is not None:
            x = entry["step"] / initial_n
            y = entry["report"]["gwcc_frac"]
            x_values.append(x)
            y_values.append(y)

    return x_values, y_values

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    G, cfg = build_graph()
    N = G.number_of_nodes()
    k = int(N * REMOVAL_FRACTION)

    print(f"Simulating removal of {k} nodes ({REMOVAL_FRACTION*100}%)")
    print("NOTE: CI Adaptive calculation is computationally intensive. This may take a few minutes...")

    # 1. Random
    x_rand, y_rand = run_random_attack(G, k, STEPS)

    # 2. Targeted Degree (Adaptive)
    x_deg, y_deg = run_targeted_attack(G, k, "degree", "Degree", STEPS, adaptive=True)

    # 3. Targeted CI (Adaptive enabled for report verification)
    x_ci, y_ci = run_targeted_attack(G, k, "CI", "CI", STEPS, adaptive=True)

    # Plotting
    print("Plotting results...", flush=True)
    plt.figure(figsize=(10, 6))

    plt.plot(x_rand, y_rand, label='Random Failure', color='green', marker='o', markersize=4, linestyle='-')
    plt.plot(x_deg, y_deg, label='Targeted (Degree - Adaptive)', color='blue', marker='^', markersize=4, linestyle='-')
    plt.plot(x_ci, y_ci, label='Targeted (CI - Adaptive)', color='red', marker='x', markersize=4, linestyle='-')

    plt.title('GWCC Robustness: Random vs Targeted (Degree, CI)', fontsize=14)
    plt.xlabel('Fraction of Nodes Removed (f)', fontsize=12)
    plt.ylabel('GWCC Fraction', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    save_path = os.path.join(OUTPUT_DIR, IMG_NAME)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}", flush=True)

if __name__ == "__main__":
    main()
