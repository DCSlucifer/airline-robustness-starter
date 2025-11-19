"""
Command-line driver for running network robustness simulations.

This module serves as the entry point for executing various attack and defense
scenarios via the command line. It handles configuration loading, graph construction,
simulation execution, and results logging.
"""
from __future__ import annotations
import argparse
import json
import os
import pandas as pd
import networkx as nx
import yaml
from typing import Dict
from .data_io import load_airports, load_routes, merge_airports_routes
from .graph_build import build_digraph
from .centrality import node_centralities
from .metrics import topological_report
from .attacks import (
    targeted_node_removal,
    random_node_failures,
    edge_betweenness_attack,
    geographic_attack_radius,
    community_bridge_attack
)
from .defenses import greedy_edge_addition, node_hardening_list

def build_graph_from_config(cfg: Dict) -> nx.DiGraph:
    """
    Constructs the airline network graph based on the provided configuration.

    Args:
        cfg: A dictionary containing configuration parameters, specifically
             paths to 'airports_csv' and 'routes_csv'.

    Returns:
        A NetworkX DiGraph representing the airline network with distances calculated.
    """
    airports = load_airports(cfg["airports_csv"])
    routes = load_routes(cfg["routes_csv"])
    airports, routes = merge_airports_routes(airports, routes)
    G = build_digraph(airports, routes, add_distance=True)
    return G

def main():
    """
    Main execution function.

    Parses command-line arguments, loads configuration, builds the graph,
    runs the specified simulation (attack or defense), and saves the results.
    """
    ap = argparse.ArgumentParser(description="Airline Network Robustness Simulator")
    ap.add_argument("--config", default="config/default.yaml", help="Path to YAML config file")
    ap.add_argument("--attack", default="targeted_nodes",
                    choices=["targeted_nodes", "random_nodes", "edge_betweenness", "geographic_radius", "community_bridge"],
                    help="Type of attack to simulate")
    ap.add_argument("--mode", default="attack", choices=["attack", "defense"], help="Simulation mode")
    ap.add_argument("--metric", default="degree", choices=["degree", "betweenness", "pagerank", "CI"],
                    help="Centrality metric for targeted attacks")
    ap.add_argument("--adaptive", type=lambda x: str(x).lower() != "false", default=True,
                    help="Enable adaptive recomputation of metrics (True/False)")

    # Optional overrides for config parameters
    ap.add_argument("--k", type=int, default=None, help="Number of nodes to remove (overrides config)")
    ap.add_argument("--m", type=int, default=None, help="Number of edges to remove (overrides config)")
    ap.add_argument("--R", type=int, default=None, help="Number of repetitions for random attacks (overrides config)")
    ap.add_argument("--budget", type=int, default=None, help="Budget for defense edge addition (overrides config)")
    ap.add_argument("--distance_km_max", type=float, default=None, help="Max distance for new edges (overrides config)")

    # Geographic attack parameters
    ap.add_argument("--lat", type=float, default=0.0, help="Latitude for geographic attack center")
    ap.add_argument("--lon", type=float, default=0.0, help="Longitude for geographic attack center")
    ap.add_argument("--radius_km", type=float, default=1000.0, help="Radius in km for geographic attack")

    args = ap.parse_args()

    # Load configuration from YAML file
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Override config values with command-line arguments if provided
    if args.k is not None: cfg["k_nodes"] = args.k
    if args.m is not None: cfg["m_edges"] = args.m
    if args.R is not None: cfg["repetitions_R"] = args.R
    if args.budget is not None: cfg["budget_b"] = args.budget
    if args.distance_km_max is not None: cfg["distance_km_max"] = args.distance_km_max
    cfg["adaptive"] = args.adaptive

    # Ensure output directory exists
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Build the graph and calculate baseline metrics
    G = build_graph_from_config(cfg)
    base_report = topological_report(G)

    # Save baseline report
    with open(os.path.join(cfg["output_dir"], "baseline_report.json"), "w") as f:
        json.dump(base_report, f, indent=2)

    # Execute the selected simulation mode
    if args.mode == "attack":
        if args.attack == "targeted_nodes":
            H, log = targeted_node_removal(
                G,
                k=cfg["k_nodes"],
                metric=args.metric,
                adaptive=cfg["adaptive"],
                l_ci=cfg.get("collective_influence_l", 2)
            )
        elif args.attack == "random_nodes":
            log = random_node_failures(G, k=cfg["k_nodes"], R=cfg["repetitions_R"])
            H = None  # Random attacks produce multiple outcomes, not a single graph
        elif args.attack == "edge_betweenness":
            H, log = edge_betweenness_attack(G, m=cfg["m_edges"], adaptive=cfg["adaptive"])
        elif args.attack == "geographic_radius":
            H, info = geographic_attack_radius(G, (args.lat, args.lon), args.radius_km)
            log = [info]
        elif args.attack == "community_bridge":
            H, info = community_bridge_attack(G, m=cfg["m_edges"])
            log = [info]
        else:
            raise ValueError(f"Unknown attack type: {args.attack}")

        # Save attack log
        with open(os.path.join(cfg["output_dir"], "attack_log.json"), "w") as f:
            json.dump(log, f, indent=2)

    else:
        # Defense mode: Greedy Edge Addition
        H, log = greedy_edge_addition(
            G,
            budget=cfg["budget_b"],
            max_distance_km=cfg["distance_km_max"]
        )

        # Save defense log
        with open(os.path.join(cfg["output_dir"], "defense_log.json"), "w") as f:
            json.dump(log, f, indent=2)

    print(f"Simulation complete. Results saved to {cfg['output_dir']}/")

if __name__ == "__main__":
    main()
