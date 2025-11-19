"""
Command-line driver to run attacks/defenses and dump reports.
"""
from __future__ import annotations
import argparse, json, os
import pandas as pd
import networkx as nx
from typing import Dict
from .data_io import load_airports, load_routes, merge_airports_routes
from .graph_build import build_digraph
from .centrality import node_centralities
from .metrics import topological_report
from .attacks import targeted_node_removal, random_node_failures, edge_betweenness_attack, geographic_attack_radius, community_bridge_attack
from .defenses import greedy_edge_addition, node_hardening_list

def build_graph_from_config(cfg: Dict) -> nx.DiGraph:
    airports = load_airports(cfg["airports_csv"])
    routes = load_routes(cfg["routes_csv"])
    airports, routes = merge_airports_routes(airports, routes)
    G = build_digraph(airports, routes, add_distance=True)
    return G

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--attack", default="targeted_nodes", choices=["targeted_nodes","random_nodes","edge_betweenness","geographic_radius","community_bridge"])
    ap.add_argument("--mode", default="attack", choices=["attack","defense"])
    ap.add_argument("--metric", default="degree", choices=["degree","betweenness","pagerank","CI"])
    ap.add_argument("--adaptive", type=lambda x: str(x).lower()!="false", default=True)
    ap.add_argument("--k", type=int, default=None, help="nodes to remove")
    ap.add_argument("--m", type=int, default=None, help="edges to remove")
    ap.add_argument("--R", type=int, default=None, help="repetitions for random")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--distance_km_max", type=float, default=None)
    ap.add_argument("--lat", type=float, default=0.0)
    ap.add_argument("--lon", type=float, default=0.0)
    ap.add_argument("--radius_km", type=float, default=1000.0)
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # override from CLI
    if args.k is not None: cfg["k_nodes"] = args.k
    if args.m is not None: cfg["m_edges"] = args.m
    if args.R is not None: cfg["repetitions_R"] = args.R
    if args.budget is not None: cfg["budget_b"] = args.budget
    if args.distance_km_max is not None: cfg["distance_km_max"] = args.distance_km_max
    cfg["adaptive"] = args.adaptive

    os.makedirs(cfg["output_dir"], exist_ok=True)

    G = build_graph_from_config(cfg)
    base_report = topological_report(G)
    with open(os.path.join(cfg["output_dir"], "baseline_report.json"), "w") as f:
        json.dump(base_report, f, indent=2)

    if args.mode == "attack":
        if args.attack == "targeted_nodes":
            H, log = targeted_node_removal(G, k=cfg["k_nodes"], metric=args.metric, adaptive=cfg["adaptive"], l_ci=cfg.get("collective_influence_l",2))
        elif args.attack == "random_nodes":
            log = random_node_failures(G, k=cfg["k_nodes"], R=cfg["repetitions_R"])
            H = None
        elif args.attack == "edge_betweenness":
            H, log = edge_betweenness_attack(G, m=cfg["m_edges"], adaptive=cfg["adaptive"])
        elif args.attack == "geographic_radius":
            H, info = geographic_attack_radius(G, (args.lat, args.lon), args.radius_km)
            log = [info]
        elif args.attack == "community_bridge":
            H, info = community_bridge_attack(G, m=cfg["m_edges"])
            log = [info]
        else:
            raise ValueError("Unknown attack")
        with open(os.path.join(cfg["output_dir"], "attack_log.json"), "w") as f:
            json.dump(log, f, indent=2)
    else:
        # defense mode
        H, log = greedy_edge_addition(G, budget=cfg["budget_b"], max_distance_km=cfg["distance_km_max"])
        with open(os.path.join(cfg["output_dir"], "defense_log.json"), "w") as f:
            json.dump(log, f, indent=2)

    print("Done. See outputs/ for JSON logs.")
    
if __name__ == "__main__":
    main()
