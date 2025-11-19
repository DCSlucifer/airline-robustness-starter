# check_load.py
import yaml, pandas as pd
from src.data_io import load_airports, load_routes, merge_airports_routes
from src.graph_build import build_digraph

cfg = yaml.safe_load(open("config/default.yaml"))
print("cfg:", cfg)

A = load_airports(cfg["airports_csv"])
R = load_routes(cfg["routes_csv"])
print("Airports:", A.shape, list(A.columns))
print("Routes  :", R.shape, list(R.columns))

A, R = merge_airports_routes(A, R)
G = build_digraph(A, R, add_distance=True)
print("Graph   :", f"|V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")
