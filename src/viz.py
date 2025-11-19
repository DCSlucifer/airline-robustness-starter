"""
Plotting helpers (kept minimal; Streamlit app offers interactivity).
"""
from __future__ import annotations
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx

def plot_gwcc_curve(xs: List[int], ys: List[float], out_path: str):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Removed nodes")
    plt.ylabel("GWCC fraction")
    plt.title("Robustness curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def draw_map_quick(G: nx.DiGraph, out_path: str):
    plt.figure()
    pos = {n: (G.nodes[n].get("lon",0), G.nodes[n].get("lat",0)) for n in G.nodes()}
    nx.draw(G.to_undirected(), pos=pos, with_labels=True, node_size=100)
    plt.title("Airline network (lon/lat projected)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
