"""
Visualization utilities for static plotting.

This module provides basic plotting functions using Matplotlib for generating
robustness curves and quick network visualizations. For interactive visualizations,
refer to the Streamlit application.
"""
from __future__ import annotations
import os
from typing import Dict, List
import matplotlib.pyplot as plt
import networkx as nx

def plot_gwcc_curve(xs: List[int], ys: List[float], out_path: str):
    """
    Plots the robustness curve showing the degradation of the Giant Weakly Connected Component (GWCC).

    Args:
        xs: List of integers representing the number of removed nodes (x-axis).
        ys: List of floats representing the fraction of the GWCC remaining (y-axis).
        out_path: File path to save the generated plot image.
    """
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Removed nodes")
    plt.ylabel("GWCC fraction")
    plt.title("Robustness curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def draw_map_quick(G: nx.DiGraph, out_path: str):
    """
    Generates a quick static map of the network using geographic coordinates.

    This function uses the 'lon' and 'lat' node attributes as positions for the layout.
    It is intended for quick debugging or overview, not for high-quality cartography.

    Args:
        G: The input directed graph.
        out_path: File path to save the generated map image.
    """
    plt.figure()

    # Extract positions from node attributes (defaulting to 0,0 if missing)
    pos = {n: (G.nodes[n].get("lon", 0), G.nodes[n].get("lat", 0)) for n in G.nodes()}

    # Draw the graph (converted to undirected for simpler visualization)
    nx.draw(G.to_undirected(), pos=pos, with_labels=True, node_size=100)

    plt.title("Airline network (lon/lat projected)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
