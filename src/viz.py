"""
Visualization utilities for static plotting and interactive maps.

This module provides basic plotting functions using Matplotlib for generating
robustness curves and quick network visualizations. For interactive visualizations,
refer to the Streamlit application.
"""
from __future__ import annotations
import os
from typing import Dict, List, Set, Tuple, Any, Optional
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pydeck as pdk

from .constants import (
    MAX_DISPLAY_EDGES,
    NODE_SIZE_EMPHASIZED,
    NODE_SIZE_DIMMED,
    NODE_OPACITY_EMPHASIZED,
    NODE_OPACITY_DIMMED,
    NORMAL_NODE_COLOR,
    EMPHASIZED_NODE_COLOR,
    ATTACK_NODE_COLOR,
    ATTACK_EDGE_COLOR,
    DEFENSE_EDGE_COLOR,
    HARDENED_NODE_COLOR,
    CLUSTER_NODE_COLOR,
)


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


# =============================================================================
# PyDeck Map Visualization Components
# =============================================================================

def compute_node_emphasis(
    G: nx.DiGraph,
    top_n: int,
    metric: str = "degree",
    removed_nodes: Optional[Set[str]] = None,
    hardened_nodes: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes visual emphasis properties for each node.

    Args:
        G: The input graph.
        top_n: Number of top nodes to emphasize.
        metric: Metric for ranking ("degree", "betweenness", "pagerank").
        removed_nodes: Set of nodes that were removed (attack).
        hardened_nodes: Set of nodes that were hardened (defense).

    Returns:
        Dict mapping node ID to {size, color, opacity, is_emphasized}.
    """
    removed_nodes = removed_nodes or set()
    hardened_nodes = hardened_nodes or set()

    # Compute scores for ranking
    if metric == "degree":
        scores = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(G, normalized=True)
    elif metric == "pagerank":
        scores = nx.pagerank(G, alpha=0.85)
    else:
        scores = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}

    # Get top-N nodes
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_n_nodes = set(n for n, _ in sorted_nodes[:top_n])

    emphasis = {}
    for node in G.nodes():
        if node in removed_nodes:
            emphasis[node] = {
                "size": NODE_SIZE_EMPHASIZED,
                "color": ATTACK_NODE_COLOR,
                "opacity": NODE_OPACITY_EMPHASIZED,
                "is_emphasized": True,
            }
        elif node in hardened_nodes:
            emphasis[node] = {
                "size": NODE_SIZE_EMPHASIZED,
                "color": HARDENED_NODE_COLOR,
                "opacity": NODE_OPACITY_EMPHASIZED,
                "is_emphasized": True,
            }
        elif node in top_n_nodes:
            emphasis[node] = {
                "size": NODE_SIZE_EMPHASIZED,
                "color": EMPHASIZED_NODE_COLOR,
                "opacity": NODE_OPACITY_EMPHASIZED,
                "is_emphasized": True,
            }
        else:
            emphasis[node] = {
                "size": NODE_SIZE_DIMMED,
                "color": NORMAL_NODE_COLOR,
                "opacity": NODE_OPACITY_DIMMED,
                "is_emphasized": False,
            }

    return emphasis


def build_node_layer(
    G: nx.DiGraph,
    emphasis: Dict[str, Dict[str, Any]],
    show_labels_only_emphasized: bool = True,
) -> Tuple[pdk.Layer, pd.DataFrame]:
    """
    Builds a PyDeck ScatterplotLayer with node emphasis.

    Args:
        G: The input graph.
        emphasis: Dict from compute_node_emphasis().
        show_labels_only_emphasized: If True, only emphasized nodes get labels.

    Returns:
        Tuple of (PyDeck Layer, DataFrame for tooltip).
    """
    nodes = []
    for node, data in G.nodes(data=True):
        if "lat" not in data or "lon" not in data:
            continue

        node_emphasis = emphasis.get(node, {})
        show_label = (
            not show_labels_only_emphasized or
            node_emphasis.get("is_emphasized", False)
        )

        nodes.append({
            "iata": node,
            "name": data.get("name", node) if show_label else "",
            "lat": data["lat"],
            "lon": data["lon"],
            "size": node_emphasis.get("size", NODE_SIZE_DIMMED),
            "color": node_emphasis.get("color", NORMAL_NODE_COLOR),
        })

    df = pd.DataFrame(nodes)
    if df.empty:
        return None, df

    layer = pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="size",
        pickable=True,
        radius_min_pixels=2,
        radius_max_pixels=15,
    )
    return layer, df


def build_edge_layer(
    G: nx.DiGraph,
    removed_edges: Optional[Set[Tuple[str, str]]] = None,
    added_edges: Optional[Set[Tuple[str, str]]] = None,
    sample_limit: int = MAX_DISPLAY_EDGES,
) -> List[pdk.Layer]:
    """
    Builds PyDeck ArcLayers for normal, removed, and added edges.

    Args:
        G: The input graph.
        removed_edges: Set of (u, v) edges removed by attack.
        added_edges: Set of (u, v) edges added by defense.
        sample_limit: Max edges to display for performance.

    Returns:
        List of PyDeck Layers (normal edges, attack edges, defense edges).
    """
    removed_edges = removed_edges or set()
    added_edges = added_edges or set()

    normal_arcs = []
    attack_arcs = []
    defense_arcs = []

    all_edges = list(G.edges())
    # Add removed edges back for visualization
    all_edges.extend(removed_edges)

    # Sample if too many
    import random
    if len(all_edges) > sample_limit:
        all_edges = random.sample(all_edges, sample_limit)

    for u, v in all_edges:
        # Get coordinates
        u_data = G.nodes.get(u, {})
        v_data = G.nodes.get(v, {})

        u_lat, u_lon = u_data.get("lat"), u_data.get("lon")
        v_lat, v_lon = v_data.get("lat"), v_data.get("lon")

        if None in (u_lat, u_lon, v_lat, v_lon):
            continue

        arc = {
            "source": [u_lon, u_lat],
            "target": [v_lon, v_lat],
            "source_iata": u,
            "dest_iata": v,
        }

        if (u, v) in removed_edges:
            attack_arcs.append(arc)
        elif (u, v) in added_edges:
            defense_arcs.append(arc)
        else:
            normal_arcs.append(arc)

    layers = []

    # Normal edges (subtle)
    if normal_arcs:
        df_normal = pd.DataFrame(normal_arcs)
        layers.append(pdk.Layer(
            "ArcLayer",
            df_normal,
            get_source_position="source",
            get_target_position="target",
            get_source_color=[80, 80, 120, 40],
            get_target_color=[80, 80, 120, 40],
            get_width=1,
            width_min_pixels=1,
        ))

    # Attack edges (red, prominent)
    if attack_arcs:
        df_attack = pd.DataFrame(attack_arcs)
        layers.append(pdk.Layer(
            "ArcLayer",
            df_attack,
            get_source_position="source",
            get_target_position="target",
            get_source_color=ATTACK_EDGE_COLOR,
            get_target_color=ATTACK_EDGE_COLOR,
            get_width=3,
            width_min_pixels=2,
        ))

    # Defense edges (green, prominent)
    if defense_arcs:
        df_defense = pd.DataFrame(defense_arcs)
        layers.append(pdk.Layer(
            "ArcLayer",
            df_defense,
            get_source_position="source",
            get_target_position="target",
            get_source_color=DEFENSE_EDGE_COLOR,
            get_target_color=DEFENSE_EDGE_COLOR,
            get_width=3,
            width_min_pixels=2,
        ))

    return layers


def build_cluster_layer(
    cluster_aggregates: List[Dict[str, Any]],
) -> Optional[pdk.Layer]:
    """
    Builds a PyDeck layer for cluster super-nodes.

    Args:
        cluster_aggregates: List from clustering.cluster_aggregates().

    Returns:
        PyDeck ScatterplotLayer for clusters, or None if empty.
    """
    if not cluster_aggregates:
        return None

    clusters = []
    for agg in cluster_aggregates:
        # Scale size by total degree (log scale for better visualization)
        import math
        size = 50000 + 20000 * math.log1p(agg["total_degree"])

        clusters.append({
            "lat": agg["centroid_lat"],
            "lon": agg["centroid_lon"],
            "size": size,
            "color": CLUSTER_NODE_COLOR,
            "label": f"Cluster {agg['cluster_id']} ({agg['node_count']} nodes)",
            "node_count": agg["node_count"],
            "total_degree": agg["total_degree"],
        })

    df = pd.DataFrame(clusters)
    return pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="size",
        pickable=True,
        radius_min_pixels=8,
        radius_max_pixels=40,
    )


def render_interactive_map(
    G: nx.DiGraph,
    emphasis: Dict[str, Dict[str, Any]],
    show_labels_only_emphasized: bool = True,
    removed_edges: Optional[Set[Tuple[str, str]]] = None,
    added_edges: Optional[Set[Tuple[str, str]]] = None,
    cluster_aggregates: Optional[List[Dict[str, Any]]] = None,
    use_clusters: bool = False,
) -> pdk.Deck:
    """
    Renders a complete interactive map with all visualization layers.

    Args:
        G: The input graph.
        emphasis: Node emphasis dict from compute_node_emphasis().
        show_labels_only_emphasized: Label visibility setting.
        removed_edges: Attacked edges to highlight.
        added_edges: Defense edges to highlight.
        cluster_aggregates: Cluster data for super-node view.
        use_clusters: If True, show clusters instead of individual nodes.

    Returns:
        PyDeck Deck object ready for display.
    """
    layers = []

    # Edge layers (always shown behind nodes)
    edge_layers = build_edge_layer(G, removed_edges, added_edges)
    layers.extend(edge_layers)

    # Node layer (or cluster layer)
    if use_clusters and cluster_aggregates:
        cluster_layer = build_cluster_layer(cluster_aggregates)
        if cluster_layer:
            layers.append(cluster_layer)
        # Also show unclustered individual nodes (small clusters)
        from .clustering import get_unclustered_nodes
        unclustered = set(get_unclustered_nodes(G, {}))  # Will handle in app
    else:
        node_layer, _ = build_node_layer(G, emphasis, show_labels_only_emphasized)
        if node_layer:
            layers.append(node_layer)

    # View state centered on data
    view_state = pdk.ViewState(
        latitude=20.0,
        longitude=0.0,
        zoom=1.5,
        pitch=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "{iata}: {name}\n{label}"},
        map_style="mapbox://styles/mapbox/dark-v10",
    )

