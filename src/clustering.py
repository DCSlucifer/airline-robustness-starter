"""
Node clustering utilities for network visualization.

Provides community-based and geographic clustering to aggregate minor nodes
into super-nodes for improved visual hierarchy and performance.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import networkx as nx
from functools import lru_cache

from .constants import CLUSTER_GRID_SIZE_DEG, MIN_CLUSTER_SIZE


def community_clustering(G: nx.DiGraph) -> Dict[str, int]:
    """
    Clusters nodes using label propagation community detection.

    Args:
        G: The input directed graph.

    Returns:
        Dictionary mapping node ID to cluster ID.
    """
    if G.number_of_nodes() == 0:
        return {}

    # Use undirected view for community detection
    U = G.to_undirected()
    from networkx.algorithms.community import label_propagation_communities

    communities = list(label_propagation_communities(U))

    node_to_cluster = {}
    for cluster_id, community in enumerate(communities):
        for node in community:
            node_to_cluster[node] = cluster_id

    return node_to_cluster


def geographic_clustering(
    G: nx.DiGraph,
    grid_size_deg: float = CLUSTER_GRID_SIZE_DEG
) -> Dict[str, int]:
    """
    Clusters nodes by geographic grid cells (no external dependencies).

    Divides the world into grid cells and assigns nodes to cells based on lat/lon.

    Args:
        G: The input directed graph with 'lat' and 'lon' node attributes.
        grid_size_deg: Size of grid cells in degrees.

    Returns:
        Dictionary mapping node ID to cluster ID (grid cell index).
    """
    if G.number_of_nodes() == 0:
        return {}

    cell_to_id = {}
    node_to_cluster = {}
    next_cluster_id = 0

    for node, data in G.nodes(data=True):
        lat = data.get("lat")
        lon = data.get("lon")

        if lat is None or lon is None:
            # Assign to a special "unknown" cluster
            cell_key = ("unknown",)
        else:
            # Compute grid cell
            cell_lat = int(lat // grid_size_deg)
            cell_lon = int(lon // grid_size_deg)
            cell_key = (cell_lat, cell_lon)

        if cell_key not in cell_to_id:
            cell_to_id[cell_key] = next_cluster_id
            next_cluster_id += 1

        node_to_cluster[node] = cell_to_id[cell_key]

    return node_to_cluster


def cluster_aggregates(
    G: nx.DiGraph,
    clusters: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Computes aggregate statistics for each cluster (super-node data).

    Args:
        G: The input directed graph.
        clusters: Mapping of node ID to cluster ID.

    Returns:
        List of dicts with cluster info: id, centroid_lat, centroid_lon,
        total_degree, node_count, member_nodes.
    """
    if not clusters:
        return []

    # Group nodes by cluster
    cluster_nodes: Dict[int, List[str]] = {}
    for node, cluster_id in clusters.items():
        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []
        cluster_nodes[cluster_id].append(node)

    aggregates = []
    for cluster_id, nodes in cluster_nodes.items():
        # Skip small clusters (they remain as individual nodes)
        if len(nodes) < MIN_CLUSTER_SIZE:
            continue

        # Compute centroid
        lats, lons = [], []
        total_degree = 0

        for node in nodes:
            data = G.nodes.get(node, {})
            lat = data.get("lat")
            lon = data.get("lon")
            if lat is not None and lon is not None:
                lats.append(lat)
                lons.append(lon)
            # Sum degrees as size proxy
            total_degree += G.in_degree(node) + G.out_degree(node)

        centroid_lat = sum(lats) / len(lats) if lats else 0.0
        centroid_lon = sum(lons) / len(lons) if lons else 0.0

        aggregates.append({
            "cluster_id": cluster_id,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "total_degree": total_degree,
            "node_count": len(nodes),
            "member_nodes": nodes,
        })

    return aggregates


def get_unclustered_nodes(
    G: nx.DiGraph,
    clusters: Dict[str, int]
) -> List[str]:
    """
    Returns nodes that are not part of any significant cluster.

    These are nodes in clusters smaller than MIN_CLUSTER_SIZE.

    Args:
        G: The input graph.
        clusters: Mapping of node ID to cluster ID.

    Returns:
        List of node IDs that should be shown individually.
    """
    if not clusters:
        return list(G.nodes())

    # Count cluster sizes
    cluster_sizes: Dict[int, int] = {}
    for cluster_id in clusters.values():
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    # Return nodes in small clusters
    unclustered = []
    for node, cluster_id in clusters.items():
        if cluster_sizes[cluster_id] < MIN_CLUSTER_SIZE:
            unclustered.append(node)

    return unclustered
