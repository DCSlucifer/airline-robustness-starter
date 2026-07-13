"""
Node clustering utilities for network visualization.

Provides community-based and geographic clustering to aggregate minor nodes
into super-nodes for improved visual hierarchy and performance.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx

from .constants import CLUSTER_GRID_SIZE_DEG, MIN_CLUSTER_SIZE


def _node_sort_key(node: Any) -> tuple[str, str, str]:
    node_type = type(node)
    return node_type.__module__, node_type.__qualname__, repr(node)


def _valid_coordinates(lat: Any, lon: Any) -> bool:
    try:
        latitude = float(lat)
        longitude = float(lon)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90.0 <= latitude <= 90.0
        and -180.0 <= longitude <= 180.0
    )


def community_clustering(G: nx.DiGraph) -> dict[str, int]:
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

    communities = [
        sorted(community, key=_node_sort_key) for community in label_propagation_communities(U)
    ]
    communities.sort(key=lambda community: tuple(_node_sort_key(node) for node in community))

    node_to_cluster = {}
    for cluster_id, community in enumerate(communities):
        for node in community:
            node_to_cluster[node] = cluster_id

    return node_to_cluster


def geographic_clustering(
    G: nx.DiGraph, grid_size_deg: float = CLUSTER_GRID_SIZE_DEG
) -> dict[str, int]:
    """
    Clusters nodes by geographic grid cells (no external dependencies).

    Divides the world into grid cells and assigns nodes to cells based on lat/lon.

    Args:
        G: The input directed graph with 'lat' and 'lon' node attributes.
        grid_size_deg: Size of grid cells in degrees.

    Returns:
        Dictionary mapping node ID to cluster ID (grid cell index).
    """
    if not math.isfinite(grid_size_deg) or grid_size_deg <= 0:
        raise ValueError("grid_size_deg must be positive and finite")

    if G.number_of_nodes() == 0:
        return {}

    node_cells = {}

    for node, data in G.nodes(data=True):
        lat = data.get("lat")
        lon = data.get("lon")

        # Nodes without usable positions remain individual map nodes rather than
        # being aggregated into an artificial cluster at (0, 0).
        if not _valid_coordinates(lat, lon):
            continue

        cell_lat = int(float(lat) // grid_size_deg)
        cell_lon = int(float(lon) // grid_size_deg)
        node_cells[node] = (cell_lat, cell_lon)

    cell_to_id = {cell: index for index, cell in enumerate(sorted(set(node_cells.values())))}
    return {node: cell_to_id[cell] for node, cell in node_cells.items()}


def cluster_aggregates(G: nx.DiGraph, clusters: dict[str, int]) -> list[dict[str, Any]]:
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
    cluster_nodes: dict[int, list[str]] = {}
    for node, cluster_id in clusters.items():
        if node not in G:
            continue
        if cluster_id not in cluster_nodes:
            cluster_nodes[cluster_id] = []
        cluster_nodes[cluster_id].append(node)

    aggregates = []
    for cluster_id in sorted(cluster_nodes, key=_node_sort_key):
        nodes = sorted(cluster_nodes[cluster_id], key=_node_sort_key)
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
            if _valid_coordinates(lat, lon):
                lats.append(float(lat))
                lons.append(float(lon))
            # Sum degrees as size proxy
            total_degree += G.in_degree(node) + G.out_degree(node)

        if not lats:
            continue

        centroid_lat = sum(lats) / len(lats)
        centroid_lon = sum(lons) / len(lons)

        aggregates.append(
            {
                "cluster_id": cluster_id,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "total_degree": total_degree,
                "node_count": len(nodes),
                "member_nodes": nodes,
            }
        )

    return aggregates


def get_unclustered_nodes(G: nx.DiGraph, clusters: dict[str, int]) -> list[str]:
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
    cluster_sizes: dict[int, int] = {}
    for node in G.nodes():
        if node not in clusters:
            continue
        cluster_id = clusters[node]
        cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

    # Return nodes in small clusters and graph nodes absent from a partial mapping.
    return [
        node
        for node in G.nodes()
        if node not in clusters or cluster_sizes[clusters[node]] < MIN_CLUSTER_SIZE
    ]
