"""
Centralized configuration constants for the airline robustness analysis framework.

This module defines constants used throughout the codebase to avoid magic numbers
and ensure consistency across modules.
"""

# ============================================================================
# Visualization Constants
# ============================================================================
MAX_DISPLAY_EDGES = 5000  # Maximum edges to render in map visualization (for browser performance)

# ============================================================================
# Algorithm Parameters
# ============================================================================
DEFAULT_DAMPING_FACTOR = 0.85  # Standard PageRank damping factor
DEFAULT_CI_DISTANCE = 2  # Default distance parameter for Collective Influence metric
DEFAULT_HOP_LIMIT = 4  # Default hop limit for OD reachability calculation

# ============================================================================
# Defense Strategy Parameters
# ============================================================================
DEFAULT_MAX_ROUTE_DISTANCE_KM = 3000  # Maximum distance for new routes in defense strategies
DEFAULT_TOP_N_PER_COMMUNITY = 10  # Top-N nodes per community to consider for edge addition

# ============================================================================
# Geographic Constants
# ============================================================================
EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers (for Haversine formula)

# ============================================================================
# Visual Hierarchy Constants
# ============================================================================
DEFAULT_TOP_N_HIGHLIGHTED = 20  # Default number of top nodes to emphasize
NODE_SIZE_EMPHASIZED = 80000  # Radius for emphasized nodes (meters)
NODE_SIZE_DIMMED = 25000  # Radius for dimmed nodes (meters)
NODE_OPACITY_EMPHASIZED = 220  # Alpha for emphasized nodes (0-255)
NODE_OPACITY_DIMMED = 50  # Alpha for dimmed nodes (0-255)

# ============================================================================
# Attack/Defense Visualization Colors (RGBA)
# ============================================================================
NORMAL_NODE_COLOR = [100, 100, 180, 160]  # Blue-gray for normal nodes
EMPHASIZED_NODE_COLOR = [255, 140, 0, 220]  # Orange for top-N nodes
ATTACK_NODE_COLOR = [220, 50, 50, 220]  # Red for removed nodes
ATTACK_EDGE_COLOR = [220, 50, 50, 150]  # Red for removed edges
DEFENSE_EDGE_COLOR = [50, 200, 100, 180]  # Green for added edges
HARDENED_NODE_COLOR = [50, 150, 220, 220]  # Blue for hardened nodes
CLUSTER_NODE_COLOR = [120, 80, 200, 180]  # Purple for cluster super-nodes

# ============================================================================
# Clustering Constants
# ============================================================================
CLUSTER_GRID_SIZE_DEG = 5.0  # Grid cell size in degrees for geographic clustering
MIN_CLUSTER_SIZE = 3  # Minimum nodes to form a cluster
