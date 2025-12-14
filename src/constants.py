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
