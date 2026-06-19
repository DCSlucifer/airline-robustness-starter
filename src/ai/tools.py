"""Tool registry mapping LLM-callable schemas to the real simulation functions.

The LLM selects a tool and arguments; ``run_tool`` executes the corresponding
simulation on the graph and returns grounded metrics. The LLM never computes
these numbers.
"""
from __future__ import annotations
from typing import Any, Dict, List

import networkx as nx

from ..metrics import topological_report
from ..attacks import (
    targeted_node_removal,
    edge_betweenness_attack,
    geographic_attack_radius,
    community_bridge_attack,
)
from ..defenses import greedy_edge_addition

__all__ = ["TOOL_SPECS", "TOOL_NAMES", "run_tool"]

_METRIC_ENUM = ["degree", "betweenness", "pagerank"]

TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "targeted_attack",
        "description": (
            "Remove the k most central airports (hubs) ranked by a centrality "
            "metric. Use for scenarios like 'what if we lose the busiest hubs' "
            "or 'attack the most connected airports'."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": _METRIC_ENUM},
                "k": {"type": "integer", "description": "Number of airports to remove"},
            },
            "required": ["metric", "k"],
            "additionalProperties": False,
        },
    },
    {
        "name": "geographic_attack",
        "description": (
            "Disable all airports within a radius (km) of a latitude/longitude. "
            "Use for regional disasters: hurricanes, earthquakes, 'what if a "
            "storm hits a region'."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_km": {"type": "number"},
            },
            "required": ["lat", "lon", "radius_km"],
            "additionalProperties": False,
        },
    },
    {
        "name": "edge_attack",
        "description": (
            "Remove the m highest-betweenness routes (critical bridge connections). "
            "Use for 'what if key routes are cut' scenarios."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "m": {"type": "integer", "description": "Number of routes to remove"},
            },
            "required": ["m"],
            "additionalProperties": False,
        },
    },
    {
        "name": "community_bridge_attack",
        "description": (
            "Remove m routes that bridge different network communities/regions. "
            "Use for 'what if connections between regions are severed'."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "m": {"type": "integer", "description": "Number of bridge routes to remove"},
            },
            "required": ["m"],
            "additionalProperties": False,
        },
    },
    {
        "name": "defend",
        "description": (
            "Add up to 'budget' new routes (within max_distance_km) to improve "
            "connectivity. Use for 'how can we make the network more resilient' "
            "or 'add routes to recover'."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "budget": {"type": "integer", "description": "Number of routes to add"},
                "max_distance_km": {"type": "number"},
            },
            "required": ["budget", "max_distance_km"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}


def run_tool(name: str, args: Dict[str, Any], G: nx.DiGraph) -> Dict[str, Any]:
    """Execute the named simulation tool with validated args; return grounded metrics."""
    if name not in TOOL_NAMES:
        raise KeyError(f"Unknown tool: {name}")

    baseline = topological_report(G, fast_mode=True)

    if name == "targeted_attack":
        H, log = targeted_node_removal(
            G, k=args["k"], metric=args["metric"], adaptive=True, fast_mode=True
        )
        removed = [e["removed_node"] for e in log if e.get("removed_node")]
        return {
            "baseline": baseline,
            "after": topological_report(H, fast_mode=True),
            "removed_nodes": removed,
        }

    if name == "geographic_attack":
        H, info = geographic_attack_radius(
            G, (args["lat"], args["lon"]), args["radius_km"]
        )
        return {
            "baseline": baseline,
            "after": topological_report(H, fast_mode=True),
            "removed_nodes": list(info.get("removed_nodes", [])),
        }

    if name == "edge_attack":
        H, log = edge_betweenness_attack(G, m=args["m"], adaptive=True, fast_mode=True)
        removed = [e["removed_edge"] for e in log if e.get("removed_edge")]
        return {
            "baseline": baseline,
            "after": topological_report(H, fast_mode=True),
            "removed_edges": removed,
        }

    if name == "community_bridge_attack":
        H, info = community_bridge_attack(G, m=args["m"])
        return {
            "baseline": baseline,
            "after": topological_report(H, fast_mode=True),
            "removed_edges": list(info.get("removed_edges", [])),
        }

    # name == "defend"
    H, log = greedy_edge_addition(
        G, budget=args["budget"], max_distance_km=args["max_distance_km"], fast_mode=True
    )
    added = [e for entry in log for e in entry.get("added_edges", [])]
    return {
        "baseline": baseline,
        "after": topological_report(H, fast_mode=True),
        "added_edges": added,
    }
