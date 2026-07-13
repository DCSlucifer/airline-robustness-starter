"""Golden evaluation set: natural-language queries -> expected tool + key arguments.

Argument expectations list only the deterministic parts of a query (e.g. an explicit
count or radius). Geographic coordinates are inferred by the model and intentionally omitted.
"""

from __future__ import annotations

from typing import Any

GOLDEN_SET: list[dict[str, Any]] = [
    {
        "query": "What if we lose the 5 busiest hub airports?",
        "expected_tool": "targeted_attack",
        "expected_args": {"metric": "degree", "k": 5},
    },
    {
        "query": "Remove the 10 most central airports by betweenness.",
        "expected_tool": "targeted_attack",
        "expected_args": {"metric": "betweenness", "k": 10},
    },
    {
        "query": "Attack the top 3 airports ranked by pagerank.",
        "expected_tool": "targeted_attack",
        "expected_args": {"metric": "pagerank", "k": 3},
    },
    {
        "query": "What happens if a hurricane hits the US East Coast within 500 km?",
        "expected_tool": "geographic_attack",
        "expected_args": {"radius_km": 500},
    },
    {
        "query": "Simulate an earthquake disabling every airport within 300 km of Tokyo.",
        "expected_tool": "geographic_attack",
        "expected_args": {"radius_km": 300},
    },
    {
        "query": "A regional storm knocks out airports in a 1000 km radius around London.",
        "expected_tool": "geographic_attack",
        "expected_args": {"radius_km": 1000},
    },
    {
        "query": "Cut the 8 most critical routes by edge betweenness.",
        "expected_tool": "edge_attack",
        "expected_args": {"m": 8},
    },
    {
        "query": "What if the 15 busiest flight connections are severed?",
        "expected_tool": "edge_attack",
        "expected_args": {"m": 15},
    },
    {
        "query": "Sever the 12 routes that bridge different regions.",
        "expected_tool": "community_bridge_attack",
        "expected_args": {"m": 12},
    },
    {
        "query": "Remove the 6 connections between network communities.",
        "expected_tool": "community_bridge_attack",
        "expected_args": {"m": 6},
    },
    {
        "query": "Add 4 new routes within 3000 km to improve resilience.",
        "expected_tool": "defend",
        "expected_args": {"budget": 4, "max_distance_km": 3000},
    },
    {
        "query": "Make the network more robust by adding 2 short routes under 1500 km.",
        "expected_tool": "defend",
        "expected_args": {"budget": 2, "max_distance_km": 1500},
    },
]
