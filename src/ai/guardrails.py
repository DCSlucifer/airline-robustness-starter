"""Validation and clamping of LLM-supplied tool arguments.

No LLM-supplied value reaches a simulation function without passing through
``validate_and_clamp``. This is the production-safety boundary.
"""
from __future__ import annotations
from typing import Any, Dict

import networkx as nx

from .tools import TOOL_NAMES

__all__ = ["validate_and_clamp", "GuardrailError"]

_METRICS = {"degree", "betweenness", "pagerank"}
_MAX_BUDGET = 10
_MAX_RADIUS_KM = 20000.0
_MAX_DISTANCE_KM = 20000.0

_REQUIRED = {
    "targeted_attack": ("metric", "k"),
    "geographic_attack": ("lat", "lon", "radius_km"),
    "edge_attack": ("m",),
    "community_bridge_attack": ("m",),
    "defend": ("budget", "max_distance_km"),
}


class GuardrailError(ValueError):
    """Raised when LLM-supplied arguments are unusable or unsafe."""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def validate_and_clamp(name: str, args: Dict[str, Any], G: nx.DiGraph) -> Dict[str, Any]:
    """Return a new args dict with safe, in-range values. Raise GuardrailError if invalid."""
    if name not in TOOL_NAMES:
        raise GuardrailError(f"Tool '{name}' is not in the allowed tool list")

    required = _REQUIRED.get(name)
    if required is None:
        # A tool exists in TOOL_NAMES but has no argument policy here — fail safe
        # rather than leak a KeyError.
        raise GuardrailError(f"Tool '{name}' has no argument policy defined")

    for key in required:
        if key not in args:
            raise GuardrailError(f"Missing required argument '{key}' for tool '{name}'")

    out: Dict[str, Any] = dict(args)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if "metric" in out and out["metric"] not in _METRICS:
        raise GuardrailError(f"Invalid metric '{out['metric']}'")

    if "k" in out:
        out["k"] = int(_clamp(int(out["k"]), 1, max(1, n_nodes)))
    if "m" in out:
        out["m"] = int(_clamp(int(out["m"]), 1, max(1, n_edges)))
    if "budget" in out:
        out["budget"] = int(_clamp(int(out["budget"]), 1, _MAX_BUDGET))
    if "radius_km" in out:
        out["radius_km"] = _clamp(float(out["radius_km"]), 1.0, _MAX_RADIUS_KM)
    if "max_distance_km" in out:
        out["max_distance_km"] = _clamp(float(out["max_distance_km"]), 100.0, _MAX_DISTANCE_KM)

    return out
