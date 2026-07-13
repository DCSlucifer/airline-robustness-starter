"""Command-line driver for reproducible airline robustness simulations."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from .attacks import (
    community_bridge_attack,
    edge_betweenness_attack,
    geographic_attack_radius,
    random_node_failures,
    targeted_node_removal,
)
from .data_io import load_airports, load_routes, merge_airports_routes
from .defenses import greedy_edge_addition
from .graph_build import build_digraph
from .metrics import topological_report

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
_REQUIRED_CONFIG = {
    "airports_csv",
    "routes_csv",
    "output_dir",
    "k_nodes",
    "m_edges",
    "repetitions_R",
    "budget_b",
    "distance_km_max",
}


class SimulationConfigError(ValueError):
    """Raised when a simulation configuration is missing or invalid."""


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without reading configuration or touching the filesystem."""
    parser = argparse.ArgumentParser(description="Airline Network Robustness Simulator")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config path")
    parser.add_argument(
        "--attack",
        default="targeted_nodes",
        choices=[
            "targeted_nodes",
            "random_nodes",
            "edge_betweenness",
            "geographic_radius",
            "community_bridge",
        ],
        help="Attack strategy",
    )
    parser.add_argument(
        "--mode", default="attack", choices=["attack", "defense"], help="Simulation mode"
    )
    parser.add_argument(
        "--metric",
        default="degree",
        choices=["degree", "betweenness", "pagerank", "CI"],
        help="Centrality metric for targeted attacks",
    )
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable adaptive score recomputation",
    )
    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use deterministic sampled topology metrics for large graphs",
    )
    parser.add_argument("--k", type=int, help="Nodes to remove (overrides config)")
    parser.add_argument("--m", type=int, help="Edges to remove (overrides config)")
    parser.add_argument(
        "--R", "--repetitions", dest="repetitions", type=int, help="Random-failure repetitions"
    )
    parser.add_argument("--budget", type=int, help="Defense edge budget")
    parser.add_argument(
        "--distance-km-max",
        "--distance_km_max",
        dest="distance_km_max",
        type=float,
        help="Maximum length of a proposed defense route",
    )
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (overrides config)")
    parser.add_argument("--lat", type=float, default=0.0, help="Geographic attack latitude")
    parser.add_argument("--lon", type=float, default=0.0, help="Geographic attack longitude")
    parser.add_argument(
        "--radius-km",
        "--radius_km",
        dest="radius_km",
        type=float,
        default=1000.0,
        help="Geographic attack radius in km",
    )
    return parser


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the YAML configuration at *path*."""
    config_path = Path(path)
    if not config_path.is_file():
        raise SimulationConfigError(f"Configuration file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SimulationConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise SimulationConfigError(f"Configuration must be a YAML mapping: {config_path}")

    cfg = dict(raw)
    missing = sorted(_REQUIRED_CONFIG - cfg.keys())
    if missing:
        raise SimulationConfigError(f"Configuration missing keys: {', '.join(missing)}")

    for key in ("airports_csv", "routes_csv", "output_dir"):
        if not isinstance(cfg[key], (str, Path)) or not str(cfg[key]).strip():
            raise SimulationConfigError(f"Configuration value '{key}' must be a non-empty path")

    for key in ("k_nodes", "m_edges", "repetitions_R", "budget_b"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise SimulationConfigError(f"Configuration value '{key}' must be a positive integer")

    distance = cfg["distance_km_max"]
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or distance <= 0:
        raise SimulationConfigError("Configuration value 'distance_km_max' must be positive")

    cfg.setdefault("adaptive", True)
    cfg.setdefault("fast_mode", False)
    cfg.setdefault("random_seed", 42)
    cfg.setdefault("hops_H", 4)
    return cfg


def apply_overrides(cfg: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Return a copy of *cfg* with explicit CLI overrides applied."""
    out = dict(cfg)
    overrides = {
        "k_nodes": args.k,
        "m_edges": args.m,
        "repetitions_R": args.repetitions,
        "budget_b": args.budget,
        "distance_km_max": args.distance_km_max,
        "random_seed": args.seed,
        "output_dir": args.output_dir,
        "adaptive": args.adaptive,
        "fast_mode": args.fast,
    }
    out.update({key: value for key, value in overrides.items() if value is not None})
    return out


def build_graph_from_config(cfg: Mapping[str, Any]) -> nx.DiGraph:
    """Construct a directed airline graph from validated configuration paths."""
    airports = load_airports(str(cfg["airports_csv"]))
    routes = load_routes(str(cfg["routes_csv"]))
    airports, routes = merge_airports_routes(airports, routes)
    return build_digraph(airports, routes, add_distance=True)


def _json_safe(value: Any) -> Any:
    """Convert tuples and non-finite floats into standards-compliant JSON values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json_atomic(path: str | Path, value: Any) -> Path:
    """Atomically write standards-compliant JSON and return the final path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(_json_safe(value), handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def _scenario_name(args: argparse.Namespace) -> str:
    if args.mode == "defense":
        return "defense"
    if args.attack == "targeted_nodes":
        return f"targeted_nodes_{args.metric.lower()}"
    return args.attack


def run_simulation(cfg: Mapping[str, Any], args: argparse.Namespace) -> list[Path]:
    """Execute one configured scenario and return every JSON artifact written."""
    graph = build_graph_from_config(cfg)
    fast_mode = bool(cfg.get("fast_mode", False))
    baseline = topological_report(graph, H=int(cfg.get("hops_H", 4)), fast_mode=fast_mode)
    output_dir = Path(cfg["output_dir"])
    written = [write_json_atomic(output_dir / "baseline_report.json", baseline)]

    if args.mode == "defense":
        _, log = greedy_edge_addition(
            graph,
            budget=int(cfg["budget_b"]),
            max_distance_km=float(cfg["distance_km_max"]),
            fast_mode=fast_mode,
        )
        written.append(write_json_atomic(output_dir / "defense_log.json", log))
        return written

    if args.attack == "targeted_nodes":
        _, log = targeted_node_removal(
            graph,
            k=int(cfg["k_nodes"]),
            metric=args.metric,
            adaptive=bool(cfg["adaptive"]),
            l_ci=int(cfg.get("collective_influence_l", 2)),
            fast_mode=fast_mode,
        )
    elif args.attack == "random_nodes":
        log = random_node_failures(
            graph,
            k=int(cfg["k_nodes"]),
            R=int(cfg["repetitions_R"]),
            seed=int(cfg.get("random_seed", 42)),
            fast_mode=fast_mode,
        )
    elif args.attack == "edge_betweenness":
        _, log = edge_betweenness_attack(
            graph,
            m=int(cfg["m_edges"]),
            adaptive=bool(cfg["adaptive"]),
            fast_mode=fast_mode,
        )
    elif args.attack == "geographic_radius":
        _, info = geographic_attack_radius(
            graph, (args.lat, args.lon), args.radius_km, fast_mode=fast_mode
        )
        log = [info]
    elif args.attack == "community_bridge":
        _, info = community_bridge_attack(graph, m=int(cfg["m_edges"]), fast_mode=fast_mode)
        log = [info]
    else:  # pragma: no cover - argparse choices make this unreachable
        raise SimulationConfigError(f"Unknown attack type: {args.attack}")

    canonical = output_dir / "attack_log.json"
    scenario = output_dir / f"attack_log_{_scenario_name(args)}.json"
    written.extend([write_json_atomic(canonical, log), write_json_atomic(scenario, log)])
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Return a process status instead of terminating test callers."""
    args = build_parser().parse_args(argv)
    try:
        cfg = apply_overrides(load_config(args.config), args)
        written = run_simulation(cfg, args)
    except (OSError, KeyError, TypeError, SimulationConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Simulation complete. Wrote {len(written)} artifact(s) to {Path(cfg['output_dir'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
