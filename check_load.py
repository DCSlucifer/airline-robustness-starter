"""Validate configuration, data loading, and graph construction from the command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from src.simulate import build_graph_from_config, load_config


def main(argv: Sequence[str] | None = None) -> int:
    """Run the load check and return a shell-friendly status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        graph = build_graph_from_config(cfg)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Load check failed: {exc}", file=sys.stderr)
        return 2

    print(f"Configuration: {args.config}")
    print(f"Airports: {cfg['airports_csv']}")
    print(f"Routes: {cfg['routes_csv']}")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
