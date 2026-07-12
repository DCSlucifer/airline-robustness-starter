"""Plot robustness artifacts produced by :mod:`src.simulate`."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


class PlotDataError(ValueError):
    """Raised when simulation artifacts are missing or malformed."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PlotDataError(f"Artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PlotDataError(f"Invalid JSON artifact {path}: {exc}") from exc


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _metric(report: Mapping[str, Any], key: str) -> float | None:
    aliases = {
        "gwcc_frac": ("gwcc_frac", "gwcc"),
        "aspl_gwcc": ("aspl_gwcc", "aspl"),
    }
    for candidate in aliases[key]:
        value = _finite_number(report.get(candidate))
        if value is not None:
            return value
    return None


def _legacy_curve(entries: list[Any], metric: str) -> tuple[list[float], list[float]] | None:
    points = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        fraction = _finite_number(entry.get("fraction_removed"))
        value = _metric(entry, metric)
        if fraction is not None and value is not None:
            points.append((fraction, value))
    if not points:
        return None
    points.sort()
    return [point[0] for point in points], [point[1] for point in points]


def load_curve(
    log_path: str | Path,
    baseline: Mapping[str, Any],
    metric: str,
) -> tuple[list[float], list[float]]:
    """Convert one attack log into fraction-removed and metric series."""
    raw = _load_json(Path(log_path))
    if isinstance(raw, Mapping):
        raw = raw.get("steps", [])
    if not isinstance(raw, list):
        raise PlotDataError(f"Attack log must contain a JSON list: {log_path}")

    legacy = _legacy_curve(raw, metric)
    if legacy is not None:
        return legacy

    n_nodes = _finite_number(baseline.get("n_nodes"))
    base_value = _metric(baseline, metric)
    if not n_nodes or base_value is None:
        raise PlotDataError("baseline_report.json is missing n_nodes or topology metrics")

    points: list[tuple[float, float]] = [(0.0, base_value)]
    random_points: list[tuple[float, float]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        report = entry.get("report")
        if not isinstance(report, Mapping):
            continue
        value = _metric(report, metric)
        if value is None:
            continue

        step = _finite_number(entry.get("step"))
        if step is not None and entry.get("removed_node") is not None:
            points.append((step / n_nodes, value))
            continue

        removed_nodes = entry.get("removed_nodes")
        if isinstance(removed_nodes, list):
            random_points.append((len(removed_nodes) / n_nodes, value))

    if random_points:
        fraction = statistics.fmean(point[0] for point in random_points)
        value = statistics.fmean(point[1] for point in random_points)
        points.append((fraction, value))

    if len(points) == 1:
        raise PlotDataError(f"No plottable '{metric}' reports found in {log_path}")

    points = sorted(set(points))
    return [point[0] for point in points], [point[1] for point in points]


def discover_attack_logs(input_dir: str | Path) -> list[Path]:
    """Return non-canonical node-attack logs in a deterministic order."""
    root = Path(input_dir)
    logs = sorted(root.glob("attack_log_targeted_nodes_*.json"))
    random_log = root / "attack_log_random_nodes.json"
    if random_log.is_file():
        logs.insert(0, random_log)

    # Support the original filenames when users already have those artifacts.
    for legacy in ("attack_log_random.json", "attack_log_degree.json", "attack_log_ci.json"):
        path = root / legacy
        if path.is_file() and path not in logs:
            logs.append(path)
    return logs


def _label(path: Path) -> str:
    labels = {
        "attack_log_random_nodes": "Random failure",
        "attack_log_random": "Random failure",
        "attack_log_targeted_nodes_degree": "Targeted (degree)",
        "attack_log_degree": "Targeted (degree)",
        "attack_log_targeted_nodes_ci": "Targeted (collective influence)",
        "attack_log_ci": "Targeted (collective influence)",
        "attack_log_targeted_nodes_betweenness": "Targeted (betweenness)",
        "attack_log_targeted_nodes_pagerank": "Targeted (PageRank)",
    }
    return labels.get(path.stem, path.stem.removeprefix("attack_log_").replace("_", " ").title())


def _demo_curves(metric: str) -> dict[str, tuple[list[float], list[float]]]:
    x = [0.0, 0.05, 0.10, 0.15, 0.20]
    if metric == "gwcc_frac":
        return {
            "Random failure (demo)": (x, [1.0, 0.98, 0.95, 0.90, 0.85]),
            "Targeted degree (demo)": (x, [1.0, 0.80, 0.60, 0.40, 0.20]),
            "Targeted CI (demo)": (x, [1.0, 0.70, 0.40, 0.15, 0.05]),
        }
    return {"Targeted degree (demo)": (x, [3.5, 3.8, 4.5, 5.2, 6.0])}


def _curves(input_dir: Path, metric: str, demo: bool) -> dict[str, tuple[list[float], list[float]]]:
    if demo:
        return _demo_curves(metric)

    logs = discover_attack_logs(input_dir)
    if not logs:
        raise PlotDataError(
            f"No named attack logs found in {input_dir}. Run src.simulate first or pass --demo."
        )
    baseline = _load_json(input_dir / "baseline_report.json")
    if not isinstance(baseline, Mapping):
        raise PlotDataError("baseline_report.json must contain a JSON object")
    return {_label(path): load_curve(path, baseline, metric) for path in logs}


def plot_robustness_curve(
    input_dir: str | Path = "outputs",
    output_dir: str | Path = "outputs",
    *,
    demo: bool = False,
) -> Path:
    """Plot GWCC degradation for every discovered node-attack log."""
    curves = _curves(Path(input_dir), "gwcc_frac", demo)
    destination = Path(output_dir) / "robustness_curve_gwcc.png"
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 6))
    for label, (x, y) in curves.items():
        axis.plot(x, y, label=label, marker="o", linewidth=2, markersize=5)
    axis.set_title("Network Robustness: GWCC Degradation")
    axis.set_xlabel("Fraction of nodes removed")
    axis.set_ylabel("GWCC fraction")
    axis.grid(True, linestyle="--", alpha=0.7)
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_efficiency_drop(
    input_dir: str | Path = "outputs",
    output_dir: str | Path = "outputs",
    *,
    demo: bool = False,
) -> Path:
    """Plot ASPL change for discovered node-attack logs."""
    curves = _curves(Path(input_dir), "aspl_gwcc", demo)
    destination = Path(output_dir) / "efficiency_drop_aspl.png"
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 6))
    for label, (x, y) in curves.items():
        axis.plot(x, y, label=label, marker="s", linewidth=2, markersize=5)
    axis.set_title("Attack Impact on Network Efficiency")
    axis.set_xlabel("Fraction of nodes removed")
    axis.set_ylabel("Average shortest path length (hops)")
    axis.grid(True, linestyle="--", alpha=0.7)
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Render clearly labelled synthetic demonstration data instead of simulation logs",
    )
    args = parser.parse_args(argv)

    try:
        paths = [
            plot_robustness_curve(args.input_dir, args.output_dir, demo=args.demo),
            plot_efficiency_drop(args.input_dir, args.output_dir, demo=args.demo),
        ]
    except (OSError, PlotDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("Generated plots:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
