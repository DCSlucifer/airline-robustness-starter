"""Plotting consumes the real simulation artifact schema without silent dummy data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from plot_results import (  # noqa: E402 - backend must be selected before pyplot import
    PlotDataError,
    load_curve,
    main,
    plot_efficiency_drop,
    plot_robustness_curve,
)


def _write_artifacts(directory: Path) -> None:
    directory.mkdir(exist_ok=True)
    (directory / "baseline_report.json").write_text(
        json.dumps({"n_nodes": 10, "gwcc_frac": 1.0, "aspl_gwcc": 2.0}),
        encoding="utf-8",
    )
    (directory / "attack_log_targeted_nodes_degree.json").write_text(
        json.dumps(
            [
                {
                    "step": 1,
                    "removed_node": "AAA",
                    "report": {"gwcc_frac": 0.9, "aspl_gwcc": 2.2},
                },
                {"step": 2, "removed_node": "BBB", "report": None},
                {
                    "step": 3,
                    "removed_node": "CCC",
                    "report": {"gwcc_frac": 0.6, "aspl_gwcc": 2.8},
                },
            ]
        ),
        encoding="utf-8",
    )


def test_load_curve_reads_nested_simulation_reports(tmp_path):
    _write_artifacts(tmp_path)
    baseline = {"n_nodes": 10, "gwcc_frac": 1.0, "aspl_gwcc": 2.0}

    x, y = load_curve(tmp_path / "attack_log_targeted_nodes_degree.json", baseline, "gwcc_frac")

    assert x == [0.0, 0.1, 0.3]
    assert y == [1.0, 0.9, 0.6]


def test_plot_functions_generate_nonempty_images_from_real_logs(tmp_path):
    artifacts = tmp_path / "artifacts"
    images = tmp_path / "images"
    _write_artifacts(artifacts)

    robustness = plot_robustness_curve(artifacts, images)
    efficiency = plot_efficiency_drop(artifacts, images)

    assert robustness.stat().st_size > 1000
    assert efficiency.stat().st_size > 1000


def test_missing_logs_fail_instead_of_silently_using_demo_data(tmp_path):
    with pytest.raises(PlotDataError, match="No named attack logs"):
        plot_robustness_curve(tmp_path, tmp_path / "images")


def test_demo_mode_is_explicit_and_runnable(tmp_path):
    assert main(["--input-dir", str(tmp_path), "--output-dir", str(tmp_path), "--demo"]) == 0
    assert (tmp_path / "robustness_curve_gwcc.png").is_file()
    assert (tmp_path / "efficiency_drop_aspl.png").is_file()
