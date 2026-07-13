"""Regression tests for the CLI configuration and artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.simulate import (
    SimulationConfigError,
    build_parser,
    load_config,
    main,
    write_json_atomic,
)


def _write_config(tmp_path: Path) -> Path:
    airports_path = tmp_path / "airports.csv"
    routes_path = tmp_path / "routes.csv"
    output_dir = tmp_path / "results"

    pd.DataFrame(
        [
            {
                "airport_id": 1,
                "name": "Alpha",
                "city": "A",
                "country": "Test",
                "iata": "AAA",
                "icao": "AAAA",
                "lat": 0.0,
                "lon": 0.0,
            },
            {
                "airport_id": 2,
                "name": "Beta",
                "city": "B",
                "country": "Test",
                "iata": "BBB",
                "icao": "BBBB",
                "lat": 1.0,
                "lon": 0.0,
            },
            {
                "airport_id": 3,
                "name": "Gamma",
                "city": "C",
                "country": "Test",
                "iata": "CCC",
                "icao": "CCCC",
                "lat": 0.0,
                "lon": 1.0,
            },
        ]
    ).to_csv(airports_path, index=False)
    pd.DataFrame(
        [
            {"source_iata": "AAA", "dest_iata": "BBB"},
            {"source_iata": "BBB", "dest_iata": "CCC"},
            {"source_iata": "CCC", "dest_iata": "AAA"},
        ]
    ).to_csv(routes_path, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "airports_csv": str(airports_path),
                "routes_csv": str(routes_path),
                "output_dir": str(output_dir),
                "k_nodes": 1,
                "m_edges": 1,
                "repetitions_R": 2,
                "budget_b": 1,
                "distance_km_max": 3000,
                "adaptive": True,
                "random_seed": 42,
                "hops_H": 2,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_parser_accepts_boolean_optional_flags():
    parser = build_parser()

    assert parser.parse_args(["--adaptive"]).adaptive is True
    assert parser.parse_args(["--no-adaptive"]).adaptive is False
    assert parser.parse_args(["--fast"]).fast is True
    assert parser.parse_args(["--no-fast"]).fast is False


def test_load_config_rejects_missing_required_keys(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("output_dir: results\n", encoding="utf-8")

    with pytest.raises(SimulationConfigError, match="missing keys"):
        load_config(path)


def test_random_cli_is_reproducible_and_writes_named_artifacts(tmp_path):
    config = _write_config(tmp_path)
    args = [
        "--config",
        str(config),
        "--attack",
        "random_nodes",
        "--R",
        "2",
        "--seed",
        "7",
        "--fast",
    ]

    assert main(args) == 0
    output = tmp_path / "results"
    canonical = json.loads((output / "attack_log.json").read_text(encoding="utf-8"))
    named = json.loads((output / "attack_log_random_nodes.json").read_text(encoding="utf-8"))
    baseline = json.loads((output / "baseline_report.json").read_text(encoding="utf-8"))

    assert canonical == named
    assert len(canonical) == 2
    assert baseline["n_nodes"] == 3

    assert main(args) == 0
    repeated = json.loads((output / "attack_log.json").read_text(encoding="utf-8"))
    assert repeated == canonical


def test_cli_returns_nonzero_for_missing_config(tmp_path, capsys):
    status = main(["--config", str(tmp_path / "missing.yaml")])

    assert status == 2
    assert "Configuration file not found" in capsys.readouterr().err


def test_atomic_json_is_standard_compliant(tmp_path):
    path = write_json_atomic(tmp_path / "result.json", {"edge": ("A", "B"), "aspl": float("inf")})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "edge": ["A", "B"],
        "aspl": None,
    }
    assert not list(tmp_path.glob("*.tmp"))
