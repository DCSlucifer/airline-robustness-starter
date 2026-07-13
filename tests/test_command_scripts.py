"""Shell-facing helper scripts must propagate failures to the caller."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import check_load
import run_demo


def test_check_load_returns_nonzero_for_missing_config(tmp_path, capsys):
    status = check_load.main(["--config", str(tmp_path / "missing.yaml")])

    assert status == 2
    assert "Load check failed" in capsys.readouterr().err


def test_demo_launcher_propagates_streamlit_exit_code(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(run_demo.subprocess, "run", fake_run)

    assert run_demo.main(["--port", "8765"]) == 17
    assert "--server.port=8765" in captured["command"]
    assert "--server.address=127.0.0.1" in captured["command"]
    assert captured["kwargs"]["check"] is False


def test_demo_launcher_rejects_invalid_port():
    with pytest.raises(SystemExit):
        run_demo.main(["--port", "70000"])
