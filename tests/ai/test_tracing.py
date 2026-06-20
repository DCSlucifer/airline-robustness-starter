import json

import pytest

from src.ai.tracing import TurnTrace, log_turn, trace_turn


def test_log_turn_appends_jsonl(tmp_path):
    path = tmp_path / "traces.jsonl"
    log_turn(TurnTrace(query="q1", tool_name="defend", arguments={"budget": 2}), path)
    log_turn(TurnTrace(query="q2", tool_name="edge_attack", arguments={"m": 3}), path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["query"] == "q1"
    assert rec0["tool_name"] == "defend"
    assert rec0["arguments"] == {"budget": 2}
    assert rec0["timestamp"]  # auto-filled


def test_trace_turn_records_latency_and_fields(tmp_path):
    path = tmp_path / "traces.jsonl"
    with trace_turn("what if?", path) as tr:
        tr.tool_name = "targeted_attack"
        tr.arguments = {"metric": "degree", "k": 5}

    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["tool_name"] == "targeted_attack"
    assert rec["arguments"]["k"] == 5
    assert rec["latency_ms"] >= 0.0
    assert rec["error"] is None


def test_trace_turn_records_error_and_reraises(tmp_path):
    path = tmp_path / "traces.jsonl"
    with pytest.raises(ValueError):
        with trace_turn("boom", path) as tr:
            tr.tool_name = "defend"
            raise ValueError("kaboom")

    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["tool_name"] == "defend"
    assert "kaboom" in rec["error"]
    assert rec["latency_ms"] >= 0.0
