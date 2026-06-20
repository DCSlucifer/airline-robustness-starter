"""Lightweight observability: append one JSON record per assistant turn.

Each turn is recorded to a JSONL file (query, tool, arguments, latency, optional
tokens/cost, error). This is the observability/LLMOps layer for the assistant.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

__all__ = ["TurnTrace", "log_turn", "trace_turn", "DEFAULT_TRACE_PATH"]

DEFAULT_TRACE_PATH = Path("outputs/ai_traces.jsonl")


@dataclass
class TurnTrace:
    """One assistant turn's observability record."""

    query: str
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = ""


def log_turn(trace: TurnTrace, path: Path = DEFAULT_TRACE_PATH) -> None:
    """Append one trace record as a single JSON line, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = asdict(trace)
    if not record.get("timestamp"):
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


@contextmanager
def trace_turn(query: str, path: Path = DEFAULT_TRACE_PATH) -> Iterator[TurnTrace]:
    """Time a turn and write its trace on exit, recording and re-raising any error."""
    trace = TurnTrace(query=query)
    start = time.perf_counter()
    try:
        yield trace
    except Exception as exc:  # noqa: BLE001 - record every failure, then re-raise
        trace.error = repr(exc)
        raise
    finally:
        trace.latency_ms = (time.perf_counter() - start) * 1000.0
        log_turn(trace, path)
