"""Pure helpers for Streamlit UI state and runtime validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import networkx as nx

from src.ai.rag.index import index_status


def sanitize_data_path(filename: str, allowed_dir: str | Path) -> Path:
    """Resolve one direct CSV filename without allowing traversal or symlink escapes."""
    raw_name = filename.strip()
    if not raw_name or "/" in raw_name or "\\" in raw_name or Path(raw_name).is_absolute():
        raise ValueError("Select a CSV filename directly inside the data directory")
    if Path(raw_name).suffix.lower() != ".csv":
        raise ValueError("Only CSV data files are supported")

    root = Path(allowed_dir).resolve()
    resolved = (root / raw_name).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Access denied: {filename!r} is outside the data directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"Data file not found: {raw_name}")
    return resolved


def prefer_graph(graph: nx.DiGraph | None, fallback: nx.DiGraph) -> nx.DiGraph:
    """Return ``graph`` unless it is absent, preserving valid empty graphs."""
    return fallback if graph is None else graph


def graph_fingerprint(graph: nx.DiGraph) -> str:
    """Return a stable clustering key for topology and geographic coordinates."""
    digest = hashlib.sha256()

    def add(value: Any) -> None:
        payload = repr(value).encode("utf-8", errors="backslashreplace")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)

    add(graph.is_directed())
    for node, data in sorted(graph.nodes(data=True), key=lambda item: repr(item[0])):
        add("node")
        add(node)
        add(data.get("lat"))
        add(data.get("lon"))

    for source, destination in sorted(
        graph.edges(), key=lambda edge: (repr(edge[0]), repr(edge[1]))
    ):
        add("edge")
        add(source)
        add(destination)

    return digest.hexdigest()


def fresh_load_state(graph: nx.DiGraph) -> dict[str, Any]:
    """Build the scenario state installed after a successful dataset load."""
    return {
        "G": graph,
        "G_base": graph,
        "attack_log": [],
        "defense_log": [],
        "baseline_report": None,
        "hardened_nodes": set(),
        "defense_base_attack_step": 0,
        "H_attack": None,
        "H_defense": None,
        "atk_step": 0,
        "def_step": 0,
        "fast_mode": graph.number_of_nodes() > 800,
        "ai_result": None,
        "rag_result": None,
    }


def committed_scenario_state(graph: nx.DiGraph) -> dict[str, Any]:
    """Build derived state installed when replay becomes the new baseline."""
    return {
        "G_base": graph,
        "attack_log": [],
        "defense_log": [],
        "baseline_report": None,
        "hardened_nodes": set(),
        "defense_base_attack_step": 0,
        "H_attack": None,
        "H_defense": None,
        "atk_step": 0,
        "def_step": 0,
        "ai_result": None,
        "rag_result": None,
    }


def rag_index_readiness(index_path: str | Path) -> tuple[bool, str | None]:
    """Validate the files needed to load a persisted RAG vector store."""
    index_path = Path(index_path)
    metadata_path = index_path.with_suffix(".meta.json")

    if not index_path.is_file():
        return False, "Knowledge index is not built. Run: python -m src.ai.rag.index"
    if not metadata_path.is_file():
        return False, "Knowledge index metadata is missing. Rebuild the knowledge index."

    status = index_status(index_path, expected_model="text-embedding-3-small")
    if not status.ready:
        return False, f"Knowledge index is invalid: {status.message}. Rebuild the knowledge index."
    return True, None


def provider_error_message(error: Exception, service: str) -> str:
    """Map provider failures to actionable messages without echoing exception text."""
    error_name = type(error).__name__.lower()
    status_code = getattr(error, "status_code", None)

    if status_code in {401, 403} or "auth" in error_name or "permission" in error_name:
        return f"{service} authentication failed. Check the API key for the selected provider."
    if status_code == 429 or "ratelimit" in error_name or "rate_limit" in error_name:
        return f"{service} is rate-limited. Wait briefly and try again."
    if "timeout" in error_name:
        return f"{service} timed out. Try again in a moment."
    if "connection" in error_name:
        return f"{service} could not reach the provider. Check the network and try again."
    return f"{service} request failed. Check the provider settings and try again."


def safe_error_metadata(error: Exception) -> tuple[str, int | None]:
    """Return non-secret exception metadata suitable for operational logging."""
    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        status_code = None
    return type(error).__name__, status_code
