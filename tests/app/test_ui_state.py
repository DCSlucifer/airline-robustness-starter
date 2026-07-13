import networkx as nx
import pytest

from src.ai.rag.chunker import Chunk
from src.ai.rag.store import VectorStore
from src.app.ui_state import (
    committed_scenario_state,
    fresh_load_state,
    graph_fingerprint,
    prefer_graph,
    provider_error_message,
    rag_index_readiness,
    safe_error_metadata,
    sanitize_data_path,
)


def _graph(prefix: str, *, coordinate_offset: float = 0.0) -> nx.DiGraph:
    graph = nx.DiGraph()
    for index in range(3):
        graph.add_node(
            f"{prefix}{index}",
            lat=coordinate_offset + index,
            lon=coordinate_offset - index,
        )
    graph.add_edges_from(
        [
            (f"{prefix}0", f"{prefix}1"),
            (f"{prefix}1", f"{prefix}2"),
            (f"{prefix}2", f"{prefix}0"),
        ]
    )
    return graph


def test_prefer_graph_preserves_an_empty_graph() -> None:
    empty = nx.DiGraph()
    fallback = _graph("fallback")

    assert prefer_graph(empty, fallback) is empty
    assert prefer_graph(None, fallback) is fallback


def test_graph_fingerprint_distinguishes_equal_size_graphs_and_coordinates() -> None:
    first = _graph("A")
    different_nodes = _graph("B")
    different_coordinates = _graph("A", coordinate_offset=10.0)

    assert first.number_of_nodes() == different_nodes.number_of_nodes()
    assert first.number_of_edges() == different_nodes.number_of_edges()
    assert graph_fingerprint(first) != graph_fingerprint(different_nodes)
    assert graph_fingerprint(first) != graph_fingerprint(different_coordinates)
    assert graph_fingerprint(first) == graph_fingerprint(first.copy())


def test_scenario_state_resets_derived_results_and_selects_fast_mode() -> None:
    large = nx.DiGraph()
    large.add_nodes_from(range(801))

    loaded = fresh_load_state(large)
    committed = committed_scenario_state(nx.DiGraph())

    assert loaded["fast_mode"] is True
    assert loaded["ai_result"] is None
    assert loaded["rag_result"] is None
    assert loaded["H_attack"] is None
    assert loaded["H_defense"] is None
    assert committed["G_base"].number_of_nodes() == 0
    assert committed["ai_result"] is None
    assert committed["rag_result"] is None


def test_rag_index_readiness_requires_valid_metadata(tmp_path) -> None:
    index_path = tmp_path / "index.npz"

    ready, message = rag_index_readiness(index_path)
    assert ready is False
    assert "not built" in message

    store = VectorStore()
    store.add([Chunk("text", {"title": "Source"})], [[1.0, 0.0]])
    store.save(index_path, embedding_model="text-embedding-3-small")
    metadata_path = index_path.with_suffix(".meta.json")
    metadata_path.unlink()
    ready, message = rag_index_readiness(index_path)
    assert ready is False
    assert "metadata is missing" in message

    store.save(index_path, embedding_model="text-embedding-3-small")
    metadata_path.write_text("not json", encoding="utf-8")
    ready, message = rag_index_readiness(index_path)
    assert ready is False
    assert "Knowledge index is invalid" in message

    store.save(index_path, embedding_model="text-embedding-3-small")
    assert rag_index_readiness(index_path) == (True, None)


def test_provider_errors_do_not_echo_exception_text() -> None:
    secret = "sk-do-not-log"

    class AuthenticationError(Exception):
        status_code = 401

    error = AuthenticationError(f"invalid credential: {secret}")
    message = provider_error_message(error, "Assistant")
    error_type, status_code = safe_error_metadata(error)

    assert "authentication failed" in message
    assert secret not in message
    assert (error_type, status_code) == ("AuthenticationError", 401)
    assert secret not in f"{error_type} {status_code}"


def test_sanitize_data_path_accepts_only_direct_csv_files(tmp_path) -> None:
    csv_path = tmp_path / "airports.csv"
    csv_path.write_text("iata,lat,lon\nAAA,0,0\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not data", encoding="utf-8")

    assert sanitize_data_path(" airports.csv ", tmp_path) == csv_path.resolve()

    for invalid in ("../airports.csv", r"..\airports.csv", "notes.txt", str(csv_path.resolve())):
        with pytest.raises(ValueError):
            sanitize_data_path(invalid, tmp_path)

    with pytest.raises(FileNotFoundError):
        sanitize_data_path("missing.csv", tmp_path)
