import pytest

from src.ai.rag.chunker import Chunk
from src.ai.rag.store import VectorStore


def _store():
    s = VectorStore()
    chunks = [
        Chunk("alpha", {"title": "A"}),
        Chunk("beta", {"title": "B"}),
        Chunk("gamma", {"title": "C"}),
    ]
    s.add(chunks, [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    return s


def test_search_returns_nearest_first():
    hits = _store().search([1.0, 0.0], k=2)
    assert hits[0].source["title"] == "A"
    assert hits[1].source["title"] == "C"
    assert hits[0].score >= hits[1].score


def test_search_empty_store_returns_empty():
    assert VectorStore().search([1.0, 0.0], k=3) == []


def test_add_length_mismatch_raises():
    with pytest.raises(ValueError):
        VectorStore().add([Chunk("x", {})], [[1.0], [2.0]])


def test_save_load_roundtrip(tmp_path):
    s = _store()
    p = tmp_path / "index.npz"
    s.save(p)
    loaded = VectorStore.load(p)
    assert len(loaded) == 3
    assert loaded.search([0.0, 1.0], k=1)[0].source["title"] == "B"
