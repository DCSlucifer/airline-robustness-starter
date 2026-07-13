import json

import pytest

from src.ai.rag.chunker import Chunk
from src.ai.rag.store import IndexFormatError, VectorStore


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


def test_store_rejects_invalid_vector_shapes_and_dimensions():
    store = VectorStore()

    with pytest.raises(ValueError, match="two-dimensional"):
        store.add([Chunk("x", {})], [[]])

    store.add([Chunk("x", {})], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add([Chunk("y", {})], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="query embedding dimension mismatch"):
        store.search([1.0], k=1)
    with pytest.raises(ValueError, match="k must be positive"):
        store.search([1.0, 0.0], k=0)


def test_load_requires_both_artifacts(tmp_path):
    path = tmp_path / "index.npz"
    path.write_bytes(b"not-an-index")

    with pytest.raises(IndexFormatError, match="requires both"):
        VectorStore.load(path)


def test_load_validates_model_and_nonempty_contract(tmp_path):
    path = tmp_path / "index.npz"
    store = _store()
    store.save(path, embedding_model="model-a", corpus_sha256="abc")

    loaded = VectorStore.load(path, expected_model="model-a", require_nonempty=True)
    assert loaded.embedding_model == "model-a"
    assert loaded.corpus_sha256 == "abc"

    with pytest.raises(IndexFormatError, match="model mismatch"):
        VectorStore.load(path, expected_model="model-b")

    empty_path = tmp_path / "empty.npz"
    VectorStore().save(empty_path, embedding_model="model-a")
    with pytest.raises(IndexFormatError, match="empty"):
        VectorStore.load(empty_path, require_nonempty=True)


def test_load_rejects_inconsistent_metadata(tmp_path):
    path = tmp_path / "index.npz"
    _store().save(path)
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["count"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexFormatError, match="inconsistent"):
        VectorStore.load(path)
