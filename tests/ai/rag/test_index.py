import json

import pytest

from src.ai.rag.embedder import FakeEmbedder
from src.ai.rag.index import KnowledgeBaseError, build_index, index_status, load_kb_docs
from src.ai.rag.store import VectorStore


def _make_kb(tmp_path):
    (tmp_path / "a.md").write_text("Alpha doc.\n\nMore alpha.", encoding="utf-8")
    (tmp_path / "b.md").write_text("Beta doc.", encoding="utf-8")
    manifest = [
        {"title": "A", "slug": "a", "url": "ua", "revid": 1},
        {"title": "B", "slug": "b", "url": "ub", "revid": 2},
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_kb_docs_uses_manifest(tmp_path):
    _make_kb(tmp_path)
    assert {d["title"] for d in load_kb_docs(tmp_path)} == {"A", "B"}


def test_build_index_persists_searchable_store(tmp_path):
    _make_kb(tmp_path)
    idx = tmp_path / "index.npz"
    store = build_index(FakeEmbedder(dim=16), kb_dir=tmp_path, index_path=idx)
    assert len(store) >= 2
    assert idx.exists()
    loaded = VectorStore.load(idx, require_nonempty=True)
    assert len(loaded) == len(store)
    assert loaded.dimension == 16
    assert loaded.embedding_model.endswith("FakeEmbedder:16")
    assert loaded.corpus_sha256
    assert index_status(idx).ready is True


def test_load_kb_docs_fails_fast_without_nonempty_manifest(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="manifest not found"):
        load_kb_docs(tmp_path)

    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="at least one article"):
        load_kb_docs(tmp_path)


def test_load_kb_docs_requires_every_manifest_document(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"title": "A", "slug": "a", "url": "u", "revid": 1}]),
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeBaseError, match="document not found"):
        load_kb_docs(tmp_path)


def test_build_index_does_not_create_empty_artifacts(tmp_path):
    index_path = tmp_path / "index.npz"

    with pytest.raises(KnowledgeBaseError):
        build_index(FakeEmbedder(), kb_dir=tmp_path, index_path=index_path)

    assert not index_path.exists()
    assert index_status(index_path).ready is False
