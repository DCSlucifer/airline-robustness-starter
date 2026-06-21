import json

from src.ai.rag.embedder import FakeEmbedder
from src.ai.rag.index import build_index, load_kb_docs
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
    assert len(VectorStore.load(idx)) == len(store)
