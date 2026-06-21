"""Build a VectorStore from the cached KB: load -> chunk -> embed -> persist."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

from .chunker import chunk_document
from .corpus import KB_DIR
from .embedder import Embedder
from .store import VectorStore

__all__ = ["INDEX_PATH", "load_kb_docs", "build_store", "build_index"]

INDEX_PATH = KB_DIR / "index.npz"


def load_kb_docs(kb_dir: Path = KB_DIR) -> List[Dict[str, Any]]:
    """Read cached .md files into {title, url, revid, text} dicts using the manifest metadata."""
    kb_dir = Path(kb_dir)
    manifest_path = kb_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    by_slug = {m["slug"]: m for m in manifest}
    docs: List[Dict[str, Any]] = []
    for md in sorted(kb_dir.glob("*.md")):
        meta = by_slug.get(md.stem, {"title": md.stem, "url": "", "revid": None})
        docs.append(
            {
                "title": meta["title"],
                "url": meta.get("url", ""),
                "revid": meta.get("revid"),
                "text": md.read_text(encoding="utf-8"),
            }
        )
    return docs


def build_store(docs: List[Dict[str, Any]], embedder: Embedder, max_chars: int = 800) -> VectorStore:
    store = VectorStore()
    for doc in docs:
        chunks = chunk_document(doc["text"], {"title": doc["title"], "url": doc["url"]}, max_chars=max_chars)
        if not chunks:
            continue
        store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store


def build_index(embedder: Embedder, kb_dir: Path = KB_DIR, index_path: Path = INDEX_PATH) -> VectorStore:
    store = build_store(load_kb_docs(kb_dir), embedder)
    store.save(index_path)
    return store


def main() -> None:  # pragma: no cover - needs OPENAI_API_KEY
    import os
    from .embedder import OpenAIEmbedder

    store = build_index(OpenAIEmbedder(api_key=os.environ.get("OPENAI_API_KEY")))
    print(f"Indexed {len(store)} chunks -> {INDEX_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()
