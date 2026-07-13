"""Build and validate a persisted vector index from a cached knowledge-base snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import chunk_document
from .corpus import KB_DIR
from .embedder import Embedder, OpenAIEmbedder
from .store import IndexFormatError, VectorStore

__all__ = [
    "INDEX_PATH",
    "IndexStatus",
    "KnowledgeBaseError",
    "load_kb_docs",
    "build_store",
    "build_index",
    "index_status",
]

INDEX_PATH = KB_DIR / "index.npz"


class KnowledgeBaseError(ValueError):
    """Raised when cached corpus files are missing or inconsistent."""


@dataclass(frozen=True)
class IndexStatus:
    ready: bool
    message: str
    chunk_count: int = 0
    embedding_model: str | None = None


def load_kb_docs(kb_dir: Path = KB_DIR) -> list[dict[str, Any]]:
    """Load exactly the Markdown documents declared by a non-empty manifest."""
    root = Path(kb_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise KnowledgeBaseError(f"knowledge manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeBaseError(f"cannot read knowledge manifest: {exc}") from exc
    if not isinstance(manifest, list) or not manifest:
        raise KnowledgeBaseError("knowledge manifest must contain at least one article")

    docs: list[dict[str, Any]] = []
    slugs: set[str] = set()
    for position, item in enumerate(manifest, start=1):
        if not isinstance(item, dict):
            raise KnowledgeBaseError(f"manifest entry {position} must be an object")
        missing = {"title", "slug", "url", "revid"} - item.keys()
        if missing:
            raise KnowledgeBaseError(
                f"manifest entry {position} missing keys: {', '.join(sorted(missing))}"
            )
        slug = str(item["slug"])
        if not slug or slug in slugs:
            raise KnowledgeBaseError(f"manifest contains an empty or duplicate slug: {slug!r}")
        slugs.add(slug)

        markdown_path = root / f"{slug}.md"
        if not markdown_path.is_file():
            raise KnowledgeBaseError(f"knowledge document not found: {markdown_path}")
        text = markdown_path.read_text(encoding="utf-8").strip()
        if not text:
            raise KnowledgeBaseError(f"knowledge document is empty: {markdown_path}")
        docs.append(
            {
                "title": str(item["title"]),
                "url": str(item["url"]),
                "revid": item["revid"],
                "text": text,
            }
        )
    return docs


def _embedder_id(embedder: Embedder) -> str:
    model = getattr(embedder, "model", None)
    if isinstance(model, str) and model:
        return model
    dimension = getattr(embedder, "dim", None)
    suffix = f":{dimension}" if dimension is not None else ""
    return f"{type(embedder).__module__}.{type(embedder).__qualname__}{suffix}"


def _corpus_sha256(docs: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for doc in docs:
        payload = json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_store(
    docs: list[dict[str, Any]], embedder: Embedder, max_chars: int = 800
) -> VectorStore:
    if not docs:
        raise KnowledgeBaseError("cannot build an index from an empty corpus")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    store = VectorStore()
    for doc in docs:
        chunks = chunk_document(
            doc["text"],
            {"title": doc["title"], "url": doc["url"], "revid": doc.get("revid")},
            max_chars=max_chars,
        )
        if chunks:
            store.add(chunks, embedder.embed([chunk.text for chunk in chunks]))
    if not store:
        raise KnowledgeBaseError("corpus produced no indexable chunks")
    return store


def build_index(
    embedder: Embedder,
    kb_dir: Path = KB_DIR,
    index_path: Path = INDEX_PATH,
) -> VectorStore:
    docs = load_kb_docs(kb_dir)
    store = build_store(docs, embedder)
    store.save(
        index_path,
        embedding_model=_embedder_id(embedder),
        corpus_sha256=_corpus_sha256(docs),
    )
    return store


def index_status(
    index_path: str | Path = INDEX_PATH, *, expected_model: str | None = None
) -> IndexStatus:
    """Return a user-facing readiness result without raising on bad local artifacts."""
    try:
        store = VectorStore.load(index_path, expected_model=expected_model, require_nonempty=True)
    except (IndexFormatError, OSError, ValueError) as exc:
        return IndexStatus(False, str(exc))
    return IndexStatus(
        True,
        "Knowledge index ready",
        chunk_count=len(store),
        embedding_model=store.embedding_model,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", type=Path, default=KB_DIR)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--model", default="text-embedding-3-small")
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("error: OPENAI_API_KEY is required to build the embedding index", file=sys.stderr)
        return 2
    try:
        store = build_index(
            OpenAIEmbedder(api_key=api_key, model=args.model),
            kb_dir=args.kb_dir,
            index_path=args.index,
        )
    except (OSError, KnowledgeBaseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Indexed {len(store)} chunks -> {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
