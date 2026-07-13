"""Validated in-memory cosine vector store with portable persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chunker import Chunk

__all__ = ["Hit", "IndexFormatError", "VectorStore"]

INDEX_SCHEMA_VERSION = 1


class IndexFormatError(ValueError):
    """Raised when persisted vector-index artifacts are missing or inconsistent."""


@dataclass(frozen=True)
class Hit:
    score: float
    text: str
    source: dict[str, Any]


class VectorStore:
    """Brute-force cosine search over normalized chunk embeddings."""

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._sources: list[dict[str, Any]] = []
        self._matrix: np.ndarray | None = None
        self.embedding_model: str | None = None
        self.corpus_sha256: str | None = None

    @property
    def dimension(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[1])

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] == 0:
            raise ValueError("vectors must form a non-empty two-dimensional matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("vectors must contain only finite numbers")
        if self._matrix is not None and matrix.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"embedding dimension mismatch: expected {self._matrix.shape[1]}, got {matrix.shape[1]}"
            )

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._matrix = matrix if self._matrix is None else np.vstack([self._matrix, matrix])
        self._texts.extend(chunk.text for chunk in chunks)
        self._sources.extend(dict(chunk.source) for chunk in chunks)

    def search(self, query_vector: list[float], k: int = 4) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._matrix.shape[1]:
            actual = query.shape[0] if query.ndim == 1 else tuple(query.shape)
            raise ValueError(
                f"query embedding dimension mismatch: expected {self._matrix.shape[1]}, got {actual}"
            )
        if not np.isfinite(query).all():
            raise ValueError("query embedding must contain only finite numbers")

        query = query / (np.linalg.norm(query) or 1.0)
        scores = self._matrix @ query
        top = np.argsort(-scores)[:k]
        return [Hit(float(scores[i]), self._texts[i], self._sources[i]) for i in top]

    def __len__(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    def save(
        self,
        path: str | Path,
        *,
        embedding_model: str | None = None,
        corpus_sha256: str | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.zeros((0, 0), np.float32)
        np.savez_compressed(str(destination), matrix=matrix)
        metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "count": len(self._texts),
            "dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "embedding_model": embedding_model,
            "corpus_sha256": corpus_sha256,
            "texts": self._texts,
            "sources": self._sources,
        }
        destination.with_suffix(".meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        self.embedding_model = embedding_model
        self.corpus_sha256 = corpus_sha256

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_model: str | None = None,
        require_nonempty: bool = False,
    ) -> VectorStore:
        index_path = Path(path)
        metadata_path = index_path.with_suffix(".meta.json")
        if not index_path.is_file() or not metadata_path.is_file():
            raise IndexFormatError(
                f"index requires both {index_path.name} and {metadata_path.name}"
            )

        try:
            with np.load(str(index_path), allow_pickle=False) as archive:
                matrix = np.asarray(archive["matrix"], dtype=np.float32)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise IndexFormatError(f"cannot read vector index: {exc}") from exc

        if not isinstance(metadata, dict):
            raise IndexFormatError("index metadata must be a JSON object")
        if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise IndexFormatError("unsupported index metadata schema")
        texts = metadata.get("texts")
        sources = metadata.get("sources")
        if not isinstance(texts, list) or not isinstance(sources, list):
            raise IndexFormatError("index metadata texts/sources must be lists")
        if not all(isinstance(text, str) for text in texts) or not all(
            isinstance(source, dict) for source in sources
        ):
            raise IndexFormatError("index metadata contains invalid text/source records")
        if matrix.ndim != 2 or matrix.shape[0] != len(texts) or len(texts) != len(sources):
            raise IndexFormatError("index matrix rows do not match metadata records")
        if metadata.get("count") != len(texts) or metadata.get("dimension") != matrix.shape[1]:
            raise IndexFormatError("index count/dimension metadata is inconsistent")
        if not np.isfinite(matrix).all():
            raise IndexFormatError("index matrix contains non-finite values")
        if require_nonempty and not texts:
            raise IndexFormatError("vector index is empty")

        model = metadata.get("embedding_model")
        if expected_model and model != expected_model:
            raise IndexFormatError(
                f"index embedding model mismatch: expected {expected_model!r}, found {model!r}"
            )

        store = cls()
        store._matrix = matrix if matrix.size else None
        store._texts = list(texts)
        store._sources = [dict(source) for source in sources]
        store.embedding_model = model if isinstance(model, str) else None
        corpus_sha256 = metadata.get("corpus_sha256")
        store.corpus_sha256 = corpus_sha256 if isinstance(corpus_sha256, str) else None
        return store
