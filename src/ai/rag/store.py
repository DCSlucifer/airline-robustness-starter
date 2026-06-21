"""In-memory vector store: numpy cosine similarity with persistence."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .chunker import Chunk

__all__ = ["Hit", "VectorStore"]


@dataclass
class Hit:
    score: float
    text: str
    source: Dict[str, Any]


class VectorStore:
    """Brute-force cosine search over chunk embeddings. Sufficient for a small corpus."""

    def __init__(self):
        self._texts: List[str] = []
        self._sources: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim), L2-normalized

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            return
        mat = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        self._matrix = mat if self._matrix is None else np.vstack([self._matrix, mat])
        self._texts.extend(c.text for c in chunks)
        self._sources.extend(dict(c.source) for c in chunks)

    def search(self, query_vector: List[float], k: int = 4) -> List[Hit]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._matrix @ q
        top = np.argsort(-scores)[:k]
        return [Hit(float(scores[i]), self._texts[i], self._sources[i]) for i in top]

    def __len__(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.zeros((0, 0), np.float32)
        np.savez(str(path), matrix=matrix)
        meta = {"texts": self._texts, "sources": self._sources}
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VectorStore":
        path = Path(path)
        store = cls()
        matrix = np.load(str(path))["matrix"]
        store._matrix = matrix if matrix.size else None
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        store._texts = meta["texts"]
        store._sources = meta["sources"]
        return store
