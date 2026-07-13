"""Provider-swappable text embedders. Only OpenAIEmbedder talks to a provider SDK."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Protocol

__all__ = ["Embedder", "FakeEmbedder", "OpenAIEmbedder"]


class Embedder(Protocol):
    """Maps texts to fixed-length vectors. The RAG layer depends only on this."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests: a unit vector from each text's hash."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        raw: list[float] = []
        seed = text.encode("utf-8")
        i = 0
        while len(raw) < self.dim:
            digest = hashlib.sha256(seed + str(i).encode("utf-8")).digest()
            for b in digest:
                raw.append((b / 255.0) * 2.0 - 1.0)
                if len(raw) >= self.dim:
                    break
            i += 1
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


class OpenAIEmbedder:
    """OpenAI-backed embedder. The only embedder that imports `openai` (lazily)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        _client: Any = None,
    ):
        self.model = model
        if _client is not None:
            self._client = _client
        else:
            import openai  # lazy, so tests don't require the SDK

            self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]
