# RAG Resilience Advisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, citation-backed RAG "Resilience Advisor" to the airline-robustness app, retrieving from a cached Wikipedia corpus.

**Architecture:** A self-contained `src/ai/rag/` package: an `Embedder` interface (FakeEmbedder for tests, OpenAIEmbedder for real use), a numpy cosine `VectorStore`, a reproducible Wikipedia corpus fetcher, an index builder, and an `advisor.answer()` that retrieves top-k chunks and synthesizes a cited answer via the existing `LLMClient` (extended with a `chat` method). A retrieval `recall@k` eval and a Streamlit panel complete it.

**Tech Stack:** Python 3.10+, numpy, openai (embeddings), Pydantic, Streamlit. Stdlib `urllib` for fetching. No new dependency.

## Global Constraints

- Python 3.10+; type hints + module docstrings consistent with the existing codebase (see `src/ai/llm_client.py`).
- No new dependency: embeddings via `openai` (present), vectors via `numpy` (present), fetch via stdlib `urllib.request`. No FAISS.
- Provider isolation: only `OpenAIEmbedder` imports `openai`, lazily inside `__init__`; an injectable `_client` makes it testable without the SDK.
- All non-network logic is offline-testable via `FakeEmbedder` + `FakeLLMClient.chat` (no API key, no network in tests).
- RAG code lives only in `src/ai/rag/`; do NOT modify existing simulation modules. The only existing file modified is `src/ai/llm_client.py` (additive `chat` method) and `src/app/streamlit_app.py` (additive panel).
- Citations must be grounded in retrieved chunks; never fabricated.
- TDD: write failing test → confirm fail → minimal impl → confirm pass → commit. Do NOT `git push`.

---

### Task R1: Embedder interface + FakeEmbedder + OpenAIEmbedder

**Files:**
- Create: `src/ai/rag/__init__.py`
- Create: `src/ai/rag/embedder.py`
- Create: `tests/ai/rag/__init__.py`
- Test: `tests/ai/rag/test_embedder.py`

**Interfaces:**
- Produces: `Embedder` Protocol (`embed(texts: list[str]) -> list[list[float]]`); `FakeEmbedder(dim=16)`; `OpenAIEmbedder(api_key=None, model="text-embedding-3-small", _client=None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/rag/__init__.py` (empty), then `tests/ai/rag/test_embedder.py`:

```python
from src.ai.rag.embedder import FakeEmbedder, OpenAIEmbedder


def test_fake_embedder_is_deterministic_and_fixed_dim():
    emb = FakeEmbedder(dim=16)
    assert emb.embed(["hello"]) == emb.embed(["hello"])
    assert len(emb.embed(["hello"])[0]) == 16


def test_fake_embedder_different_texts_differ():
    va, vb = FakeEmbedder(dim=16).embed(["alpha", "beta"])
    assert va != vb


def test_fake_embedder_unit_norm():
    v = FakeEmbedder(dim=16).embed(["x"])[0]
    assert abs(sum(c * c for c in v) ** 0.5 - 1.0) < 1e-9


class _StubEmbItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _StubEmbResponse:
    def __init__(self, data):
        self.data = data


class _StubEmbeddings:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _StubOpenAI:
    def __init__(self, response):
        self.embeddings = _StubEmbeddings(response)


def test_openai_embedder_parses_response_and_uses_model():
    response = _StubEmbResponse([_StubEmbItem([0.1, 0.2]), _StubEmbItem([0.3, 0.4])])
    emb = OpenAIEmbedder(_client=_StubOpenAI(response))
    assert emb.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert emb._client.embeddings.last_kwargs["model"] == "text-embedding-3-small"
    assert emb._client.embeddings.last_kwargs["input"] == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/rag/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.rag'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/rag/__init__.py`:

```python
"""Retrieval-Augmented Generation: the Resilience Advisor."""
```

Create `src/ai/rag/embedder.py`:

```python
"""Provider-swappable text embedders. Only OpenAIEmbedder talks to a provider SDK."""
from __future__ import annotations
import hashlib
import math
from typing import Any, List, Optional, Protocol

__all__ = ["Embedder", "FakeEmbedder", "OpenAIEmbedder"]


class Embedder(Protocol):
    """Maps texts to fixed-length vectors. The RAG layer depends only on this."""

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests: a unit vector from each text's hash."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> List[float]:
        raw: List[float] = []
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
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        _client: Any = None,
    ):
        self.model = model
        if _client is not None:
            self._client = _client
        else:
            import openai  # lazy, so tests don't require the SDK
            self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/rag/test_embedder.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/rag/__init__.py src/ai/rag/embedder.py tests/ai/rag/__init__.py tests/ai/rag/test_embedder.py
git commit -m "feat(rag): add Embedder interface with Fake and OpenAI implementations"
```

---

### Task R2: Chunker + VectorStore

**Files:**
- Create: `src/ai/rag/chunker.py`
- Create: `src/ai/rag/store.py`
- Test: `tests/ai/rag/test_chunker.py`, `tests/ai/rag/test_store.py`

**Interfaces:**
- Produces: `Chunk(text: str, source: dict)`; `chunk_document(text, source, max_chars=800) -> list[Chunk]`;
  `Hit(score: float, text: str, source: dict)`; `VectorStore` with `.add(chunks, vectors)`,
  `.search(query_vector, k=4) -> list[Hit]`, `len()`, `.save(path)`, `VectorStore.load(path)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/ai/rag/test_chunker.py`:

```python
from src.ai.rag.chunker import chunk_document, Chunk


def test_chunk_preserves_source_metadata():
    chunks = chunk_document("para one\n\npara two", {"title": "T", "url": "u"}, max_chars=800)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source["title"] == "T"


def test_chunk_packs_until_max_chars():
    text = "\n\n".join(["x" * 300, "y" * 300, "z" * 300])
    chunks = chunk_document(text, {}, max_chars=700)
    assert len(chunks) == 2


def test_chunk_empty_text_returns_empty():
    assert chunk_document("   ", {}) == []
```

Create `tests/ai/rag/test_store.py`:

```python
import pytest

from src.ai.rag.chunker import Chunk
from src.ai.rag.store import VectorStore


def _store():
    s = VectorStore()
    chunks = [Chunk("alpha", {"title": "A"}), Chunk("beta", {"title": "B"}), Chunk("gamma", {"title": "C"})]
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ai/rag/test_chunker.py tests/ai/rag/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.ai.rag.chunker` / `src.ai.rag.store`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/rag/chunker.py`:

```python
"""Split documents into retrievable chunks that retain their source metadata."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = ["Chunk", "chunk_document"]


@dataclass
class Chunk:
    text: str
    source: Dict[str, Any] = field(default_factory=dict)


def chunk_document(text: str, source: Dict[str, Any], max_chars: int = 800) -> List[Chunk]:
    """Split on blank lines (paragraphs), packing paragraphs up to max_chars per chunk."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Chunk] = []
    buf = ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 2 > max_chars:
            chunks.append(Chunk(text=buf, source=dict(source)))
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(Chunk(text=buf, source=dict(source)))
    return chunks
```

Create `src/ai/rag/store.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ai/rag/test_chunker.py tests/ai/rag/test_store.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/rag/chunker.py src/ai/rag/store.py tests/ai/rag/test_chunker.py tests/ai/rag/test_store.py
git commit -m "feat(rag): add paragraph chunker and numpy cosine vector store"
```

---

### Task R3: Wikipedia corpus fetcher

**Files:**
- Create: `src/ai/rag/corpus.py`
- Test: `tests/ai/rag/test_corpus.py`

**Interfaces:**
- Produces: `Article(title: str)`; `WIKI_ARTICLES: list[Article]`; `KB_DIR: Path`;
  `parse_extract(api_json: dict) -> dict`; `fetch_corpus(articles=None, kb_dir=KB_DIR, fetcher=...) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/rag/test_corpus.py`:

```python
import json

from src.ai.rag.corpus import fetch_corpus, parse_extract, Article


def _api_json(title, text, revid):
    return {"query": {"pages": {"1": {"title": title, "extract": text, "pageid": 1, "lastrevid": revid}}}}


def test_parse_extract_pulls_fields():
    parsed = parse_extract(_api_json("Network science", "Body text.", 999))
    assert parsed["title"] == "Network science"
    assert parsed["text"] == "Body text."
    assert parsed["revid"] == 999


def test_fetch_corpus_caches_files_and_manifest(tmp_path):
    arts = [Article("Network science"), Article("Centrality")]
    responses = {
        "Network+science": json.dumps(_api_json("Network science", "Net body.", 11)),
        "Centrality": json.dumps(_api_json("Centrality", "Cent body.", 22)),
    }

    def fake_fetcher(url):
        for key, body in responses.items():
            if key in url:
                return body
        raise AssertionError(f"unexpected url: {url}")

    manifest = fetch_corpus(arts, kb_dir=tmp_path, fetcher=fake_fetcher)
    assert len(manifest) == 2
    assert (tmp_path / "manifest.json").exists()
    body = (tmp_path / "network-science.md").read_text(encoding="utf-8")
    assert "Net body." in body
    assert "oldid=11" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/rag/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.rag.corpus'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/rag/corpus.py`:

```python
"""Wikipedia corpus: an article manifest and an offline-caching, reproducible fetcher.

`fetch_corpus` records each article's resolved revision id and writes plain-text extracts to
`data/kb/`. The cached files are committed, so the index rebuilds offline and deterministically.
"""
from __future__ import annotations
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Article", "WIKI_ARTICLES", "KB_DIR", "parse_extract", "fetch_corpus"]

KB_DIR = Path("data/kb")
WIKI_API = "https://en.wikipedia.org/w/api.php"


@dataclass(frozen=True)
class Article:
    title: str


WIKI_ARTICLES: List[Article] = [
    Article("Network science"),
    Article("Centrality"),
    Article("Betweenness centrality"),
    Article("Scale-free network"),
    Article("Robustness of complex networks"),
    Article("Spoke–hub distribution paradigm"),
    Article("Airline hub"),
    Article("Air travel disruption after the 2010 Eyjafjallajökull eruption"),
    Article("Impact of the COVID-19 pandemic on aviation"),
    Article("Flight cancellation and delay"),
]


def _api_url(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": "1",
        "format": "json",
        "redirects": "1",
        "titles": title,
    }
    return WIKI_API + "?" + urllib.parse.urlencode(params)


def _default_fetcher(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed Wikipedia host
        return resp.read().decode("utf-8")


def _slug(title: str) -> str:
    return title.lower().replace(" ", "-").replace("/", "-")


def parse_extract(api_json: Dict[str, Any]) -> Dict[str, Any]:
    """Pull title, plain text, pageid, revid from a Wikipedia Action API response."""
    page = next(iter(api_json["query"]["pages"].values()))
    return {
        "title": page["title"],
        "text": page.get("extract", ""),
        "pageid": page.get("pageid"),
        "revid": page.get("lastrevid"),
    }


def fetch_corpus(
    articles: Optional[List[Article]] = None,
    kb_dir: Path = KB_DIR,
    fetcher: Callable[[str], str] = _default_fetcher,
) -> List[Dict[str, Any]]:
    """Fetch each article's plain-text extract, cache it to kb_dir, return manifest entries."""
    articles = articles if articles is not None else WIKI_ARTICLES
    kb_dir = Path(kb_dir)
    kb_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for art in articles:
        parsed = parse_extract(json.loads(fetcher(_api_url(art.title))))
        revid = parsed["revid"]
        url = (
            "https://en.wikipedia.org/w/index.php?"
            + urllib.parse.urlencode({"title": parsed["title"], "oldid": revid})
        )
        slug = _slug(parsed["title"])
        header = (
            f"# {parsed['title']}\n\n"
            f"<!-- source: {url} | revid: {revid} | fetched: {datetime.now(timezone.utc).isoformat()} -->\n\n"
        )
        (kb_dir / f"{slug}.md").write_text(header + parsed["text"], encoding="utf-8")
        manifest.append({"title": parsed["title"], "slug": slug, "url": url, "revid": revid})
    (kb_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:  # pragma: no cover - network, one-shot
    entries = fetch_corpus()
    print(f"Cached {len(entries)} articles to {KB_DIR}/")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/rag/test_corpus.py -v`
Expected: PASS (2 passed)

Note: the manifest `url` uses `urlencode`, which renders the revision as `oldid=11` — the test asserts this substring.

- [ ] **Step 5: Commit**

```bash
git add src/ai/rag/corpus.py tests/ai/rag/test_corpus.py
git commit -m "feat(rag): add reproducible Wikipedia corpus fetcher (stdlib urllib)"
```

---

### Task R4: Index builder

**Files:**
- Create: `src/ai/rag/index.py`
- Test: `tests/ai/rag/test_index.py`

**Interfaces:**
- Consumes: `chunk_document`, `VectorStore`, `Embedder`, `KB_DIR`.
- Produces: `INDEX_PATH: Path`; `load_kb_docs(kb_dir=KB_DIR) -> list[dict]`;
  `build_store(docs, embedder, max_chars=800) -> VectorStore`; `build_index(embedder, kb_dir=KB_DIR, index_path=INDEX_PATH) -> VectorStore`.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/rag/test_index.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/rag/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.rag.index'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/rag/index.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/rag/test_index.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/rag/index.py tests/ai/rag/test_index.py
git commit -m "feat(rag): add index builder (kb -> chunk -> embed -> persist)"
```

---

### Task R5: LLMClient.chat + Advisor

**Files:**
- Modify: `src/ai/llm_client.py` (additive `chat` on the Protocol + 3 classes; add `chat_response` to FakeLLMClient)
- Create: `src/ai/rag/advisor.py`
- Modify: `tests/ai/test_llm_client.py` (add one chat test), `tests/ai/test_openai_client.py` (add one chat test)
- Test: `tests/ai/rag/test_advisor.py`

**Interfaces:**
- Consumes: `LLMClient`, `Embedder`, `VectorStore`, `Hit`.
- Produces: `LLMClient.chat(system: str, user: str) -> str` on all clients;
  `RagAnswer(question, answer, sources)`; `ADVISOR_SYSTEM_PROMPT`; `render_context(hits) -> str`;
  `answer(question, client, embedder, store, k=4) -> RagAnswer`.

- [ ] **Step 1: Write the failing tests**

Create `tests/ai/rag/test_advisor.py`:

```python
from src.ai.schemas import ToolSelection
from src.ai.llm_client import FakeLLMClient
from src.ai.rag.embedder import FakeEmbedder
from src.ai.rag.store import VectorStore, Hit
from src.ai.rag.chunker import Chunk
from src.ai.rag.advisor import answer, RagAnswer, render_context


def _client(text):
    return FakeLLMClient(
        selection=ToolSelection(name="x", arguments={}),
        explanation="n/a",
        chat_response=text,
    )


def _store(embedder):
    s = VectorStore()
    chunks = [
        Chunk("Betweenness measures bridge importance.", {"title": "Betweenness centrality", "url": "u1"}),
        Chunk("The 2010 ash cloud closed European airspace.", {"title": "Eyjafjallajokull", "url": "u2"}),
    ]
    s.add(chunks, embedder.embed([c.text for c in chunks]))
    return s


def test_answer_returns_grounded_sources():
    emb = FakeEmbedder(dim=16)
    res = answer("what is betweenness?", _client("Bridges matter [1]."), emb, _store(emb), k=2)
    assert isinstance(res, RagAnswer)
    assert "[1]" in res.answer
    assert "Betweenness centrality" in {s["title"] for s in res.sources}


def test_answer_dedupes_sources():
    emb = FakeEmbedder(dim=16)
    s = VectorStore()
    s.add([Chunk("a", {"title": "T", "url": "u"}), Chunk("b", {"title": "T", "url": "u"})], emb.embed(["a", "b"]))
    res = answer("q", _client("ans [1]"), emb, s, k=2)
    assert len(res.sources) == 1


def test_answer_empty_store_no_fabrication():
    res = answer("q", _client("unused"), FakeEmbedder(dim=16), VectorStore(), k=3)
    assert res.sources == []
    assert "No relevant sources" in res.answer


def test_render_context_numbers_sources():
    ctx = render_context([Hit(0.9, "txt", {"title": "T"})])
    assert "[1]" in ctx and "T" in ctx
```

Append to `tests/ai/test_llm_client.py` (the stub classes `_StubResponse`, `_StubBlock`, `_StubAnthropic` already exist in that file):

```python
def test_claude_client_chat_returns_text():
    response = _StubResponse([_StubBlock("text", text="Hello from chat.")])
    client = ClaudeClient(_client=_StubAnthropic(response))
    assert client.chat("sys", "user") == "Hello from chat."
```

Append to `tests/ai/test_openai_client.py` (the stub classes already exist there):

```python
def test_openai_client_chat_returns_content():
    response = _StubCompletion([_StubChoice(_StubMessage(content="Hello chat."))])
    client = OpenAIClient(_client=_StubOpenAI(response))
    assert client.chat("sys", "user") == "Hello chat."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ai/rag/test_advisor.py tests/ai/test_llm_client.py tests/ai/test_openai_client.py -v`
Expected: FAIL — `ModuleNotFoundError` for `src.ai.rag.advisor`, and `AttributeError`/`TypeError` for the new `chat` / `chat_response` not existing yet.

- [ ] **Step 3a: Extend `src/ai/llm_client.py` (additive)**

In the `LLMClient` Protocol, add a third method after `explain`:

```python
    def chat(self, system: str, user: str) -> str: ...
```

In `FakeLLMClient.__init__`, add a parameter and store it (keep existing params):

```python
    def __init__(self, selection: ToolSelection, explanation: str = "(no explanation)",
                 chat_response: str = "(chat)"):
        self._selection = selection
        self._explanation = explanation
        self._chat_response = chat_response
```

Add this method to `FakeLLMClient`:

```python
    def chat(self, system: str, user: str) -> str:
        return self._chat_response
```

Add this method to `ClaudeClient`:

```python
    def chat(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.explainer_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return next(b.text for b in resp.content if b.type == "text")
```

Add this method to `OpenAIClient`:

```python
    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.explainer_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
```

- [ ] **Step 3b: Create `src/ai/rag/advisor.py`**

```python
"""Resilience Advisor: retrieve relevant chunks and answer with grounded citations."""
from __future__ import annotations
from typing import Any, Dict, List

from pydantic import BaseModel

from ..llm_client import LLMClient
from .embedder import Embedder
from .store import Hit, VectorStore

__all__ = ["RagAnswer", "ADVISOR_SYSTEM_PROMPT", "render_context", "answer"]

ADVISOR_SYSTEM_PROMPT = (
    "You answer questions about airline-network robustness and aviation disruptions using ONLY the "
    "numbered sources provided. Cite sources inline as [1], [2] matching their numbers. If the sources "
    "do not contain the answer, say so plainly. Never invent facts or citations."
)


class RagAnswer(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]


def render_context(hits: List[Hit]) -> str:
    """Number the retrieved chunks so the model can cite them as [1], [2], ..."""
    return "\n\n".join(
        f"[{i}] ({h.source.get('title', '?')})\n{h.text}" for i, h in enumerate(hits, start=1)
    )


def _dedupe_sources(hits: List[Hit]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for h in hits:
        key = (h.source.get("title"), h.source.get("url"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": h.source.get("title"), "url": h.source.get("url")})
    return out


def answer(
    question: str,
    client: LLMClient,
    embedder: Embedder,
    store: VectorStore,
    k: int = 4,
) -> RagAnswer:
    """Retrieve top-k chunks and synthesize a cited answer grounded in them."""
    hits = store.search(embedder.embed([question])[0], k=k)
    if not hits:
        return RagAnswer(question=question, answer="No relevant sources found.", sources=[])
    user_msg = f"Question: {question}\n\nSources:\n{render_context(hits)}"
    text = client.chat(ADVISOR_SYSTEM_PROMPT, user_msg)
    return RagAnswer(question=question, answer=text, sources=_dedupe_sources(hits))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ai/rag/test_advisor.py tests/ai/test_llm_client.py tests/ai/test_openai_client.py -v`
Expected: PASS (advisor 4 + existing llm_client tests incl. new chat + existing openai tests incl. new chat)

- [ ] **Step 5: Commit**

```bash
git add src/ai/llm_client.py src/ai/rag/advisor.py tests/ai/test_llm_client.py tests/ai/test_openai_client.py tests/ai/rag/test_advisor.py
git commit -m "feat(rag): add LLMClient.chat and the citation-grounded advisor"
```

---

### Task R6: Retrieval eval (recall@k)

**Files:**
- Create: `src/ai/rag/eval_rag.py`
- Test: `tests/ai/rag/test_eval_rag.py`

**Interfaces:**
- Consumes: `Embedder`, `VectorStore`.
- Produces: `RetrievalCase(question, expected_title)`; `RetrievalReport(n_cases, recall_at_k, hits)`;
  `evaluate_retrieval(embedder, store, dataset, k=4) -> RetrievalReport`; `format_report(report, k=4) -> str`;
  `GOLDEN_QUESTIONS: list[RetrievalCase]`.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/rag/test_eval_rag.py`:

```python
from src.ai.rag.embedder import FakeEmbedder
from src.ai.rag.store import VectorStore
from src.ai.rag.chunker import Chunk
from src.ai.rag.eval_rag import (
    RetrievalCase, RetrievalReport, evaluate_retrieval, format_report, GOLDEN_QUESTIONS,
)


def _store(emb, items):
    s = VectorStore()
    chunks = [Chunk(text, {"title": title}) for text, title in items]
    s.add(chunks, emb.embed([c.text for c in chunks]))
    return s


def test_recall_all_hit():
    emb = FakeEmbedder(dim=16)
    store = _store(emb, [("topic alpha", "Alpha"), ("topic beta", "Beta")])
    # question text identical to a chunk -> identical FakeEmbedder vector -> guaranteed top hit
    cases = [RetrievalCase("topic alpha", "Alpha"), RetrievalCase("topic beta", "Beta")]
    assert evaluate_retrieval(emb, store, cases, k=1).recall_at_k == 1.0


def test_recall_partial():
    emb = FakeEmbedder(dim=16)
    store = _store(emb, [("topic alpha", "Alpha")])
    cases = [RetrievalCase("topic alpha", "Alpha"), RetrievalCase("topic alpha", "Gamma")]
    assert evaluate_retrieval(emb, store, cases, k=1).recall_at_k == 0.5


def test_format_report_contains_recall():
    rep = RetrievalReport(n_cases=2, recall_at_k=1.0, hits=[True, True])
    assert "recall@4: 100.0%" in format_report(rep, k=4)


def test_golden_questions_nonempty_and_typed():
    assert len(GOLDEN_QUESTIONS) >= 6
    assert all(isinstance(c, RetrievalCase) and c.question and c.expected_title for c in GOLDEN_QUESTIONS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ai/rag/test_eval_rag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ai.rag.eval_rag'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ai/rag/eval_rag.py`:

```python
"""Offline retrieval evaluation: measures recall@k of the vector store on a golden set."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from .embedder import Embedder
from .store import VectorStore

__all__ = [
    "RetrievalCase", "RetrievalReport", "evaluate_retrieval", "format_report", "GOLDEN_QUESTIONS",
]


@dataclass
class RetrievalCase:
    question: str
    expected_title: str


@dataclass
class RetrievalReport:
    n_cases: int
    recall_at_k: float
    hits: List[bool]


def evaluate_retrieval(
    embedder: Embedder,
    store: VectorStore,
    dataset: List[RetrievalCase],
    k: int = 4,
) -> RetrievalReport:
    """For each case, retrieve top-k and check whether the expected source title is present."""
    hits: List[bool] = []
    for case in dataset:
        results = store.search(embedder.embed([case.question])[0], k=k)
        titles = {h.source.get("title") for h in results}
        hits.append(case.expected_title in titles)
    n = len(dataset) or 1
    return RetrievalReport(n_cases=len(dataset), recall_at_k=sum(hits) / n, hits=hits)


def format_report(report: RetrievalReport, k: int = 4) -> str:
    return f"Cases: {report.n_cases}\nrecall@{k}: {report.recall_at_k:.1%}"


# Golden questions paired with the Wikipedia article title expected to answer them.
GOLDEN_QUESTIONS: List[RetrievalCase] = [
    RetrievalCase("What does betweenness centrality measure?", "Betweenness centrality"),
    RetrievalCase("What is a scale-free network?", "Scale-free network"),
    RetrievalCase("Why are some networks robust to random failures but fragile to targeted attack?",
                  "Robustness of complex networks"),
    RetrievalCase("How does a hub-and-spoke route structure work?", "Spoke–hub distribution paradigm"),
    RetrievalCase("What disrupted European air travel in 2010?",
                  "Air travel disruption after the 2010 Eyjafjallajökull eruption"),
    RetrievalCase("How did COVID-19 affect aviation?", "Impact of the COVID-19 pandemic on aviation"),
    RetrievalCase("What is network science?", "Network science"),
]


def main() -> None:  # pragma: no cover - needs OPENAI_API_KEY and a built index
    import os
    from .embedder import OpenAIEmbedder
    from .index import INDEX_PATH

    embedder = OpenAIEmbedder(api_key=os.environ.get("OPENAI_API_KEY"))
    store = VectorStore.load(INDEX_PATH)
    print(format_report(evaluate_retrieval(embedder, store, GOLDEN_QUESTIONS, k=4), k=4))


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ai/rag/test_eval_rag.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ai/rag/eval_rag.py tests/ai/rag/test_eval_rag.py
git commit -m "feat(rag): add retrieval recall@k eval with a golden question set"
```

---

### Task R7: Streamlit Resilience Advisor panel

**Files:**
- Modify: `src/app/streamlit_app.py` (additive imports + a new panel block)

**Interfaces:**
- Consumes: `answer` (as `rag_answer`), `OpenAIEmbedder`, `VectorStore`, `INDEX_PATH`, and the existing
  `make_client`, plus the `api_key` text input already defined in the "ASK AI" panel.

- [ ] **Step 1: Add imports**

In `src/app/streamlit_app.py`, find:

```python
from src.ai.factory import make_client
```

Immediately after it, add:

```python
from src.ai.rag.advisor import answer as rag_answer
from src.ai.rag.embedder import OpenAIEmbedder
from src.ai.rag.store import VectorStore
from src.ai.rag.index import INDEX_PATH
```

- [ ] **Step 2: Add the panel block**

Find the end of the "ASK AI" result-rendering block:

```python
    ai_result = st.session_state.get("ai_result")
    if ai_result:
        st.markdown(f"**Tool:** `{ai_result['tool_name']}`  ")
        st.caption(f"args: {ai_result['arguments']}")
        st.write(ai_result["explanation"])
```

Immediately AFTER that block (same 4-space indentation, still inside `with left:`), add:

```python
    st.caption("RESILIENCE ADVISOR (RAG)")
    st.caption("Uses OpenAI for retrieval + answer. Enter an OpenAI key in the Ask AI panel.")
    rag_q = st.text_input(
        "Ask about resilience / disruptions",
        key="rag_q",
        placeholder="What disrupted European air travel in 2010?",
    )
    if st.button("Ask Advisor", use_container_width=True):
        if not api_key:
            st.warning("Enter an OpenAI API key in the Ask AI panel above.")
        elif not rag_q:
            st.warning("Type a question first.")
        elif not INDEX_PATH.exists():
            st.error("Knowledge index not built. Run: python -m src.ai.rag.index")
        else:
            with st.spinner("Retrieving..."):
                try:
                    store = VectorStore.load(INDEX_PATH)
                    embedder = OpenAIEmbedder(api_key=api_key)
                    res = rag_answer(rag_q, make_client("openai", api_key=api_key), embedder, store)
                    st.session_state["rag_result"] = res.model_dump()
                except Exception as e:
                    st.error(f"Advisor error: {e}")

    rag_result = st.session_state.get("rag_result")
    if rag_result:
        st.write(rag_result["answer"])
        if rag_result["sources"]:
            st.caption("Sources")
            for i, s in enumerate(rag_result["sources"], start=1):
                st.markdown(f"{i}. [{s['title']}]({s['url']})")
```

- [ ] **Step 3: Verify (no new automated test — UI glue)**

Run: `python -c "import ast; ast.parse(open('src/app/streamlit_app.py', encoding='utf-8').read())"` (exit 0)
Run: `python -m pytest tests/ -q` (full suite still passes)
Do NOT launch Streamlit (needs a key and a built index — that is the manual R8 step).

- [ ] **Step 4: Commit**

```bash
git add src/app/streamlit_app.py
git commit -m "feat(rag): add Resilience Advisor panel to the Streamlit app"
```

---

### Task R8: Build the real index + README (MANUAL — needs OPENAI_API_KEY)

This task needs a real OpenAI key and (for the live demo) a browser, so an automated worker cannot
complete it. Document it and stop; the human runs the build and fills the number.

**Files:**
- Generate + commit: `data/kb/*.md`, `data/kb/manifest.json`, `data/kb/index.npz`, `data/kb/index.meta.json`
- Modify: `README.md` (add a RAG section)

- [ ] **Step 1: Fetch the corpus and build the index** (human, with key)

```bash
export OPENAI_API_KEY=sk-...        # PowerShell: $env:OPENAI_API_KEY="sk-..."
python -m src.ai.rag.corpus         # caches data/kb/*.md + manifest.json
python -m src.ai.rag.index          # builds data/kb/index.npz (+ .meta.json)
python -m src.ai.rag.eval_rag       # prints recall@4 — record this number
```

- [ ] **Step 2: Commit the prebuilt corpus + index** (so the deployed app needs no rebuild)

```bash
git add data/kb
git commit -m "data(rag): add cached Wikipedia corpus and prebuilt vector index"
```

- [ ] **Step 3: Add a README RAG section**

Add under the existing "AI What-If Assistant" section a short "Resilience Advisor (RAG)" subsection
describing: the Wikipedia corpus (revision-pinned, cached), the embedder/store interfaces, the
`recall@k` number from Step 1, the offline test path, and how to rebuild
(`python -m src.ai.rag.corpus && python -m src.ai.rag.index`). Commit it.

```bash
git add README.md
git commit -m "docs(rag): document the Resilience Advisor and record retrieval recall@k"
```

---

## Self-Review

**Spec coverage:**
- §2 standalone advisor with citations → R5 (advisor) + R7 (panel). ✓
- §3 architecture (corpus/embedder/chunker/store/index/advisor/eval_rag) → R1–R6. ✓
- §3 `LLMClient.chat` additive → R5. ✓
- §3 vector store = numpy cosine, no FAISS → R2. ✓
- §4 Wikipedia corpus, oldid-recorded, cached → R3 + R8. ✓
- §5 offline testing via FakeEmbedder/FakeLLMClient → R1–R6 tests. ✓
- §6 deployment: prebuilt index committed, query-only embedding → R7 (load index, embed query) + R8 (commit index). ✓
- §7 error handling: missing index, empty retrieval, missing key → R7 (index/key guards) + R5 (empty retrieval). ✓
- §8 phasing R1–R8 → all tasks present. ✓
- §9 recall@k recorded → R6 + R8. ✓

**Placeholder scan:** No "TBD"/"implement later"; every code step contains complete code. R8 is explicitly a manual/human task (needs a key), with exact commands — not a placeholder.

**Type consistency:** `Embedder.embed(list[str]) -> list[list[float]]`, `VectorStore.add/search/save/load`,
`Hit(score,text,source)`, `Chunk(text,source)`, `RagAnswer(question,answer,sources)`,
`answer(question,client,embedder,store,k)`, `evaluate_retrieval(embedder,store,dataset,k)`,
`LLMClient.chat(system,user)` — names/signatures are consistent across R1–R7.

**Note for executor:** R8 cannot be completed without the user's OpenAI key; the Streamlit panel (R7)
already degrades gracefully when `data/kb/index.npz` is absent, so R1–R7 are fully shippable and tested
before R8.
