"""Resilience Advisor: retrieve relevant chunks and answer with grounded citations."""

from __future__ import annotations

from typing import Any

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
    sources: list[dict[str, Any]]


def render_context(hits: list[Hit], numbers: list[int]) -> str:
    """Render each chunk prefixed by its SOURCE number so [n] aligns with the source list.

    `numbers[i]` is the 1-based index (within the deduped source list) of `hits[i]`. Chunks from
    the same article therefore share a number, keeping inline [n] markers in one-to-one
    correspondence with the rendered sources.
    """
    return "\n\n".join(
        f"[{num}] ({h.source.get('title', '?')})\n{h.text}"
        for num, h in zip(numbers, hits, strict=True)
    )


def _dedupe_sources(hits: list[Hit]) -> list[dict[str, Any]]:
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
    # Dedupe first, then number each hit by its source's index so the inline [n] markers the
    # model emits line up exactly with the deduped source list the UI renders.
    sources = _dedupe_sources(hits)
    index_of = {(s["title"], s["url"]): i for i, s in enumerate(sources, start=1)}
    numbers = [index_of[(h.source.get("title"), h.source.get("url"))] for h in hits]
    user_msg = f"Question: {question}\n\nSources:\n{render_context(hits, numbers)}"
    text = client.chat(ADVISOR_SYSTEM_PROMPT, user_msg)
    return RagAnswer(question=question, answer=text, sources=sources)
