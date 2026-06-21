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
