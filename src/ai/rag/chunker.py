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
