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
