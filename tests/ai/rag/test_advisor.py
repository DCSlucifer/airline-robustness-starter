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
