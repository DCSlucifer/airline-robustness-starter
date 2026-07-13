import pytest

from src.ai.llm_client import FakeLLMClient
from src.ai.rag.advisor import CitationValidationError, RagAnswer, answer, render_context
from src.ai.rag.chunker import Chunk
from src.ai.rag.embedder import FakeEmbedder
from src.ai.rag.store import Hit, VectorStore
from src.ai.schemas import ToolSelection


def _client(text):
    return FakeLLMClient(
        selection=ToolSelection(name="x", arguments={}),
        explanation="n/a",
        chat_response=text,
    )


def _store(embedder):
    s = VectorStore()
    chunks = [
        Chunk(
            "Betweenness measures bridge importance.",
            {"title": "Betweenness centrality", "url": "u1"},
        ),
        Chunk(
            "The 2010 ash cloud closed European airspace.",
            {"title": "Eyjafjallajokull", "url": "u2"},
        ),
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
    s.add(
        [Chunk("a", {"title": "T", "url": "u"}), Chunk("b", {"title": "T", "url": "u"})],
        emb.embed(["a", "b"]),
    )
    res = answer("q", _client("ans [1]"), emb, s, k=2)
    assert len(res.sources) == 1


def test_answer_empty_store_no_fabrication():
    res = answer("q", _client("unused"), FakeEmbedder(dim=16), VectorStore(), k=3)
    assert res.sources == []
    assert "No relevant sources" in res.answer


def test_render_context_numbers_sources():
    ctx = render_context([Hit(0.9, "txt", {"title": "T"})], [1])
    assert "[1]" in ctx and "T" in ctx


class _RecordingClient:
    """Captures the user message passed to chat so we can inspect the numbered context."""

    def __init__(self, text="ans [1]"):
        self.text = text
        self.last_user = None

    def select_tool(self, query, tools):  # pragma: no cover - unused by the advisor
        raise NotImplementedError

    def explain(self, query, tool_name, result):  # pragma: no cover - unused
        return ""

    def chat(self, system, user):
        self.last_user = user
        return self.text


def test_citation_numbers_align_with_deduped_sources():
    # Two chunks from article A and one from B -> deduped to 2 sources. The context the model sees
    # must number A's chunks as [1] (shared) and B as [2] -- never [3], which would point past the
    # 2-item source list the UI renders.
    emb = FakeEmbedder(dim=16)
    s = VectorStore()
    s.add(
        [
            Chunk("a1", {"title": "A", "url": "ua"}),
            Chunk("a2", {"title": "A", "url": "ua"}),
            Chunk("b1", {"title": "B", "url": "ub"}),
        ],
        emb.embed(["a1", "a2", "b1"]),
    )
    client = _RecordingClient()
    res = answer("q", client, emb, s, k=3)
    assert len(res.sources) == 2
    assert {src["title"] for src in res.sources} == {"A", "B"}
    # 3 chunks but only 2 deduped sources -> context numbers are exactly {1, 2}, never 3.
    assert "[1]" in client.last_user
    assert "[2]" in client.last_user
    assert "[3]" not in client.last_user


@pytest.mark.parametrize("text", ["answer without citation", "invalid [99]"])
def test_answer_rejects_ungrounded_citations(text):
    embedder = FakeEmbedder(dim=16)

    with pytest.raises(CitationValidationError):
        answer("question", _client(text), embedder, _store(embedder), k=2)


@pytest.mark.parametrize("question", ["", "   "])
def test_answer_rejects_empty_question(question):
    with pytest.raises(ValueError, match="must not be empty"):
        answer(question, _client("unused [1]"), FakeEmbedder(), VectorStore())


def test_render_context_requires_matching_number_count():
    with pytest.raises(ValueError):
        render_context([Hit(0.9, "a", {"title": "A"})], [])
