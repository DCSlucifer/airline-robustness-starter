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
