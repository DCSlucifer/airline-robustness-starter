"""Offline retrieval evaluation: measures recall@k of the vector store on a golden set."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .embedder import Embedder
from .store import VectorStore

__all__ = [
    "RetrievalCase",
    "RetrievalReport",
    "evaluate_retrieval",
    "format_report",
    "GOLDEN_QUESTIONS",
]


@dataclass
class RetrievalCase:
    question: str
    expected_title: str


@dataclass
class RetrievalReport:
    n_cases: int
    recall_at_k: float
    hits: list[bool]


def evaluate_retrieval(
    embedder: Embedder,
    store: VectorStore,
    dataset: list[RetrievalCase],
    k: int = 4,
) -> RetrievalReport:
    """For each case, retrieve top-k and check whether the expected source title is present."""
    hits: list[bool] = []
    for case in dataset:
        results = store.search(embedder.embed([case.question])[0], k=k)
        titles = {h.source.get("title") for h in results}
        hits.append(case.expected_title in titles)
    n = len(dataset) or 1
    return RetrievalReport(n_cases=len(dataset), recall_at_k=sum(hits) / n, hits=hits)


def format_report(report: RetrievalReport, k: int = 4) -> str:
    return f"Cases: {report.n_cases}\nrecall@{k}: {report.recall_at_k:.1%}"


# Golden questions paired with the Wikipedia article title expected to answer them.
GOLDEN_QUESTIONS: list[RetrievalCase] = [
    RetrievalCase("What does betweenness centrality measure?", "Betweenness centrality"),
    RetrievalCase("What is a scale-free network?", "Scale-free network"),
    RetrievalCase(
        "Why are some networks robust to random failures but fragile to targeted attack?",
        "Robustness of complex networks",
    ),
    RetrievalCase(
        "How does a hub-and-spoke route structure work?", "Spoke–hub distribution paradigm"
    ),
    RetrievalCase(
        "What disrupted European air travel in 2010?",
        "Air travel disruption after the 2010 Eyjafjallajökull eruption",
    ),
    RetrievalCase(
        "How did COVID-19 affect aviation?", "Impact of the COVID-19 pandemic on aviation"
    ),
    RetrievalCase("What is network science?", "Network science"),
]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live opt-in
    from .embedder import OpenAIEmbedder
    from .index import INDEX_PATH
    from .store import IndexFormatError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument("--model", default="text-embedding-3-small")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("--k must be positive")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("error: OPENAI_API_KEY is required for retrieval evaluation", file=sys.stderr)
        return 2

    try:
        store = VectorStore.load(args.index, expected_model=args.model, require_nonempty=True)
        embedder = OpenAIEmbedder(api_key=api_key, model=args.model)
        report = evaluate_retrieval(embedder, store, GOLDEN_QUESTIONS, k=args.k)
    except (IndexFormatError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(report, k=args.k))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
