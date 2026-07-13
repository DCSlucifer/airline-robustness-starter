# Design Spec — RAG Resilience Advisor

**Date:** 2026-06-21
**Status:** Approved (brainstorming) — ready for implementation planning
**Branch:** `feat/ai-whatif-assistant` (continues)
**Extends:** the AI What-If Assistant. Adds a standalone Retrieval-Augmented Generation (RAG) Q&A
feature ("Resilience Advisor") that answers aviation-resilience questions with cited sources.

## 1. Motivation

The owner is building this as a CV project for a GenAI/LLM Engineer role on a tight budget. RAG is
on nearly every AI-Engineer JD. This adds a clean, demonstrable RAG pipeline that also showcases
production thinking: reproducible ingestion, swappable embedder/store behind interfaces, an offline
test path, a retrieval-quality metric, and resource-light deployment.

## 2. Scope

### In scope
A standalone **"Resilience Advisor"** panel in the Streamlit app. The user asks a natural-language
question about network robustness or aviation disruptions. The system retrieves relevant chunks from
a cached Wikipedia corpus and an LLM answers **with numbered citations** (each linking to the source
article revision — this doubles as CC BY-SA attribution).

### Out of scope (future work)
- Wiring RAG into the What-If explainer step.
- Re-ranking / hybrid (keyword + vector) search.
- Multi-turn conversational memory.
- A managed/online vector database.

### Core principles
- RAG lives entirely in `src/ai/rag/`, isolated from the existing What-If code.
- The embedder and the vector store are each hidden behind an interface (like `LLMClient`), so the
  provider/backend is swappable ("not vendor-locked").
- Everything except the real OpenAI embedding call is testable offline (no key, no network).

## 3. Architecture

```
src/ai/rag/
├── __init__.py
├── corpus.py     # WIKI_ARTICLES (title + pinned oldid); fetch_corpus() -> cache data/kb/*.md + manifest.json
├── embedder.py   # Embedder (Protocol) + FakeEmbedder (tests) + OpenAIEmbedder (lazy, text-embedding-3-small)
├── chunker.py    # split a document into chunks (by paragraph, with a max-char cap)
├── store.py      # VectorStore: numpy cosine search + persist/load (data/kb/index.npz + meta.json)
├── index.py      # build_index(): load kb -> chunk -> embed -> persist VectorStore
├── advisor.py    # answer(question, client, embedder, store, k) -> RagAnswer{answer, sources[]}
└── eval_rag.py   # retrieval eval: golden question -> expected source; measures recall@k
```

Additive extension to the existing LLM layer:
- Add `chat(system: str, user: str) -> str` to the `LLMClient` Protocol and implement it on
  `ClaudeClient`, `OpenAIClient`, and `FakeLLMClient`. The advisor uses `client.chat(...)` to
  synthesize the final answer. This is purely additive; existing methods are unchanged.

**Vector store decision:** numpy brute-force cosine similarity. The corpus is small (a few hundred
chunks), so brute force is sufficient, zero-dependency, and trivially testable. FAISS is intentionally
NOT used (overkill at this scale); the `VectorStore` interface leaves it as an easy future swap.

### Data flow

```
[build, offline, run once]
  WIKI_ARTICLES (title + oldid)
    -> corpus.fetch_corpus()  -> data/kb/<slug>.md (+ manifest.json)   [committed to repo]
    -> index.build_index()    -> chunk -> OpenAIEmbedder.embed -> data/kb/index.npz + meta.json [committed]

[query, runtime]
  question
    -> embedder.embed([question]) -> query vector
    -> store.search(query_vector, k) -> top-k chunks (each with source title + url + oldid)
    -> advisor.answer(): client.chat(RAG_SYSTEM_PROMPT, context+question) -> answer text
    -> RagAnswer{answer, sources=[deduped {title, url}]}   rendered in Streamlit
```

### Interfaces (produced)
- `Embedder` Protocol: `embed(texts: list[str]) -> list[list[float]]`.
- `FakeEmbedder(dim=...)`: deterministic vectors derived from hashing each text (no model/network).
- `OpenAIEmbedder(api_key=None, model="text-embedding-3-small", _client=None)`: lazy `openai` import,
  injectable `_client` for tests.
- `chunk_document(text, source) -> list[Chunk]` where `Chunk` carries `text` + source metadata.
- `VectorStore`: `.add(chunks, vectors)`, `.search(query_vector, k) -> list[Hit]`, `.save(path)`,
  `VectorStore.load(path)`.
- `RagAnswer` (Pydantic): `question, answer, sources: list[{title, url}]`.
- `answer(question, client, embedder, store, k=4) -> RagAnswer`.
- `evaluate_retrieval(embedder, store, dataset) -> RetrievalReport` with `recall_at_k`.

## 4. Corpus

A curated list of ~10-15 Wikipedia articles, each pinned to a specific revision id (`oldid`) for
reproducibility. Topics span network-science methodology and real aviation disruptions, e.g.:
network science, centrality, betweenness centrality, scale-free networks, robustness of complex
networks, hub-and-spoke distribution, the 2010 Eyjafjallajökull air-travel disruption, the airspace
shutdown after the September 11 attacks, and the impact of COVID-19 on aviation.

`fetch_corpus()` uses the Wikipedia Action API with `prop=extracts&explaintext` (returns clean plain
text — no wikitext parsing) pinned by `oldid`, and writes each article to `data/kb/<slug>.md` with a
small header (title, url, oldid, fetched_at) plus a `manifest.json`. These files are committed so the
index can be rebuilt offline and the demo is deterministic.

## 5. Testing

- `FakeEmbedder` + `FakeLLMClient.chat` make the chunker, store, index builder, advisor, and the
  retrieval eval fully testable offline — no API key, no network — so CI stays green and fast.
- `chunker`: asserts chunk boundaries and that source metadata is preserved.
- `store`: cosine search returns the nearest known vectors in order; save/load round-trips.
- `index`: builds a store from a tiny in-test kb using `FakeEmbedder`.
- `advisor`: with `FakeEmbedder` + a `FakeLLMClient`, returns a `RagAnswer` whose `sources` come from
  the retrieved chunks (citations are grounded in retrieval, not invented).
- `eval_rag`: golden questions over a small fixed corpus; asserts `recall_at_k` math.
- `corpus`: the API-response parsing/caching is tested with a stubbed HTTP response (no live network).

## 6. Deployment & reproducibility

- The prebuilt `index.npz` + `data/kb/*.md` are committed, so the deployed app does NOT re-embed the
  corpus. At query time it makes ONE small embedding call for the question (OpenAI, BYOK) and runs a
  numpy search — light enough for Streamlit Community Cloud's memory limits.
- `oldid` pinning + committed cache + committed index = deterministic, offline-rebuildable.

## 7. Error handling

- Missing `index.npz` → `advisor`/UI raises a clear "run `python -m src.ai.rag.index` first" error.
- Empty retrieval (no chunk above a similarity floor) → answer "No relevant sources found." with no
  fabricated citation.
- Missing embedding key → the UI prompts for a BYOK key (same pattern as the What-If panel).

## 8. Phasing (subagent-driven)

| Task | Content | Tests |
|------|---------|-------|
| R1 | `Embedder` interface + `FakeEmbedder` + `OpenAIEmbedder` (lazy) | stub, offline |
| R2 | `chunker` + numpy `VectorStore` (search/persist/load) | known vectors |
| R3 | `corpus.py`: article manifest + fetch/cache | stubbed HTTP |
| R4 | `index.py` builder: kb -> chunk -> embed -> persist | FakeEmbedder |
| R5 | `LLMClient.chat` (additive) + `advisor.answer` (RagAnswer + citations) | Fake + Fake |
| R6 | `eval_rag.py` retrieval recall@k + golden questions | fixed corpus |
| R7 | Streamlit "Resilience Advisor" panel | syntax + full suite |
| R8 (manual) | Build real index with a key; README RAG section; commit index | manual |

Stop points: after R5 the advisor works from the CLI; R7 gives the UI demo; R6 yields the
recall@k number for the CV.

## 9. Success criteria

- A user can ask a resilience question in the app and get an answer whose citations point to the
  actual retrieved Wikipedia sources (no invented citations).
- The entire pipeline (minus the real OpenAI embed call) is covered by offline tests; CI stays green.
- A `recall@k` number is produced by the retrieval eval and recorded in the README.
- The deployed app embeds only the query at runtime (no corpus re-embedding) and runs within
  Streamlit Cloud resource limits.

## 10. Dependencies

**No new dependency.** `openai` (already present) provides embeddings; `numpy` (already present)
provides the vector store. The corpus fetch script uses stdlib `urllib.request` (no `requests`
dependency) — the fetcher is offline/one-shot and not on the runtime path. FAISS is intentionally
omitted (see §3).

## 11. CV narrative

"A retrieval-augmented Resilience Advisor with a reproducible Wikipedia ingestion pipeline (revision-
pinned), an embedder and vector store swappable behind interfaces, a retrieval-recall@k evaluation,
fully offline-tested, and a resource-light BYOK deployment."
