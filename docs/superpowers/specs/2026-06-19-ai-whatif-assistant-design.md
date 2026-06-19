# Design Spec — AI What-If Assistant (Natural-Language Robustness Analysis)

**Date:** 2026-06-19
**Status:** Approved (brainstorming) — ready for implementation planning
**Goal:** Add a GenAI/LLM layer to the existing Airline Network Robustness project to make it a strong **AI Engineer** portfolio piece.

---

## 1. Context & Motivation

The existing project is a solid **network-science** tool: it models the global airline network (~7,700 airports, ~67,600 routes from OpenFlights) as a directed graph and simulates attacks/defenses with topological metrics (GWCC, GSCC, ASPL, diameter, OD-within-H-hops). It ships a deployed Streamlit + PyDeck dashboard, a CLI, and 69 unit tests.

**Gap:** The repo contains no AI/ML. The owner is applying for **GenAI / LLM Engineer** roles, so the portfolio needs a genuine, well-engineered LLM layer.

**Key enabler:** The simulation functions (`targeted_node_removal`, `random_node_failures`, `edge_betweenness_attack`, `geographic_attack_radius`, `community_bridge_attack`, `greedy_edge_addition`, `node_hardening_list`) already have clean, well-typed signatures — they are ready-made **tools** for LLM function calling. The AI layer sits on top of the engine; existing simulation logic is not modified.

---

## 2. Scope

### In scope
A **"Ask AI"** panel inside the existing Streamlit app. The user asks a question in natural language (e.g. *"What happens if a major hurricane hits the US East Coast?"*). The LLM:
1. Interprets intent and selects the correct simulation tool + parameters (function calling).
2. The real simulation engine executes the tool (numbers are never invented by the LLM).
3. The LLM explains the results in plain language (which metrics dropped, why, defense suggestions).

### Out of scope (explicitly deferred — "future work")
- Autonomous agent that loops over many scenarios on its own.
- RAG over aviation literature / incident reports.
- Multi-turn conversational memory beyond the current scenario.

These are documented as future work so they can be discussed with a recruiter without being built now (YAGNI).

### Core principle
The LLM **orchestrates**; it does not compute. All metrics come from the existing simulation engine. This grounding is the central anti-hallucination guarantee and the main engineering talking point.

---

## 3. Architecture

A new package `src/ai/`, fully decoupled from the simulation code:

```
src/ai/
├── llm_client.py    # Thin provider interface: .complete(messages, tools) -> response
│                    # Default: ClaudeClient. Adding a provider = adding one class.
├── tools.py         # Tool registry: JSON tool schemas <-> real attack/defense functions
├── orchestrator.py  # Loop: parse intent -> tool_call -> execute -> observe -> explain
├── schemas.py       # Pydantic models: ParsedIntent, ToolCall, AssistantAnswer
├── guardrails.py    # Validate & cap parameters (k, budget, radius); enforce tool whitelist
└── prompts.py       # Versioned system prompt + tool descriptions (no scattered hardcoding)
```

The Streamlit integration adds an **"Ask AI"** panel; layer/state-building helpers may be extracted out of `streamlit_app.py` as part of the hardening track (Section 5).

### Data flow

```
User NL query
  -> llm_client (routing model): select tool + parameters  --(structured JSON)
  -> guardrails: validate & clamp parameters; reject non-whitelisted tools
  -> tools: call the REAL simulation function on the current graph
  -> metric result (dict)
  -> llm_client (explanation model): natural-language explanation grounded in the dict
  -> Streamlit renders text + reuses existing map/metric panels
```

The tool result reuses the existing replay/metrics machinery — the AI is "a new way to drive" the engine, not a parallel implementation.

### Provider decision (deferred)
The provider is **not finalized**. The architecture hides the LLM behind `llm_client.py` so the choice is cheap to change at implementation time.

- **Recommended default:** Anthropic Claude — strongest, most stable tool-use.
- **Cost routing:** a cheap routing model (e.g. Claude Haiku 4.5, `claude-haiku-4-5`) for intent parsing + tool selection; a stronger model (e.g. Claude Sonnet 4.6, `claude-sonnet-4-6`) for the explanation step.
- **Public demo:** Bring-Your-Own-Key (user pastes their own API key in the UI) so the project owner does not absorb spam costs. Exact API details (request shape, pricing, token caps) are resolved during implementation planning.

---

## 4. AI-Engineering rigor features

These are what distinguish "calls an API" from "engineered an LLM system". Each maps to a JD skill:

1. **Structured output + validation (Pydantic)** — the LLM returns JSON conforming to `ParsedIntent`; on parse/validation failure, a controlled retry. → *reliable LLM outputs*.
2. **Guardrails / input safety** — clamp `k <= node count`, `budget <= 10`, valid `radius`; enforce a tool whitelist. The LLM can never call an unlisted function or pass app-breaking parameters. → *production safety*.
3. **Eval harness (highest-value, rare in portfolios)** — a golden set of ~20-30 NL queries, each labeled with the expected tool + parameters. A script measures **tool-selection accuracy** and **parameter-extraction accuracy**, producing a headline number (e.g. "tool-selection accuracy 95%"). → *LLM evaluation*.
4. **Observability / tracing** — per turn, log the prompt, selected tool, parameters, latency, token usage, and estimated cost. → *LLMOps*.
5. **Disciplined prompt engineering** — system prompt + tool descriptions versioned in `prompts.py`, not scattered. → easy A/B and review.
6. **Cost routing** — cheap model for routing, stronger model for explanation. → *cost optimization*.

---

## 5. Repo-hardening companion track

Runs in parallel; makes the whole repo look professional. Addresses weaknesses found during review:

1. **CI/CD** — GitHub Actions runs the 69 tests + lint on every push; green badge on README. (Currently the biggest visible gap.)
2. **Lint / format / type-check** — add `ruff` + `mypy`, configured in `pyproject.toml`.
3. **Bug fixes** — fix the indentation defect in `streamlit_app.py` (~lines 239-249), unify Vietnamese/English comments, narrow broad `except Exception` handlers.
4. **Refactor `streamlit_app.py`** (521 lines) — extract layer/state-building helpers into a focused module.
5. **"Findings" in README** — run several scenarios and publish a **robustness curve** plus quantitative conclusions (e.g. "removing 1% of nodes by betweenness collapses GWCC by ~40%"). Turns the repo from "a tool" into "research with conclusions".
6. **Dockerfile + reproducible environment** — runs anywhere.

---

## 6. Phasing

Each phase is an independently demoable commit/PR. If time runs out, stopping at Phase 2 still yields a working demo.

- **Phase 0 — Quick hardening** (~half day): CI + lint + bug fixes + Dockerfile. Clean foundation and a fast win.
- **Phase 1 — Tool layer + structured output** (core): `tools.py`, `schemas.py`, `guardrails.py`, `llm_client.py`. Unit-testable without UI.
- **Phase 2 — Orchestrator + Streamlit panel**: assemble parse->execute->explain into the "Ask AI" panel. First demoable point.
- **Phase 3 — Eval harness + observability**: golden set + accuracy script + tracing/logging. The recruiter-impressing part.
- **Phase 4 — Findings writeup + README/architecture diagram**: narrative, robustness curves, architecture diagram.

---

## 7. Portfolio narrative

> "I took a real-scale network simulation engine (7,700 airports) and built an AI agent layer using function calling, with structured output, guardrails, and an eval harness, to turn complex robustness analysis into natural-language conversation — the AI orchestrates, the real engine computes, no hallucinated numbers."

This hits four AI-Engineer JD keywords directly: **function calling, structured output, evaluation, grounding**.

---

## 8. Success criteria

- A user can ask a natural-language scenario in the Streamlit app and receive a correct, grounded answer backed by a real simulation run.
- The LLM never returns a numeric metric it did not obtain from the engine.
- Tool-selection accuracy on the golden set is measured and reported (target: high, e.g. >= 90%).
- Parameters are always validated/clamped before any simulation runs; no LLM-supplied input can break the app.
- CI is green (tests + lint) on every push.
- README presents at least one quantitative robustness finding with a chart.

---

## 9. Future work (deferred, not built)

- RAG over aviation resilience literature and incident reports (citations / grounding depth).
- Autonomous vulnerability-analyst agent that probes the network across many scenarios.
- Provider-agnostic abstraction across Claude / OpenAI / Gemini (only if a concrete need arises).
