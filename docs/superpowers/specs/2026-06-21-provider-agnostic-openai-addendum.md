# Addendum: Provider-agnostic LLM layer + OpenAI (GPT-4o mini)

**Date:** 2026-06-21
**Extends:** docs/superpowers/specs/2026-06-19-ai-whatif-assistant-design.md
**Status:** Approved — implementing on branch `feat/ai-whatif-assistant`.

## Motivation
The owner has a tight budget (personal CV project, no funding) and wants the LLM layer to be
cheap, not vendor-locked, and developable toward production. Provider choice is finalized as
**provider-agnostic with OpenAI GPT-4o mini as the default**, Claude remaining available. The
existing `LLMClient` Protocol already enables this — only additive work is needed.

## Decisions
- **No changes to existing simulation modules or to `tools.py`.** `TOOL_SPECS` stays the single
  canonical (Anthropic-shaped) tool definition; the OpenAI client translates it internally.
- **Default provider:** `openai`, model `gpt-4o-mini` for both router and explainer (configurable).
- **Cost posture:** public demo uses BYOK (viewer's key); dev can use a free tier; future RAG will
  use local embeddings (sentence-transformers) + FAISS to stay near $0.

## Three technical differences handled in `OpenAIClient`
1. Tool format: translate `TOOL_SPECS` (`name`/`description`/`input_schema`/`strict`) into OpenAI's
   `{"type":"function","function":{name,description,parameters,strict}}` via an internal adapter.
2. Response parsing: OpenAI returns the tool call at
   `choices[0].message.tool_calls[0].function.arguments` (a JSON **string** → `json.loads`); text at
   `choices[0].message.content`. Force a tool call with `tool_choice="required"`.
3. Lazy import + injectable `_client` (same pattern as `ClaudeClient`) so offline stub tests need no
   SDK, key, or network.

## Work (subagent-driven, on current branch)
- **Task A:** `OpenAIClient` + `_to_openai_tools` adapter in `llm_client.py` + stub-based tests.
- **Task B:** `factory.make_client(provider, api_key, ...)` selecting Claude/OpenAI, default from env
  `LLM_PROVIDER` (fallback "openai") + tests.
- **Task C:** provider selector in the Streamlit "Ask AI" panel; add `openai>=1.0` to requirements.

## CV narrative
"Provider-agnostic LLM layer — swap Claude <-> OpenAI by one config value — with an eval harness that
runs on any provider to compare tool-selection accuracy and cost."

## Out of scope here (next phase, separately brainstormed)
RAG (local embeddings + FAISS) and the autonomous agent loop.
