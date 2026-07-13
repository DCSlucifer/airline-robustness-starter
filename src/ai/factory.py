"""Factory that selects an LLM provider from an explicit name or the LLM_PROVIDER env var."""

from __future__ import annotations

import os
from typing import Any

from .llm_client import ClaudeClient, LLMClient, LLMConfigurationError, OpenAIClient

__all__ = ["make_client", "resolve_provider", "DEFAULT_PROVIDER"]

DEFAULT_PROVIDER = "openai"


def resolve_provider(provider: str | None = None) -> str:
    """Resolve an explicit provider or ``LLM_PROVIDER`` to its canonical name."""
    name = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name == "openai":
        return "openai"
    if name in ("anthropic", "claude"):
        return "anthropic"
    raise LLMConfigurationError(
        f"Unknown LLM provider: {name!r} (expected 'openai' or 'anthropic')"
    )


def make_client(
    provider: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> LLMClient:
    """Return an LLMClient for the chosen provider.

    Resolution order: explicit ``provider`` arg, then ``LLM_PROVIDER`` env var, then DEFAULT_PROVIDER.
    Extra kwargs (e.g. router_model, explainer_model, _client) pass through to the client.
    """
    name = resolve_provider(provider)
    if name == "openai":
        return OpenAIClient(api_key=api_key, **kwargs)
    if name == "anthropic":
        return ClaudeClient(api_key=api_key, **kwargs)
    raise AssertionError(f"Unhandled canonical LLM provider: {name}")
