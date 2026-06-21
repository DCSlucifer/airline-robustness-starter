"""Factory that selects an LLM provider from an explicit name or the LLM_PROVIDER env var."""
from __future__ import annotations
import os
from typing import Any, Optional

from .llm_client import LLMClient, ClaudeClient, OpenAIClient

__all__ = ["make_client", "DEFAULT_PROVIDER"]

DEFAULT_PROVIDER = "openai"


def make_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> LLMClient:
    """Return an LLMClient for the chosen provider.

    Resolution order: explicit ``provider`` arg, then ``LLM_PROVIDER`` env var, then DEFAULT_PROVIDER.
    Extra kwargs (e.g. router_model, explainer_model, _client) pass through to the client.
    """
    name = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if name == "openai":
        return OpenAIClient(api_key=api_key, **kwargs)
    if name in ("anthropic", "claude"):
        return ClaudeClient(api_key=api_key, **kwargs)
    raise ValueError(f"Unknown LLM provider: {name!r} (expected 'openai' or 'anthropic')")
