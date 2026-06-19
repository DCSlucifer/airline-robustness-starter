"""Provider-swappable LLM client. Only ClaudeClient touches the network/SDK."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol

from .schemas import ToolSelection
from .prompts import (
    ROUTER_SYSTEM_PROMPT,
    EXPLAINER_SYSTEM_PROMPT,
    render_explain_prompt,
)

__all__ = ["LLMClient", "FakeLLMClient", "ClaudeClient"]


class LLMClient(Protocol):
    """The orchestrator depends only on this interface, never on a provider SDK."""

    def select_tool(self, query: str, tools: List[Dict[str, Any]]) -> ToolSelection: ...

    def explain(self, query: str, tool_name: str, result: Dict[str, Any]) -> str: ...


class FakeLLMClient:
    """Deterministic client for offline tests; returns preset values."""

    def __init__(self, selection: ToolSelection, explanation: str = "(no explanation)"):
        self._selection = selection
        self._explanation = explanation

    def select_tool(self, query: str, tools: List[Dict[str, Any]]) -> ToolSelection:
        return self._selection

    def explain(self, query: str, tool_name: str, result: Dict[str, Any]) -> str:
        return self._explanation


class ClaudeClient:
    """Anthropic-backed client. The only module that imports `anthropic`.

    `_client` is injectable for tests; in production it is constructed from
    the `anthropic` SDK using ANTHROPIC_API_KEY or an explicit api_key (BYOK).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        router_model: str = "claude-haiku-4-5",
        explainer_model: str = "claude-sonnet-4-6",
        _client: Any = None,
    ):
        self.router_model = router_model
        self.explainer_model = explainer_model
        if _client is not None:
            self._client = _client
        else:
            import anthropic  # imported lazily so tests don't require the SDK
            self._client = (
                anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            )

    def select_tool(self, query: str, tools: List[Dict[str, Any]]) -> ToolSelection:
        resp = self._client.messages.create(
            model=self.router_model,
            max_tokens=1024,
            system=ROUTER_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": query}],
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        return ToolSelection(name=block.name, arguments=dict(block.input))

    def explain(self, query: str, tool_name: str, result: Dict[str, Any]) -> str:
        resp = self._client.messages.create(
            model=self.explainer_model,
            max_tokens=1024,
            system=EXPLAINER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": render_explain_prompt(query, tool_name, result)}],
        )
        return next(b.text for b in resp.content if b.type == "text")
