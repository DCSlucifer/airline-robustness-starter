"""Provider-swappable LLM clients with validated, provider-neutral responses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol

from .prompts import (
    EXPLAINER_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    render_explain_prompt,
)
from .schemas import ToolSelection

__all__ = [
    "LLMClient",
    "LLMResponseError",
    "LLMConfigurationError",
    "FakeLLMClient",
    "ClaudeClient",
    "OpenAIClient",
]


class LLMResponseError(RuntimeError):
    """A provider returned a refusal or a response that cannot satisfy the client contract."""


class LLMConfigurationError(ValueError):
    """The selected provider cannot run with the supplied local configuration."""


def _claude_content(response: Any, operation: str) -> list[Any]:
    """Return Anthropic content blocks or raise a stable domain error."""
    if getattr(response, "stop_reason", None) == "refusal":
        raise LLMResponseError(f"Anthropic refused the {operation} request")

    content = getattr(response, "content", None)
    try:
        blocks = list(content) if content is not None and not isinstance(content, str) else []
    except TypeError as exc:
        raise LLMResponseError(
            f"Anthropic {operation} response contained malformed content"
        ) from exc

    if not blocks:
        raise LLMResponseError(f"Anthropic {operation} response contained no content blocks")
    if any(getattr(block, "type", None) == "refusal" for block in blocks):
        raise LLMResponseError(f"Anthropic refused the {operation} request")
    return blocks


def _claude_text(response: Any, operation: str) -> str:
    for block in _claude_content(response, operation):
        if getattr(block, "type", None) != "text":
            continue
        value = getattr(block, "text", None)
        if isinstance(value, str) and value.strip():
            return value
    raise LLMResponseError(f"Anthropic {operation} response did not include non-empty text")


def _openai_message(response: Any, operation: str) -> Any:
    """Return the first OpenAI message or raise a stable domain error."""
    choices = getattr(response, "choices", None)
    try:
        choice_list = list(choices) if choices is not None and not isinstance(choices, str) else []
    except TypeError as exc:
        raise LLMResponseError(f"OpenAI {operation} response contained malformed choices") from exc

    if not choice_list:
        raise LLMResponseError(f"OpenAI {operation} response contained no choices")

    choice = choice_list[0]
    if getattr(choice, "finish_reason", None) == "content_filter":
        raise LLMResponseError(f"OpenAI blocked the {operation} response with its content filter")
    message = getattr(choice, "message", None)
    if message is None:
        raise LLMResponseError(f"OpenAI {operation} response did not include a message")
    if getattr(message, "refusal", None):
        raise LLMResponseError(f"OpenAI refused the {operation} request")
    return message


def _openai_text(response: Any, operation: str) -> str:
    content = getattr(_openai_message(response, operation), "content", None)
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError(f"OpenAI {operation} response did not include non-empty text")
    return content


class LLMClient(Protocol):
    """The orchestrator depends only on this interface, never on a provider SDK."""

    def select_tool(self, query: str, tools: list[dict[str, Any]]) -> ToolSelection: ...

    def explain(self, query: str, tool_name: str, result: dict[str, Any]) -> str: ...

    def chat(self, system: str, user: str) -> str: ...


class FakeLLMClient:
    """Deterministic client for offline tests; returns preset values."""

    def __init__(
        self,
        selection: ToolSelection,
        explanation: str = "(no explanation)",
        chat_response: str = "(chat)",
    ):
        self._selection = selection
        self._explanation = explanation
        self._chat_response = chat_response

    def select_tool(self, query: str, tools: list[dict[str, Any]]) -> ToolSelection:
        return self._selection

    def explain(self, query: str, tool_name: str, result: dict[str, Any]) -> str:
        return self._explanation

    def chat(self, system: str, user: str) -> str:
        return self._chat_response


class ClaudeClient:
    """Anthropic-backed client. The only module that imports `anthropic`.

    `_client` is injectable for tests; in production it is constructed from
    the `anthropic` SDK using ANTHROPIC_API_KEY or an explicit api_key (BYOK).
    """

    def __init__(
        self,
        api_key: str | None = None,
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

    def select_tool(self, query: str, tools: list[dict[str, Any]]) -> ToolSelection:
        resp = self._client.messages.create(
            model=self.router_model,
            max_tokens=1024,
            system=ROUTER_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": query}],
        )
        block = next(
            (
                candidate
                for candidate in _claude_content(resp, "router")
                if getattr(candidate, "type", None) == "tool_use"
            ),
            None,
        )
        if block is None:
            raise LLMResponseError("Anthropic router response did not include a tool call")

        name = getattr(block, "name", None)
        arguments = getattr(block, "input", None)
        if not isinstance(name, str) or not name.strip():
            raise LLMResponseError("Anthropic router tool call did not include a valid name")
        if not isinstance(arguments, Mapping):
            raise LLMResponseError("Anthropic router tool call arguments were not an object")
        try:
            return ToolSelection(name=name, arguments=dict(arguments))
        except (TypeError, ValueError) as exc:
            raise LLMResponseError("Anthropic router tool call was malformed") from exc

    def explain(self, query: str, tool_name: str, result: dict[str, Any]) -> str:
        resp = self._client.messages.create(
            model=self.explainer_model,
            max_tokens=1024,
            system=EXPLAINER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": render_explain_prompt(query, tool_name, result)}],
        )
        return _claude_text(resp, "explanation")

    def chat(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self.explainer_model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return _claude_text(resp, "chat")


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic-style TOOL_SPECS into OpenAI function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
                "strict": True,
            },
        }
        for t in tools
    ]


class OpenAIClient:
    """OpenAI-backed client. The only module besides ClaudeClient that talks to a provider SDK.

    `_client` is injectable for tests; in production it is constructed from the `openai` SDK using
    OPENAI_API_KEY or an explicit api_key (BYOK).
    """

    def __init__(
        self,
        api_key: str | None = None,
        router_model: str = "gpt-4o-mini",
        explainer_model: str = "gpt-4o-mini",
        _client: Any = None,
    ):
        self.router_model = router_model
        self.explainer_model = explainer_model
        if _client is not None:
            self._client = _client
        else:
            import openai  # imported lazily so tests don't require the SDK

            self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def select_tool(self, query: str, tools: list[dict[str, Any]]) -> ToolSelection:
        resp = self._client.chat.completions.create(
            model=self.router_model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            tools=_to_openai_tools(tools),
            tool_choice="required",
        )
        message = _openai_message(resp, "router")
        tool_calls = getattr(message, "tool_calls", None)
        try:
            calls = (
                list(tool_calls)
                if tool_calls is not None and not isinstance(tool_calls, str)
                else []
            )
        except TypeError as exc:
            raise LLMResponseError("OpenAI router response contained malformed tool calls") from exc
        if not calls:
            raise LLMResponseError("OpenAI router response did not include a tool call")

        function = getattr(calls[0], "function", None)
        name = getattr(function, "name", None)
        raw_arguments = getattr(function, "arguments", None)
        if not isinstance(name, str) or not name.strip():
            raise LLMResponseError("OpenAI router tool call did not include a valid name")
        if not isinstance(raw_arguments, str):
            raise LLMResponseError("OpenAI router tool call arguments were not JSON text")
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("OpenAI router tool call arguments were invalid JSON") from exc
        if not isinstance(arguments, dict):
            raise LLMResponseError("OpenAI router tool call arguments were not a JSON object")
        try:
            return ToolSelection(name=name, arguments=arguments)
        except (TypeError, ValueError) as exc:
            raise LLMResponseError("OpenAI router tool call was malformed") from exc

    def explain(self, query: str, tool_name: str, result: dict[str, Any]) -> str:
        resp = self._client.chat.completions.create(
            model=self.explainer_model,
            messages=[
                {"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
                {"role": "user", "content": render_explain_prompt(query, tool_name, result)},
            ],
        )
        return _openai_text(resp, "explanation")

    def chat(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.explainer_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return _openai_text(resp, "chat")
