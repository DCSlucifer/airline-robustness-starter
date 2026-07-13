import pytest

from src.ai.factory import DEFAULT_PROVIDER, make_client, resolve_provider
from src.ai.llm_client import ClaudeClient, LLMConfigurationError, OpenAIClient


def test_default_provider_is_openai():
    assert DEFAULT_PROVIDER == "openai"
    client = make_client(_client=object())
    assert isinstance(client, OpenAIClient)


def test_explicit_anthropic():
    client = make_client("anthropic", _client=object())
    assert isinstance(client, ClaudeClient)


def test_claude_alias():
    client = make_client("claude", _client=object())
    assert isinstance(client, ClaudeClient)


def test_env_var_selects_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = make_client(_client=object())
    assert isinstance(client, ClaudeClient)


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    client = make_client("openai", _client=object())
    assert isinstance(client, OpenAIClient)


def test_unknown_provider_raises():
    with pytest.raises(LLMConfigurationError):
        make_client("grok", _client=object())


def test_resolve_provider_normalizes_alias_and_whitespace(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  CLAUDE  ")

    assert resolve_provider() == "anthropic"
    assert resolve_provider(" OpenAI ") == "openai"
