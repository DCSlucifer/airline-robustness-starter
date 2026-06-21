import pytest

from src.ai.factory import make_client, DEFAULT_PROVIDER
from src.ai.llm_client import OpenAIClient, ClaudeClient


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
    with pytest.raises(ValueError):
        make_client("grok", _client=object())
