from src.ai.schemas import ToolSelection
from src.ai.llm_client import FakeLLMClient, ClaudeClient
from src.ai.prompts import render_explain_prompt


def test_fake_client_returns_canned_selection_and_explanation():
    fake = FakeLLMClient(
        selection=ToolSelection(name="targeted_attack", arguments={"metric": "degree", "k": 3}),
        explanation="Hubs removed; connectivity fell.",
    )
    sel = fake.select_tool("lose 3 hubs", tools=[])
    assert sel.name == "targeted_attack"
    assert fake.explain("q", "targeted_attack", {}) == "Hubs removed; connectivity fell."


def test_render_explain_prompt_includes_metrics():
    prompt = render_explain_prompt(
        "what if?", "targeted_attack", {"baseline": {"gwcc_frac": 1.0}, "after": {"gwcc_frac": 0.6}}
    )
    assert "targeted_attack" in prompt
    assert "gwcc_frac" in prompt


class _StubBlock:
    def __init__(self, type, name=None, input=None, text=None):
        self.type = type
        self.name = name
        self.input = input
        self.text = text


class _StubResponse:
    def __init__(self, content):
        self.content = content


class _StubMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _StubAnthropic:
    def __init__(self, response):
        self.messages = _StubMessages(response)


def test_claude_client_parses_tool_use_block():
    response = _StubResponse([_StubBlock("tool_use", name="geographic_attack",
                                         input={"lat": 40.0, "lon": -74.0, "radius_km": 500})])
    client = ClaudeClient(_client=_StubAnthropic(response))
    sel = client.select_tool("storm near NYC", tools=[{"name": "geographic_attack"}])
    assert sel.name == "geographic_attack"
    assert sel.arguments["radius_km"] == 500
    assert client._client.messages.last_kwargs["model"] == "claude-haiku-4-5"
    assert client._client.messages.last_kwargs["tool_choice"] == {"type": "any"}


def test_claude_client_parses_explanation_text():
    response = _StubResponse([_StubBlock("text", text="Connectivity dropped 40%.")])
    client = ClaudeClient(_client=_StubAnthropic(response))
    out = client.explain("q", "targeted_attack", {"baseline": {}, "after": {}})
    assert out == "Connectivity dropped 40%."
    assert client._client.messages.last_kwargs["model"] == "claude-sonnet-4-6"


def test_claude_client_chat_returns_text():
    response = _StubResponse([_StubBlock("text", text="Hello from chat.")])
    client = ClaudeClient(_client=_StubAnthropic(response))
    assert client.chat("sys", "user") == "Hello from chat."
