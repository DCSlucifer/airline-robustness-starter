import pytest

from src.ai.llm_client import LLMResponseError, OpenAIClient, _to_openai_tools
from src.ai.schemas import ToolSelection


class _StubFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string, like the real SDK


class _StubToolCall:
    def __init__(self, function):
        self.function = function


class _StubMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StubChoice:
    def __init__(self, message):
        self.message = message


class _StubCompletion:
    def __init__(self, choices):
        self.choices = choices


class _StubCompletions:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _StubChat:
    def __init__(self, response):
        self.completions = _StubCompletions(response)


class _StubOpenAI:
    def __init__(self, response):
        self.chat = _StubChat(response)


_SAMPLE_SPECS = [
    {
        "name": "geographic_attack",
        "description": "disable airports in a radius",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_km": {"type": "number"},
            },
            "required": ["lat", "lon", "radius_km"],
            "additionalProperties": False,
        },
    }
]


def test_to_openai_tools_translates_format():
    out = _to_openai_tools(_SAMPLE_SPECS)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "geographic_attack"
    assert out[0]["function"]["parameters"]["required"] == ["lat", "lon", "radius_km"]
    assert out[0]["function"]["strict"] is True


def test_openai_client_parses_tool_call():
    fn = _StubFunction("geographic_attack", '{"lat": 40.0, "lon": -74.0, "radius_km": 500}')
    response = _StubCompletion([_StubChoice(_StubMessage(tool_calls=[_StubToolCall(fn)]))])
    client = OpenAIClient(_client=_StubOpenAI(response))

    sel = client.select_tool("storm near NYC", tools=_SAMPLE_SPECS)
    assert isinstance(sel, ToolSelection)
    assert sel.name == "geographic_attack"
    assert sel.arguments["radius_km"] == 500
    kw = client._client.chat.completions.last_kwargs
    assert kw["model"] == "gpt-4o-mini"
    assert kw["tool_choice"] == "required"
    assert kw["tools"][0]["type"] == "function"


def test_openai_client_parses_explanation():
    response = _StubCompletion([_StubChoice(_StubMessage(content="Connectivity dropped 40%."))])
    client = OpenAIClient(_client=_StubOpenAI(response))

    out = client.explain("q", "targeted_attack", {"baseline": {}, "after": {}})
    assert out == "Connectivity dropped 40%."
    assert client._client.chat.completions.last_kwargs["model"] == "gpt-4o-mini"


def test_openai_client_chat_returns_content():
    response = _StubCompletion([_StubChoice(_StubMessage(content="Hello chat."))])
    client = OpenAIClient(_client=_StubOpenAI(response))
    assert client.chat("sys", "user") == "Hello chat."


def test_openai_router_rejects_empty_choices():
    client = OpenAIClient(_client=_StubOpenAI(_StubCompletion([])))

    with pytest.raises(LLMResponseError, match="no choices"):
        client.select_tool("q", tools=_SAMPLE_SPECS)


def test_openai_router_rejects_missing_tool_call():
    response = _StubCompletion([_StubChoice(_StubMessage(content="no tool", tool_calls=[]))])
    client = OpenAIClient(_client=_StubOpenAI(response))

    with pytest.raises(LLMResponseError, match="did not include a tool call"):
        client.select_tool("q", tools=_SAMPLE_SPECS)


def test_openai_router_rejects_invalid_argument_json():
    function = _StubFunction("geographic_attack", "{invalid")
    response = _StubCompletion([_StubChoice(_StubMessage(tool_calls=[_StubToolCall(function)]))])
    client = OpenAIClient(_client=_StubOpenAI(response))

    with pytest.raises(LLMResponseError, match="invalid JSON"):
        client.select_tool("q", tools=_SAMPLE_SPECS)


def test_openai_chat_rejects_refusal_without_echoing_provider_shape():
    message = _StubMessage(content=None)
    message.refusal = "blocked"
    client = OpenAIClient(_client=_StubOpenAI(_StubCompletion([_StubChoice(message)])))

    with pytest.raises(LLMResponseError, match="refused"):
        client.chat("system", "user")
