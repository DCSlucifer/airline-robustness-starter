from src.ai.rag.embedder import FakeEmbedder, OpenAIEmbedder


def test_fake_embedder_is_deterministic_and_fixed_dim():
    emb = FakeEmbedder(dim=16)
    assert emb.embed(["hello"]) == emb.embed(["hello"])
    assert len(emb.embed(["hello"])[0]) == 16


def test_fake_embedder_different_texts_differ():
    va, vb = FakeEmbedder(dim=16).embed(["alpha", "beta"])
    assert va != vb


def test_fake_embedder_unit_norm():
    v = FakeEmbedder(dim=16).embed(["x"])[0]
    assert abs(sum(c * c for c in v) ** 0.5 - 1.0) < 1e-9


class _StubEmbItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _StubEmbResponse:
    def __init__(self, data):
        self.data = data


class _StubEmbeddings:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _StubOpenAI:
    def __init__(self, response):
        self.embeddings = _StubEmbeddings(response)


def test_openai_embedder_parses_response_and_uses_model():
    response = _StubEmbResponse([_StubEmbItem([0.1, 0.2]), _StubEmbItem([0.3, 0.4])])
    emb = OpenAIEmbedder(_client=_StubOpenAI(response))
    assert emb.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert emb._client.embeddings.last_kwargs["model"] == "text-embedding-3-small"
    assert emb._client.embeddings.last_kwargs["input"] == ["a", "b"]
