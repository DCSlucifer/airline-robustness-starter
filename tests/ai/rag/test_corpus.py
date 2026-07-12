import json

from src.ai.rag.corpus import Article, _api_url, fetch_corpus, parse_extract


def test_api_url_requests_lastrevid():
    # The Action API only returns lastrevid when inprop=lastrevid is requested;
    # without it, real builds get revid=None and citations lose their pinned oldid.
    url = _api_url("Network science")
    assert "inprop=lastrevid" in url


def _api_json(title, text, revid):
    return {
        "query": {
            "pages": {"1": {"title": title, "extract": text, "pageid": 1, "lastrevid": revid}}
        }
    }


def test_parse_extract_pulls_fields():
    parsed = parse_extract(_api_json("Network science", "Body text.", 999))
    assert parsed["title"] == "Network science"
    assert parsed["text"] == "Body text."
    assert parsed["revid"] == 999


def test_fetch_corpus_caches_files_and_manifest(tmp_path):
    arts = [Article("Network science"), Article("Centrality")]
    responses = {
        "Network+science": json.dumps(_api_json("Network science", "Net body.", 11)),
        "Centrality": json.dumps(_api_json("Centrality", "Cent body.", 22)),
    }

    def fake_fetcher(url):
        for key, body in responses.items():
            if key in url:
                return body
        raise AssertionError(f"unexpected url: {url}")

    manifest = fetch_corpus(arts, kb_dir=tmp_path, fetcher=fake_fetcher)
    assert len(manifest) == 2
    assert (tmp_path / "manifest.json").exists()
    body = (tmp_path / "network-science.md").read_text(encoding="utf-8")
    assert "Net body." in body
    assert "oldid=11" in body
