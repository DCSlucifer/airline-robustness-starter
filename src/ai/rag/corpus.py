"""Wikipedia corpus: an article manifest and an offline-caching, reproducible fetcher.

`fetch_corpus` records each article's resolved revision id and writes plain-text extracts to
`data/kb/`. The cached files are committed, so the index rebuilds offline and deterministically.
"""
from __future__ import annotations
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Article", "WIKI_ARTICLES", "KB_DIR", "parse_extract", "fetch_corpus"]

KB_DIR = Path("data/kb")
WIKI_API = "https://en.wikipedia.org/w/api.php"


@dataclass(frozen=True)
class Article:
    title: str


WIKI_ARTICLES: List[Article] = [
    Article("Network science"),
    Article("Centrality"),
    Article("Betweenness centrality"),
    Article("Scale-free network"),
    Article("Robustness of complex networks"),
    Article("Spoke–hub distribution paradigm"),
    Article("Airline hub"),
    Article("Air travel disruption after the 2010 Eyjafjallajökull eruption"),
    Article("Impact of the COVID-19 pandemic on aviation"),
    Article("Flight cancellation and delay"),
]


def _api_url(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts|info",
        "inprop": "lastrevid",  # required for the API to return lastrevid (pinned-revision citations)
        "explaintext": "1",
        "format": "json",
        "redirects": "1",
        "titles": title,
    }
    return WIKI_API + "?" + urllib.parse.urlencode(params)


def _default_fetcher(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed Wikipedia host
        return resp.read().decode("utf-8")


def _slug(title: str) -> str:
    return title.lower().replace(" ", "-").replace("/", "-")


def parse_extract(api_json: Dict[str, Any]) -> Dict[str, Any]:
    """Pull title, plain text, pageid, revid from a Wikipedia Action API response."""
    page = next(iter(api_json["query"]["pages"].values()))
    return {
        "title": page["title"],
        "text": page.get("extract", ""),
        "pageid": page.get("pageid"),
        "revid": page.get("lastrevid"),
    }


def fetch_corpus(
    articles: Optional[List[Article]] = None,
    kb_dir: Path = KB_DIR,
    fetcher: Callable[[str], str] = _default_fetcher,
) -> List[Dict[str, Any]]:
    """Fetch each article's plain-text extract, cache it to kb_dir, return manifest entries."""
    articles = articles if articles is not None else WIKI_ARTICLES
    kb_dir = Path(kb_dir)
    kb_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for art in articles:
        parsed = parse_extract(json.loads(fetcher(_api_url(art.title))))
        revid = parsed["revid"]
        url = (
            "https://en.wikipedia.org/w/index.php?"
            + urllib.parse.urlencode({"title": parsed["title"], "oldid": revid})
        )
        slug = _slug(parsed["title"])
        header = (
            f"# {parsed['title']}\n\n"
            f"<!-- source: {url} | revid: {revid} | fetched: {datetime.now(timezone.utc).isoformat()} -->\n\n"
        )
        (kb_dir / f"{slug}.md").write_text(header + parsed["text"], encoding="utf-8")
        manifest.append({"title": parsed["title"], "slug": slug, "url": url, "revid": revid})
    (kb_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:  # pragma: no cover - network, one-shot
    entries = fetch_corpus()
    print(f"Cached {len(entries)} articles to {KB_DIR}/")


if __name__ == "__main__":  # pragma: no cover
    main()
