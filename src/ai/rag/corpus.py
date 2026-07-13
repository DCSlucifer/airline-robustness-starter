"""Create a revision-labelled local Wikipedia snapshot for the RAG advisor."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["Article", "WIKI_ARTICLES", "KB_DIR", "parse_extract", "fetch_corpus"]

KB_DIR = Path("data/kb")
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "airline-robustness/0.1 (+https://github.com/DCSlucifer/airline-robustness-starter)"


@dataclass(frozen=True)
class Article:
    title: str


WIKI_ARTICLES: list[Article] = [
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
        "inprop": "lastrevid",
        "explaintext": "1",
        "format": "json",
        "redirects": "1",
        "titles": title,
    }
    return WIKI_API + "?" + urllib.parse.urlencode(params)


def _default_fetcher(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _slug(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")


def parse_extract(api_json: dict[str, Any]) -> dict[str, Any]:
    """Validate and extract one Wikipedia page from an Action API response."""
    try:
        pages = api_json["query"]["pages"]
        page = next(iter(pages.values()))
    except (AttributeError, KeyError, StopIteration, TypeError) as exc:
        raise ValueError("Wikipedia response does not contain a page") from exc

    title = page.get("title")
    text = page.get("extract")
    revid = page.get("lastrevid")
    if (
        page.get("missing") is not None
        or not title
        or not isinstance(text, str)
        or not text.strip()
    ):
        raise ValueError(f"Wikipedia article is missing or empty: {title!r}")
    if isinstance(revid, bool) or not isinstance(revid, int) or revid <= 0:
        raise ValueError(f"Wikipedia response has no valid revision id for {title!r}")
    return {
        "title": title,
        "text": text,
        "pageid": page.get("pageid"),
        "revid": revid,
    }


def fetch_corpus(
    articles: list[Article] | None = None,
    kb_dir: Path = KB_DIR,
    fetcher: Callable[[str], str] = _default_fetcher,
) -> list[dict[str, Any]]:
    """Fetch every article before replacing the manifest-backed local snapshot."""
    articles = articles if articles is not None else WIKI_ARTICLES
    if not articles:
        raise ValueError("at least one Wikipedia article is required")

    fetched_at = datetime.now(timezone.utc).isoformat()
    snapshots: list[tuple[str, str]] = []
    manifest: list[dict[str, Any]] = []
    slugs: set[str] = set()
    for article in articles:
        parsed = parse_extract(json.loads(fetcher(_api_url(article.title))))
        revid = parsed["revid"]
        url = "https://en.wikipedia.org/w/index.php?" + urllib.parse.urlencode(
            {"title": parsed["title"], "oldid": revid}
        )
        slug = _slug(parsed["title"])
        if not slug or slug in slugs:
            raise ValueError(f"Wikipedia titles produced an empty or duplicate slug: {slug!r}")
        slugs.add(slug)
        header = f"# {parsed['title']}\n\n<!-- source: {url} | revid: {revid} -->\n\n"
        snapshots.append((slug, header + parsed["text"].strip() + "\n"))
        manifest.append(
            {
                "title": parsed["title"],
                "slug": slug,
                "url": url,
                "revid": revid,
                "fetched_at": fetched_at,
            }
        )

    destination = Path(kb_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for slug, body in snapshots:
        (destination / f"{slug}.md").write_text(body, encoding="utf-8")
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network, one-shot
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", type=Path, default=KB_DIR)
    args = parser.parse_args(argv)
    try:
        entries = fetch_corpus(kb_dir=args.kb_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Cached {len(entries)} articles to {args.kb_dir}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
