"""
Lightweight Wikipedia lookup for the Dharma Scholar agent.

Provides real-time article search and summary fetching via the MediaWiki API.
Used as a fallback during deep research when the local knowledge base (RAG)
returns thin results. No API key required.

Design constraints:
  - Fetches intro sections only (not full articles) to keep context small
  - Max 3 articles per lookup to stay within LLM context limits
  - 0.5s delay between requests to respect Wikipedia rate limits
  - Results are formatted as plain text ready to inject into prompts
"""

import json
import re
from urllib.parse import quote, urlparse

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# Also support http.client for the Discord bot (which avoids requests)
import http.client


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "DharmaScholar-Bot/1.0 (Buddhist text research project)"
REQUEST_TIMEOUT = 15
LOOKUP_DELAY = 0.5  # seconds between API calls


def _api_get(params):
    """
    Make a MediaWiki API GET request.

    Uses requests if available, falls back to http.client.
    Returns parsed JSON dict or None.
    """
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")

    if _HAS_REQUESTS:
        try:
            resp = requests.get(
                API_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    # Fallback: http.client
    try:
        query_string = "&".join(f"{k}={quote(str(v))}" for k, v in params.items())
        conn = http.client.HTTPSConnection("en.wikipedia.org", timeout=REQUEST_TIMEOUT)
        conn.request(
            "GET",
            f"/w/api.php?{query_string}",
            headers={"User-Agent": USER_AGENT},
        )
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        if resp.status == 200:
            return json.loads(raw)
    except Exception:
        pass
    return None


def search_articles(query, limit=5):
    """
    Search Wikipedia for articles matching a query.

    Returns a list of {"title": str, "snippet": str, "pageid": int}.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "srprop": "snippet",
    }
    data = _api_get(params)
    if not data or "query" not in data:
        return []

    results = []
    for hit in data["query"].get("search", []):
        # Strip HTML from snippet
        snippet = re.sub(r'<[^>]+>', '', hit.get("snippet", ""))
        results.append({
            "title": hit["title"],
            "snippet": snippet,
            "pageid": hit.get("pageid", 0),
        })
    return results


def fetch_summary(title):
    """
    Fetch the intro section of a Wikipedia article (plain text).

    Returns {"title": str, "text": str, "url": str} or None.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "exintro": "true",
        "explaintext": "true",
        "inprop": "url",
    }
    data = _api_get(params)
    if not data or "query" not in data:
        return None

    pages = data["query"].get("pages", [])
    if not pages:
        return None

    page = pages[0] if isinstance(pages, list) else list(pages.values())[0]

    if page.get("missing"):
        return None

    extract = page.get("extract", "")
    if not extract or len(extract) < 50:
        return None

    return {
        "title": page.get("title", title),
        "text": extract,
        "url": page.get("fullurl",
                        f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"),
    }


def lookup(query, max_articles=3, max_chars=3000):
    """
    Search Wikipedia and fetch summaries for the top matching articles.

    This is the main entry point for the deep research agent.

    Args:
        query: Search query string.
        max_articles: Maximum articles to fetch (default 3).
        max_chars: Truncate total text to this many characters.

    Returns:
        (context_str, source_urls) tuple.
        context_str: Formatted text block ready for prompt injection.
        source_urls: List of Wikipedia URLs for citation.
    """
    results = search_articles(query, limit=max_articles + 2)
    if not results:
        return "", []

    import time
    articles = []
    urls = []

    for hit in results[:max_articles]:
        time.sleep(LOOKUP_DELAY)
        summary = fetch_summary(hit["title"])
        if summary and summary["text"]:
            articles.append(summary)
            urls.append(summary["url"])

    if not articles:
        return "", []

    # Format as context block
    parts = []
    total_chars = 0
    for art in articles:
        text = art["text"]
        # Truncate individual articles if needed
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "..."

        parts.append(
            f"### {art['title']}\n"
            f"(Source: Wikipedia — {art['url']})\n\n"
            f"{text}"
        )
        total_chars += len(text)

    context_str = "\n\n---\n\n".join(parts)
    return context_str, urls
