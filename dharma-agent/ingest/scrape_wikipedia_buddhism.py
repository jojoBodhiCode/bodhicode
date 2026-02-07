"""
Scrape Wikipedia Buddhism articles via the MediaWiki API.

Crawls the Category:Buddhism category tree recursively, discovers articles,
downloads their plain-text extracts, and saves them as JSON locally with a
manifest for resumability.

Usage:
  python -m ingest.scrape_wikipedia_buddhism C:/llama-cpp/wikipedia-buddhism-data
  python -m ingest.scrape_wikipedia_buddhism C:/llama-cpp/wikipedia-buddhism-data --max-depth 3 --max-articles 500
  python -m ingest.scrape_wikipedia_buddhism C:/llama-cpp/wikipedia-buddhism-data --retry-errors
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote

try:
    import requests
except ImportError:
    print("Missing 'requests'. Install with: pip install requests")
    sys.exit(1)


# ─── Constants ────────────────────────────────────────────────────────────────

API_URL = "https://en.wikipedia.org/w/api.php"
ROOT_CATEGORY = "Category:Buddhism"
DEFAULT_DELAY = 1.0
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
USER_AGENT = "DharmaScholar-Bot/1.0 (Buddhist text research project)"

MANIFEST_FILE = "_manifest.json"

DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_ARTICLES = 2000

# Categories to skip: overly broad, meta, or off-topic subcategories
CATEGORY_BLACKLIST = {
    # Meta / maintenance
    "Category:Wikipedia articles",
    "Category:All articles",
    "Category:Articles needing",
    "Category:CS1",
    "Category:Webarchive",
    "Category:All stub articles",
    "Category:Buddhism stubs",
    "Category:Pages",
    "Category:Use dmy dates",
    "Category:Use mdy dates",
    "Category:Commons category",
    # Overly broad geographic
    "Category:Religion in Asia",
    "Category:Asian culture",
    "Category:Religion by country",
    "Category:Indian philosophy",
    "Category:Chinese philosophy",
    "Category:Japanese culture",
    # Other religions (branch out from comparative categories)
    "Category:Hinduism",
    "Category:Jainism",
    "Category:Taoism",
    "Category:Confucianism",
    "Category:Sikhism",
    "Category:Christianity",
    "Category:Islam",
    "Category:Judaism",
    "Category:New religious movements",
    # Too broad
    "Category:Vegetarianism",
    "Category:Yoga",
    "Category:Meditation",
}

# Substring-based blacklist: skip any category whose title contains these
CATEGORY_BLACKLIST_SUBSTRINGS = [
    "wikipedia", "articles", "pages", "stubs", "cs1", "webarchive",
    "use dmy", "use mdy", "all accuracy", "accuracy disputes",
    "lacking sources", "needing", "cleanup", "templates",
]


# ─── API helpers ──────────────────────────────────────────────────────────────

def api_request(session: requests.Session, params: dict, delay: float) -> Optional[dict]:
    """Make a MediaWiki API request with rate limiting and retries."""
    time.sleep(delay)
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = (attempt + 1) * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            elif resp.status_code >= 500:
                time.sleep((attempt + 1) * 2)
                continue
            else:
                return None
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 2)
            continue
    return None


def is_category_allowed(category_title: str) -> bool:
    """Check if a category should be crawled (not blacklisted)."""
    if category_title in CATEGORY_BLACKLIST:
        return False
    title_lower = category_title.lower()
    return not any(sub in title_lower for sub in CATEGORY_BLACKLIST_SUBSTRINGS)


# ─── Category tree crawler ───────────────────────────────────────────────────

def discover_category_members(
    session: requests.Session,
    category: str,
    delay: float,
) -> Tuple[List[str], List[str]]:
    """
    Fetch all members of a Wikipedia category.

    Returns:
        (article_titles, subcategory_titles)
    """
    articles = []
    subcategories = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "500",
            "cmprop": "title|type",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = api_request(session, params, delay)
        if not data or "query" not in data:
            break

        for member in data["query"].get("categorymembers", []):
            if member["type"] == "page":
                articles.append(member["title"])
            elif member["type"] == "subcat":
                subcategories.append(member["title"])

        if "continue" in data:
            cmcontinue = data["continue"].get("cmcontinue")
        else:
            break

    return articles, subcategories


def crawl_category_tree(
    session: requests.Session,
    root_category: str,
    max_depth: int,
    max_articles: int,
    delay: float,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Recursively crawl the category tree using BFS.

    Prioritizes categories closest to the root for relevance.

    Returns:
        (article_titles, article_categories) where article_categories
        maps article_title -> [list of categories it was found in]
    """
    visited_categories: Set[str] = set()
    article_titles: List[str] = []
    article_set: Set[str] = set()
    article_categories: Dict[str, List[str]] = {}

    # BFS queue: (category_title, depth)
    queue: List[Tuple[str, int]] = [(root_category, 0)]

    while queue and len(article_set) < max_articles:
        category, depth = queue.pop(0)

        if category in visited_categories:
            continue
        if not is_category_allowed(category):
            continue

        visited_categories.add(category)

        print(f"    [depth={depth}] Scanning {category} "
              f"({len(article_set)} articles so far)...")

        articles, subcats = discover_category_members(session, category, delay)

        for title in articles:
            if title not in article_set and len(article_set) < max_articles:
                article_set.add(title)
                article_titles.append(title)
                article_categories.setdefault(title, []).append(category)
            elif title in article_set:
                article_categories.setdefault(title, []).append(category)

        if depth < max_depth:
            for subcat in subcats:
                if subcat not in visited_categories:
                    queue.append((subcat, depth + 1))

    print(f"    Crawled {len(visited_categories)} categories, "
          f"discovered {len(article_titles)} articles")
    return article_titles, article_categories


# ─── Article downloader ──────────────────────────────────────────────────────

def fetch_article_text(
    session: requests.Session,
    title: str,
    delay: float,
) -> Optional[dict]:
    """
    Fetch a Wikipedia article's plain text and metadata via the API.

    Returns dict with title, text, pageid, categories, url, length
    or None on failure / disambiguation / too-short.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|categories|pageprops|info",
        "explaintext": "true",
        "exsectionformat": "plain",
        "inprop": "url",
        "cllimit": "50",
        "clshow": "!hidden",
    }

    data = api_request(session, params, delay)
    if not data or "query" not in data:
        return None

    pages = data["query"].get("pages", [])
    if not pages:
        return None

    page = pages[0] if isinstance(pages, list) else list(pages.values())[0]

    # Skip missing or disambiguation pages
    if page.get("missing"):
        return None
    if "disambiguation" in page.get("pageprops", {}):
        return None

    extract = page.get("extract", "")
    if not extract or len(extract) < 200:
        return None

    categories = [c["title"].replace("Category:", "")
                  for c in page.get("categories", [])]

    return {
        "title": page.get("title", title),
        "pageid": page.get("pageid", 0),
        "text": extract,
        "categories": categories,
        "url": page.get("fullurl",
                        f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"),
        "length": page.get("length", len(extract)),
    }


# ─── File helpers ─────────────────────────────────────────────────────────────

def title_to_filename(title: str) -> str:
    """Convert an article title to a safe filename."""
    safe = re.sub(r'[<>:"/\\|?*]', '_', title)
    safe = re.sub(r'\s+', '_', safe)
    return safe[:200]


def load_manifest(output_dir: Path) -> Dict:
    """Load the download manifest for resumability."""
    manifest_path = output_dir / MANIFEST_FILE
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "articles": {},
        "discovered_titles": [],
        "article_categories": {},
        "crawl_completed": False,
        "settings": {},
    }


def save_manifest(manifest: Dict, output_dir: Path):
    """Save the download manifest atomically."""
    manifest_path = output_dir / MANIFEST_FILE
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp_path.replace(manifest_path)


# ─── Main scrape function ────────────────────────────────────────────────────

def scrape_wikipedia_buddhism(
    output_dir: str,
    delay: float = DEFAULT_DELAY,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_articles: Optional[int] = DEFAULT_MAX_ARTICLES,
    retry_errors: bool = False,
):
    """
    Scrape Wikipedia Buddhism articles to a local directory.

    Two phases:
      1. Category tree crawl to discover article titles
      2. Download each article's plain text via the API
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    articles_dir = out / "articles"
    articles_dir.mkdir(exist_ok=True)

    manifest = load_manifest(out)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Phase 1: Category tree crawl (or use cached discovery)
    if manifest.get("discovered_titles") and manifest.get("crawl_completed"):
        all_titles = manifest["discovered_titles"]
        article_categories = manifest.get("article_categories", {})
        print(f"  Using cached article list: {len(all_titles)} articles")
    else:
        print(f"  Crawling Wikipedia category tree from {ROOT_CATEGORY}...")
        print(f"  Max depth: {max_depth}, Max articles: {max_articles}")
        all_titles, article_categories = crawl_category_tree(
            session, ROOT_CATEGORY, max_depth,
            max_articles or 999999, delay,
        )
        manifest["discovered_titles"] = all_titles
        manifest["article_categories"] = article_categories
        manifest["crawl_completed"] = True
        manifest["settings"] = {
            "max_depth": max_depth,
            "max_articles": max_articles,
            "root_category": ROOT_CATEGORY,
        }
        save_manifest(manifest, out)
        print(f"  Discovery complete: {len(all_titles)} articles found")

    # Phase 2: Download articles
    already_done = {
        title for title, info in manifest["articles"].items()
        if info.get("status") == "ok"
    }
    if retry_errors:
        to_download = [t for t in all_titles if t not in already_done]
    else:
        already_attempted = set(manifest["articles"].keys())
        to_download = [t for t in all_titles if t not in already_attempted]

    total = len(to_download)
    if total == 0:
        print(f"  All {len(already_done)} articles already downloaded. Nothing to do.")
        return

    print(f"  Downloading {total} articles ({len(already_done)} already cached)...")

    try:
        for i, title in enumerate(to_download):
            result = fetch_article_text(session, title, delay)

            if result:
                filename = title_to_filename(title) + ".json"
                filepath = articles_dir / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                manifest["articles"][title] = {
                    "path": f"articles/{filename}",
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "status": "ok",
                    "pageid": result.get("pageid", 0),
                }
            else:
                manifest["articles"][title] = {
                    "path": "",
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "status": "skipped",
                }

            if (i + 1) % 25 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                print(f"  Downloaded {i + 1}/{total} ({pct:.0f}%)")
                save_manifest(manifest, out)

    except KeyboardInterrupt:
        print("\n  Interrupted. Saving progress...")
    finally:
        save_manifest(manifest, out)

    ok = sum(1 for v in manifest["articles"].values() if v["status"] == "ok")
    errors = sum(1 for v in manifest["articles"].values()
                 if v["status"] in ("error", "skipped"))
    print(f"\n  Done. {ok} articles saved, {errors} skipped/errors.")
    if errors:
        print(f"  Re-run with --retry-errors to retry failed articles.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Wikipedia Buddhism articles via MediaWiki API"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save article JSON files "
             "(e.g. C:/llama-cpp/wikipedia-buddhism-data)",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"Seconds between requests (default {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
        help=f"Maximum category tree depth (default {DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument(
        "--max-articles", type=int, default=DEFAULT_MAX_ARTICLES,
        help=f"Maximum articles to discover (default {DEFAULT_MAX_ARTICLES})",
    )
    parser.add_argument(
        "--retry-errors", action="store_true",
        help="Retry previously failed/skipped articles",
    )
    args = parser.parse_args()

    scrape_wikipedia_buddhism(
        output_dir=args.output_dir,
        delay=args.delay,
        max_depth=args.max_depth,
        max_articles=args.max_articles,
        retry_errors=args.retry_errors,
    )
