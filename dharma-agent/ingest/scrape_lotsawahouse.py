"""
Scrape Lotsawa House (lotsawahouse.org) text pages to a local directory.

Lotsawa House hosts 6,000+ Tibetan Buddhist texts under CC BY-NC 4.0.
No bulk download or sitemap is available, so we crawl index pages to
discover individual text URLs, then fetch and save each HTML page.

Usage:
  python -m ingest.scrape_lotsawahouse C:/llama-cpp/lotsawahouse-data
  python -m ingest.scrape_lotsawahouse C:/llama-cpp/lotsawahouse-data --max-pages 10
  python -m ingest.scrape_lotsawahouse C:/llama-cpp/lotsawahouse-data --retry-errors
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependencies. Install with: pip install requests beautifulsoup4")
    sys.exit(1)


BASE_URL = "https://www.lotsawahouse.org"
DEFAULT_DELAY = 1.5  # seconds between requests
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
USER_AGENT = "DharmaScholar-Bot/1.0 (Buddhist text research; CC BY-NC 4.0 compliance)"

# Paths disallowed by robots.txt
DISALLOWED = ["/app/", "/search", "/Cgi"]

# Index pages for URL discovery
INDEX_SECTIONS = [
    "/words-of-the-buddha/",
    "/tibetan-masters/",
    "/indian-masters/",
]

MANIFEST_FILE = "_manifest.json"


def is_allowed(url: str) -> bool:
    """Check URL against robots.txt disallow rules."""
    path = urlparse(url).path
    return not any(path.startswith(d) for d in DISALLOWED)


def fetch_page(url: str, session: requests.Session, delay: float) -> Optional[str]:
    """Fetch a page with rate limiting, retries, and polite headers."""
    time.sleep(delay)

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
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


def discover_text_urls_from_index(
    session: requests.Session, delay: float
) -> Set[str]:
    """Discover text page URLs from /words-of-the-buddha/."""
    urls = set()

    html = fetch_page(f"{BASE_URL}/words-of-the-buddha/", session, delay)
    if not html:
        print("  Warning: Could not fetch /words-of-the-buddha/")
        return urls

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/words-of-the-buddha/") and href != "/words-of-the-buddha/":
            # Must be a text page, not just the index
            slug = href.rstrip("/").split("/")[-1]
            if slug and slug != "words-of-the-buddha":
                full_url = urljoin(BASE_URL, href)
                if is_allowed(full_url):
                    urls.add(full_url)

    return urls


def discover_master_pages(
    section: str, session: requests.Session, delay: float
) -> List[str]:
    """Discover master page URLs from a section index (tibetan-masters or indian-masters)."""
    master_urls = []

    html = fetch_page(f"{BASE_URL}{section}", session, delay)
    if not html:
        print(f"  Warning: Could not fetch {section}")
        return master_urls

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(section) and href != section:
            # Master page URL: /tibetan-masters/name/ (exactly one segment after section)
            parts = href.rstrip("/").split("/")
            if len(parts) == 3:  # ['', 'tibetan-masters', 'name']
                full_url = urljoin(BASE_URL, href)
                if full_url not in master_urls:
                    master_urls.append(full_url)

    return master_urls


def discover_text_urls_from_master(
    master_url: str, session: requests.Session, delay: float
) -> Set[str]:
    """Discover text page URLs from a master's page."""
    urls = set()

    html = fetch_page(master_url, session, delay)
    if not html:
        return urls

    parsed = urlparse(master_url)
    master_path = parsed.path.rstrip("/")  # e.g., /tibetan-masters/jigme-lingpa

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Text pages are one level deeper than the master page
        if href.startswith(master_path + "/") and href != master_path + "/":
            full_url = urljoin(BASE_URL, href)
            if is_allowed(full_url):
                urls.add(full_url)

    return urls


def url_to_filepath(url: str, output_dir: Path) -> Path:
    """Convert a URL to a local file path mirroring the URL structure."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if not path.endswith(".html"):
        path += ".html"
    return output_dir / path


def load_manifest(output_dir: Path) -> Dict:
    """Load the download manifest for resumability."""
    manifest_path = output_dir / MANIFEST_FILE
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"urls": {}, "discovered_urls": []}


def save_manifest(manifest: Dict, output_dir: Path):
    """Save the download manifest atomically."""
    manifest_path = output_dir / MANIFEST_FILE
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp_path.replace(manifest_path)


def scrape_lotsawahouse(
    output_dir: str,
    delay: float = DEFAULT_DELAY,
    max_pages: Optional[int] = None,
    retry_errors: bool = False,
):
    """
    Scrape Lotsawa House text pages to a local directory.

    Args:
        output_dir: Directory to save HTML files
        delay: Seconds between requests (default 1.5)
        max_pages: Maximum pages to download (None = all)
        retry_errors: If True, retry previously failed URLs
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(out)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Phase 1: Discover URLs (or use cached discovery)
    if manifest.get("discovered_urls"):
        all_urls = set(manifest["discovered_urls"])
        print(f"  Using cached URL list: {len(all_urls)} text URLs")
    else:
        print("  Discovering text URLs from Lotsawa House...")
        all_urls = set()

        # Words of the Buddha (direct text links)
        print("    Scanning /words-of-the-buddha/...")
        buddha_urls = discover_text_urls_from_index(session, delay)
        all_urls.update(buddha_urls)
        print(f"    Found {len(buddha_urls)} texts")

        # Tibetan masters
        for section in ["/tibetan-masters/", "/indian-masters/"]:
            print(f"    Scanning {section}...")
            master_pages = discover_master_pages(section, session, delay)
            print(f"    Found {len(master_pages)} masters, scanning their pages...")

            for i, master_url in enumerate(master_pages):
                text_urls = discover_text_urls_from_master(master_url, session, delay)
                all_urls.update(text_urls)
                if (i + 1) % 20 == 0:
                    print(f"      Scanned {i + 1}/{len(master_pages)} masters ({len(all_urls)} texts so far)")

            section_name = section.strip("/")
            print(f"    {section_name}: {len(all_urls)} total texts discovered so far")

        manifest["discovered_urls"] = sorted(all_urls)
        save_manifest(manifest, out)
        print(f"  Total discovered: {len(all_urls)} text URLs")

    # Phase 2: Download pages
    already_done = {
        url for url, info in manifest["urls"].items()
        if info.get("status") == "ok"
    }
    if retry_errors:
        to_download = sorted(all_urls - already_done)
    else:
        already_attempted = set(manifest["urls"].keys())
        to_download = sorted(all_urls - already_attempted)

    if max_pages is not None:
        to_download = to_download[:max_pages]

    total = len(to_download)
    if total == 0:
        print(f"  All {len(already_done)} pages already downloaded. Nothing to do.")
        return

    print(f"  Downloading {total} pages ({len(already_done)} already cached)...")

    try:
        for i, url in enumerate(to_download):
            filepath = url_to_filepath(url, out)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            html = fetch_page(url, session, delay)

            if html:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(html)

                manifest["urls"][url] = {
                    "path": str(filepath.relative_to(out)),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "status": "ok",
                }
            else:
                manifest["urls"][url] = {
                    "path": "",
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                }

            if (i + 1) % 25 == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                print(f"  Downloaded {i + 1}/{total} ({pct:.0f}%)")
                save_manifest(manifest, out)

    except KeyboardInterrupt:
        print("\n  Interrupted. Saving progress...")
    finally:
        save_manifest(manifest, out)

    ok = sum(1 for v in manifest["urls"].values() if v["status"] == "ok")
    errors = sum(1 for v in manifest["urls"].values() if v["status"] == "error")
    print(f"\n  Done. {ok} pages saved, {errors} errors.")
    if errors:
        print(f"  Re-run with --retry-errors to retry failed pages.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Lotsawa House text pages to a local directory"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save HTML files (e.g. C:/llama-cpp/lotsawahouse-data)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between requests (default {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to download (for testing)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry previously failed URLs",
    )
    args = parser.parse_args()

    scrape_lotsawahouse(
        output_dir=args.output_dir,
        delay=args.delay,
        max_pages=args.max_pages,
        retry_errors=args.retry_errors,
    )
