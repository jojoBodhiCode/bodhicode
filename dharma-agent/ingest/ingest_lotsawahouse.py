"""
Ingest Lotsawa House (lotsawahouse.org) locally cached HTML into the knowledge base.

Lotsawa House provides 6,000+ Tibetan Buddhist text translations under CC BY-NC 4.0.
This ingester reads HTML files saved by scrape_lotsawahouse.py and extracts English
text with metadata.

Expected structure (from scrape_lotsawahouse.py):
  lotsawahouse-data/
    words-of-the-buddha/*.html
    tibetan-masters/*/*.html
    indian-masters/*/*.html
    _manifest.json

Usage:
  python -m ingest.ingest_lotsawahouse /path/to/lotsawahouse-data
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing 'beautifulsoup4'. Install with: pip install beautifulsoup4")
    sys.exit(1)

from .ingest_common import TextChunk, chunk_text, enrich_metadata


# ─── Constants ────────────────────────────────────────────────────────────────

# Tibetan Unicode range for detecting Tibetan script
TIBETAN_RE = re.compile(r'[\u0F00-\u0FFF]')

# Map topic tags to text types
TOPIC_TO_TEXT_TYPE = {
    "sutra": "sutra",
    "sūtra": "sutra",
    "sutras": "sutra",
    "sūtras": "sutra",
    "tantra": "tantra",
    "tantras": "tantra",
    "dharani": "dharani",
    "dhāraṇī": "dharani",
    "prayer": "prayer",
    "prayers": "prayer",
    "aspiration prayers": "prayer",
    "praise": "praise",
    "praises": "praise",
    "song": "song",
    "songs": "song",
    "doha": "song",
    "prophecy": "prophecy",
    "commentary": "commentary",
    "commentaries": "commentary",
    "sadhana": "sadhana",
    "sādhana": "sadhana",
    "sādhanas": "sadhana",
    "advice": "advice",
    "dzogchen": "dzogchen",
    "mahamudra": "practice",
    "guru yoga": "practice",
    "meditation": "practice",
    "confession": "practice",
    "dedication": "prayer",
    "auspiciousness": "prayer",
    "quotations": "quotation",
    "testament": "scripture",
    "empowerment": "practice",
    "offering": "practice",
    "biography": "biography",
    "history": "history",
    "ngöndro": "practice",
    "vinaya": "vinaya",
    "abhidharma": "abhidharma",
}

# Canonical collection based on source section
COLLECTION_MAP = {
    "words-of-the-buddha": "Kangyur",
    "indian-masters": "Indian Treatises",
    "tibetan-masters": "Tibetan Commentarial Literature",
}


# ─── CSS classes used by Lotsawa House ────────────────────────────────────────
# Tibetan script classes (skip these)
TIBETAN_CLASSES = {
    "TibetanVerse", "HeadingTib", "TibetanInlineEnglish",
    "TibetanProse", "TibetanTitle",
}

# English content classes (keep these)
ENGLISH_CLASSES = {
    "EnglishText", "EnglishPhonetics", "Heading3", "Heading2",
    "ExplanationBold", "Explanation", "Colophon",
}


def is_tibetan_element(element) -> bool:
    """Check if a BeautifulSoup element is Tibetan script by CSS class or content."""
    classes = element.get("class", [])
    if any(c in TIBETAN_CLASSES for c in classes):
        return True

    # Check if the element is dominated by TibetanInlineEnglish spans
    tibetan_spans = element.find_all("span", class_="TibetanInlineEnglish")
    if tibetan_spans:
        tibetan_len = sum(len(s.get_text()) for s in tibetan_spans)
        total_len = len(element.get_text().strip())
        if total_len > 0 and tibetan_len / total_len > 0.5:
            return True

    # Fallback: check Unicode content for elements without class
    if not classes:
        text = element.get_text()
        if text and len(text.strip()) > 5:
            tibetan_chars = len(TIBETAN_RE.findall(text))
            total_chars = len(text.strip())
            if total_chars > 0 and tibetan_chars / total_chars > 0.3:
                return True

    return False


# ─── Metadata extraction ─────────────────────────────────────────────────────

def slug_to_title(slug: str) -> str:
    """Convert a URL slug to a readable title. 'jigme-lingpa' -> 'Jigme Lingpa'."""
    return slug.replace("-", " ").title()


def extract_text_from_html(filepath: Path) -> dict:
    """
    Extract English text and metadata from a Lotsawa House HTML file.

    Uses the site's CSS class conventions:
      - div#maintext: main content container
      - div.categories a.tag-circle: topic tags
      - .TibetanVerse, .HeadingTib: Tibetan script (skip)
      - .EnglishText, .EnglishPhonetics, .Heading3: English content (keep)

    Returns dict with title, author, translator, text, topics, etc.
    """
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
    except Exception:
        return {}

    # Title: from <h1> or <title> tag
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            title = re.sub(r'\s*\|\s*Lotsawa House\s*$', '', title)

    # Topics: from div.categories a.tag-circle[href^="/topics/"]
    topics = []
    categories_div = soup.find("div", class_="categories")
    if categories_div:
        for a in categories_div.find_all("a", href=re.compile(r'^/topics/')):
            topic_name = a.get_text(strip=True)
            if topic_name and topic_name not in topics:
                topics.append(topic_name)

    # Translator: look for "Translated by" in page text
    translator = ""
    page_text = soup.get_text()
    trans_match = re.search(
        r'[Tt]ranslated\s+by\s+([A-Z][^,\n\d]{2,80})(?:[,.]|\s+\d{4})',
        page_text,
    )
    if trans_match:
        translator = trans_match.group(1).strip()
        translator = re.sub(r'\s+(?:and|with|under|for)$', '', translator, flags=re.IGNORECASE)

    if not translator:
        trans_link = soup.find("a", href=re.compile(r'/translators/'))
        if trans_link:
            translator = trans_link.get_text(strip=True)

    # Author: "by Author Name" pattern near the beginning of the content
    author = ""
    content_div = soup.find(id="content")
    if content_div:
        content_text = content_div.get_text()[:1000]
        by_match = re.search(r'\bby\s+([A-Z][A-Za-z\u00C0-\u024F\s\-\']+?)(?:\s*\(|\s*$|\s*\n)', content_text)
        if by_match:
            candidate = by_match.group(1).strip()
            if len(candidate) < 80:
                author = candidate

    # Body text: extract from div#maintext, filtering by CSS class
    maintext = soup.find(id="maintext")
    if not maintext:
        # Fallback: try div#content
        maintext = soup.find(id="content")
    if not maintext:
        return {}

    text_parts = []
    for element in maintext.find_all(["p", "blockquote", "h2", "h3", "h4"]):
        # Skip Tibetan script elements
        if is_tibetan_element(element):
            continue

        # Skip language list, gaps
        elem_id = element.get("id", "")
        if elem_id == "lang-list":
            continue
        classes = element.get("class", [])
        if "gap" in classes:
            continue

        text = element.get_text(separator=" ", strip=True)
        if not text or len(text) < 10:
            continue

        text_parts.append(text)

    full_text = "\n\n".join(text_parts)

    # Clean whitespace
    full_text = re.sub(r' +', ' ', full_text)
    full_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_text)

    # Derive path info
    source_section = ""
    master_slug = ""
    text_slug = filepath.stem

    for section in ["words-of-the-buddha", "tibetan-masters", "indian-masters"]:
        if section in str(filepath):
            source_section = section
            break

    if source_section in ("tibetan-masters", "indian-masters"):
        master_slug = filepath.parent.name
        if not author:
            author = slug_to_title(master_slug)

    return {
        "title": title,
        "author": author,
        "translator": translator,
        "text": full_text,
        "topics": topics,
        "source_section": source_section,
        "master_slug": master_slug,
        "text_slug": text_slug,
    }


def classify_text_type(topics: List[str]) -> str:
    """Determine text type from topic tags."""
    for topic in topics:
        key = topic.lower().strip()
        if key in TOPIC_TO_TEXT_TYPE:
            return TOPIC_TO_TEXT_TYPE[key]
    return "text"


def generate_text_id(result: dict) -> str:
    """Generate a unique text ID from the URL structure."""
    section = result.get("source_section", "")
    master = result.get("master_slug", "")
    slug = result.get("text_slug", "unknown")

    if master:
        return f"LH:{section}/{master}/{slug}"
    elif section:
        return f"LH:{section}/{slug}"
    else:
        return f"LH:{slug}"


# ─── Main ingestion ──────────────────────────────────────────────────────────

def ingest_lotsawahouse(source_path: str) -> List[TextChunk]:
    """
    Ingest Lotsawa House locally cached HTML files.

    Args:
        source_path: Path to the lotsawahouse-data directory
                     (created by scrape_lotsawahouse.py)

    Returns:
        List of TextChunk objects ready for embedding
    """
    base = Path(source_path)

    if not base.exists():
        print(f"  Error: Path not found: {source_path}")
        print(f"  Run scraper first:")
        print(f"    python -m ingest.scrape_lotsawahouse {source_path}")
        return []

    # Discover HTML files
    html_files = []
    for subdir in ["words-of-the-buddha", "tibetan-masters", "indian-masters"]:
        search_dir = base / subdir
        if search_dir.exists():
            html_files.extend(search_dir.rglob("*.html"))

    if not html_files:
        print(f"  No HTML files found in {source_path}")
        print(f"  Expected subdirectories: words-of-the-buddha/, tibetan-masters/, indian-masters/")
        return []

    print(f"  Found {len(html_files)} HTML files in Lotsawa House cache")

    all_chunks = []
    file_count = 0
    skip_count = 0

    for html_file in sorted(html_files):
        try:
            result = extract_text_from_html(html_file)
            if not result or not result.get("text") or len(result["text"]) < 200:
                skip_count += 1
                continue

            text_id = generate_text_id(result)
            text_type = classify_text_type(result.get("topics", []))

            chunks = chunk_text(
                result["text"],
                base_metadata={
                    "source_file": str(html_file.relative_to(base)),
                    "title": result.get("title", ""),
                    "author": result.get("author", ""),
                    "topics": ", ".join(result.get("topics", [])),
                },
            )

            # Reconstruct source URL from file path
            rel_path = html_file.relative_to(base)
            url_path = str(rel_path).replace("\\", "/").replace(".html", "")
            source_url = f"https://www.lotsawahouse.org/{url_path}"

            # Canonical collection from section
            source_section = result.get("source_section", "")
            canonical_collection = COLLECTION_MAP.get(
                source_section, "Lotsawa House"
            )

            chunks = enrich_metadata(
                chunks,
                tradition="Vajrayana",
                text_id=text_id,
                translator=result.get("translator", ""),
                canonical_collection=canonical_collection,
                text_type=text_type,
                source_url=source_url,
            )

            all_chunks.extend(chunks)
            file_count += 1

        except Exception as e:
            print(f"  Warning: Error parsing {html_file.name}: {e}")
            continue

    print(f"  Lotsawa House: Ingested {file_count} texts -> {len(all_chunks)} chunks")
    if skip_count:
        print(f"  (Skipped {skip_count} files with insufficient content)")
    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.ingest_lotsawahouse /path/to/lotsawahouse-data")
        print("\nDownload first with:")
        print("  python -m ingest.scrape_lotsawahouse /path/to/lotsawahouse-data")
        sys.exit(1)
    chunks = ingest_lotsawahouse(sys.argv[1])
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        sample = chunks[0].text[:200].encode("ascii", errors="replace").decode()
        print(f"Sample: {sample}...")
        print(f"Metadata: {chunks[0].metadata}")
