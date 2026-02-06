"""
Ingest 84000.co Tibetan Buddhist canon translations into the knowledge base.

84000.co provides English translations of the Tibetan Kangyur and Tengyur
under CC BY-NC license, including commentaries by Nagarjuna, Dignaga,
Dharmakirti, and Vasubandhu.

Texts can be downloaded from: https://84000.co/all-translations
Each text is typically available as HTML or XML.

Expected structure:
  84000_texts/
    *.html (or *.xml)

Usage:
  python -m ingest.ingest_84000 /path/to/84000_texts
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing 'beautifulsoup4'. Install with: pip install beautifulsoup4")
    sys.exit(1)

from .ingest_common import TextChunk, chunk_text, enrich_metadata


# Known 84000 collections
COLLECTION_MAP = {
    "kangyur": "Kangyur",
    "tengyur": "Tengyur",
    "sutra": "Kangyur - Sutra",
    "tantra": "Kangyur - Tantra",
    "vinaya": "Kangyur - Vinaya",
    "abhidharma": "Tengyur - Abhidharma",
    "madhyamaka": "Tengyur - Madhyamaka",
    "yogacara": "Tengyur - Yogacara",
    "pramana": "Tengyur - Pramana",
}

# Known authors in the Tibetan canon
KNOWN_AUTHORS = {
    "nagarjuna": "Nagarjuna",
    "aryadeva": "Aryadeva",
    "candrakirti": "Candrakirti",
    "chandrakirti": "Candrakirti",
    "asanga": "Asanga",
    "vasubandhu": "Vasubandhu",
    "dignaga": "Dignaga",
    "dharmakirti": "Dharmakirti",
    "santideva": "Santideva",
    "shantideva": "Santideva",
    "maitreya": "Maitreya/Asanga",
    "tsongkhapa": "Tsongkhapa",
    "longchenpa": "Longchenpa",
    "mipham": "Mipham",
}


def extract_84000_text(filepath: Path) -> dict:
    """
    Extract text and metadata from an 84000.co download (HTML or XML).

    Returns dict with 'title', 'author', 'text', 'toh_number', 'collection'.
    """
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return {}

    # Detect format
    is_xml = filepath.suffix.lower() == ".xml" or content.strip().startswith("<?xml")

    if is_xml:
        soup = BeautifulSoup(content, "html.parser")  # lxml would be better but html.parser works
    else:
        soup = BeautifulSoup(content, "html.parser")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Try 84000-specific markup
    if not title:
        for selector in [".title", "h1", ".text-title"]:
            tag = soup.select_one(selector)
            if tag:
                title = tag.get_text(strip=True)
                break

    # Extract Tohoku number (84000's cataloging system)
    toh_number = ""
    toh_match = re.search(r'Toh\.?\s*(\d+)', content, re.IGNORECASE)
    if toh_match:
        toh_number = f"Toh {toh_match.group(1)}"

    # Detect author/attributed author
    author = ""
    fname_lower = filepath.stem.lower()
    for key, name in KNOWN_AUTHORS.items():
        if key in fname_lower or key in content.lower()[:2000]:
            author = name
            break

    # Detect collection/tradition
    collection = ""
    for key, coll_name in COLLECTION_MAP.items():
        if key in fname_lower or key in content.lower()[:2000]:
            collection = coll_name
            break

    # Determine tradition (most 84000 texts are Mahayana or Vajrayana)
    tradition = "Mahayana"
    if any(kw in fname_lower or kw in content.lower()[:3000]
           for kw in ["tantra", "vajra", "mandala", "mantra", "sadhana"]):
        tradition = "Vajrayana"

    # Remove script/style/nav
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Extract main text
    main = soup.find(id="content") or soup.find(class_="translation") or soup.find("body")
    if not main:
        return {}

    text_parts = []
    for element in main.find_all(["p", "blockquote", "div", "h2", "h3", "h4", "verse"]):
        text = element.get_text(separator=" ", strip=True)
        if text and len(text) > 30:
            text_parts.append(text)

    full_text = "\n\n".join(text_parts)

    # Clean
    full_text = re.sub(r'\s+', ' ', full_text)
    full_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_text)

    return {
        "title": title,
        "author": author,
        "text": full_text,
        "toh_number": toh_number,
        "collection": collection or "Kangyur",
        "tradition": tradition,
    }


def ingest_84000(texts_path: str) -> List[TextChunk]:
    """
    Ingest 84000.co text downloads.

    Args:
        texts_path: Path to directory containing 84000 downloads (HTML/XML)

    Returns:
        List of TextChunk objects ready for embedding
    """
    base = Path(texts_path)
    if not base.exists():
        print(f"  ❌ 84000 texts directory not found: {texts_path}")
        print(f"  💡 Download texts from: https://84000.co/all-translations")
        return []

    all_chunks = []
    file_count = 0

    # Find all HTML and XML files
    files = list(base.rglob("*.html")) + list(base.rglob("*.xml")) + list(base.rglob("*.htm"))
    print(f"  📖 Found {len(files)} files in 84000 collection")

    for filepath in sorted(files):
        try:
            result = extract_84000_text(filepath)
            if not result or not result.get("text") or len(result["text"]) < 200:
                continue

            text_id = result.get("toh_number") or result.get("title", filepath.stem)
            tradition = result.get("tradition", "Mahayana")

            # Determine text type
            text_type = "sutra"
            fname = filepath.stem.lower()
            if any(kw in fname for kw in ["commentary", "vrtti", "bhasya", "tika"]):
                text_type = "commentary"
            elif any(kw in fname for kw in ["tantra", "sadhana"]):
                text_type = "tantra"
            elif any(kw in fname for kw in ["vinaya"]):
                text_type = "vinaya"

            chunks = chunk_text(
                result["text"],
                base_metadata={
                    "source_file": str(filepath.relative_to(base)),
                    "title": result.get("title", ""),
                    "author": result.get("author", ""),
                },
            )

            chunks = enrich_metadata(
                chunks,
                tradition=tradition,
                text_id=text_id,
                translator="84000 Translation Group",
                canonical_collection=result.get("collection", "Kangyur"),
                text_type=text_type,
                source_url="https://84000.co",
            )

            all_chunks.extend(chunks)
            file_count += 1

        except Exception as e:
            print(f"  ⚠️  Error parsing {filepath.name}: {e}")
            continue

    print(f"  ✅ 84000.co: Ingested {file_count} texts → {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.ingest_84000 /path/to/84000_texts")
        sys.exit(1)
    chunks = ingest_84000(sys.argv[1])
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"Sample: {chunks[0].text[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
