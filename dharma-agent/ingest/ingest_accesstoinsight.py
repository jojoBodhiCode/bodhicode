"""
Ingest Access to Insight (accesstoinsight.org) bulk download into the knowledge base.

Access to Insight provides 1,000+ curated sutta translations available as a complete
offline bulk download in HTML format (frozen at 2013 but still excellent).
Translators include Thanissaro Bhikkhu and Bhikkhu Bodhi.

Download: https://www.accesstoinsight.org/lib/downloads/ati_website.zip

Expected structure after extraction:
  ati_website/
    html/
      tipitaka/
        dn/ mn/ sn/ an/ kn/
          .../*.html

Usage:
  python -m ingest.ingest_accesstoinsight /path/to/ati_website
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


# Map ATI directory structure to canonical info
ATI_NIKAYA_INFO = {
    "dn": {"name": "Digha Nikaya", "collection": "Sutta Pitaka", "abbr": "DN"},
    "mn": {"name": "Majjhima Nikaya", "collection": "Sutta Pitaka", "abbr": "MN"},
    "sn": {"name": "Samyutta Nikaya", "collection": "Sutta Pitaka", "abbr": "SN"},
    "an": {"name": "Anguttara Nikaya", "collection": "Sutta Pitaka", "abbr": "AN"},
    "kn": {"name": "Khuddaka Nikaya", "collection": "Khuddaka Nikaya", "abbr": "KN"},
}

# Known ATI translators
ATI_TRANSLATORS = {
    "than": "Thanissaro Bhikkhu",
    "thanissaro": "Thanissaro Bhikkhu",
    "bodhi": "Bhikkhu Bodhi",
    "soma": "Soma Thera",
    "nyanasatta": "Nyanasatta Thera",
    "nyanaponika": "Nyanaponika Thera",
    "nanamoli": "Bhikkhu Nanamoli",
    "ireland": "John D. Ireland",
    "piyadassi": "Piyadassi Thera",
    "hare": "E.M. Hare",
    "woodward": "F.L. Woodward",
}


def extract_text_from_html(filepath: Path) -> dict:
    """
    Extract sutta text and metadata from an ATI HTML file.

    Returns dict with 'title', 'translator', 'text', and 'sutta_id'.
    """
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
    except Exception:
        return {}

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else filepath.stem

    # Try to find the translator from meta tags or page content
    translator = ""
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author:
        translator = meta_author.get("content", "")

    if not translator:
        # Look for translator name in the byline
        byline = soup.find(class_="byline") or soup.find(class_="author")
        if byline:
            translator = byline.get_text(strip=True)

    if not translator:
        # Try to detect from filename patterns
        fname = filepath.stem.lower()
        for key, name in ATI_TRANSLATORS.items():
            if key in fname:
                translator = name
                break

    # Extract the main text content
    # ATI pages typically have the sutta text in the main body
    # Remove navigation, headers, footers
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Look for the main content div
    content = soup.find(id="content") or soup.find(class_="main") or soup.find("body")
    if not content:
        return {}

    # Get text, preserving paragraph structure
    text_parts = []
    for element in content.find_all(["p", "blockquote", "div", "h1", "h2", "h3", "h4"]):
        text = element.get_text(separator=" ", strip=True)
        if text and len(text) > 20:  # Skip very short fragments
            text_parts.append(text)

    full_text = "\n\n".join(text_parts)

    # Clean up common artifacts
    full_text = re.sub(r'\s+', ' ', full_text)
    full_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_text)

    # Try to extract sutta ID from title or filename
    sutta_id = ""
    # Pattern: "DN 1", "MN 26", "SN 56.11", etc.
    id_match = re.search(r'(DN|MN|SN|AN|Dhp|Ud|Iti|Snp|Thag|Thig)\s*(\d+(?:\.\d+)*)', title)
    if id_match:
        sutta_id = f"{id_match.group(1)} {id_match.group(2)}"

    return {
        "title": title,
        "translator": translator,
        "text": full_text,
        "sutta_id": sutta_id,
    }


def ingest_access_to_insight(ati_path: str) -> List[TextChunk]:
    """
    Ingest the Access to Insight offline website.

    Args:
        ati_path: Path to the extracted ATI website root

    Returns:
        List of TextChunk objects ready for embedding
    """
    base = Path(ati_path)

    # Try common paths
    tipitaka_dir = None
    for candidate in [
        base / "html" / "tipitaka",
        base / "tipitaka",
        base / "lib" / "authors",
        base,
    ]:
        if candidate.exists():
            tipitaka_dir = candidate
            break

    if not tipitaka_dir:
        print(f"  ❌ ATI tipitaka directory not found under: {ati_path}")
        print(f"  💡 Download from: https://www.accesstoinsight.org/lib/downloads/ati_website.zip")
        return []

    all_chunks = []
    file_count = 0

    html_files = list(tipitaka_dir.rglob("*.html"))
    print(f"  📖 Found {len(html_files)} HTML files in ATI")

    for html_file in sorted(html_files):
        try:
            result = extract_text_from_html(html_file)
            if not result or not result.get("text") or len(result["text"]) < 200:
                continue

            # Determine nikaya from path
            rel_path = html_file.relative_to(tipitaka_dir)
            parts = rel_path.parts
            nikaya_key = parts[0] if parts else ""
            nikaya_info = ATI_NIKAYA_INFO.get(nikaya_key, {
                "name": nikaya_key.upper() if nikaya_key else "Unknown",
                "collection": "Sutta Pitaka",
                "abbr": nikaya_key.upper() if nikaya_key else "",
            })

            text_id = result.get("sutta_id") or result.get("title", html_file.stem)

            chunks = chunk_text(
                result["text"],
                base_metadata={
                    "source_file": str(rel_path),
                    "title": result.get("title", ""),
                },
            )

            chunks = enrich_metadata(
                chunks,
                tradition="Theravada",
                text_id=text_id,
                translator=result.get("translator", ""),
                canonical_collection=nikaya_info.get("collection", "Sutta Pitaka"),
                text_type="sutta",
                source_url=f"https://www.accesstoinsight.org/tipitaka/{rel_path}",
            )

            all_chunks.extend(chunks)
            file_count += 1

        except Exception as e:
            print(f"  ⚠️  Error parsing {html_file.name}: {e}")
            continue

    print(f"  ✅ Access to Insight: Ingested {file_count} texts → {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.ingest_accesstoinsight /path/to/ati_website")
        sys.exit(1)
    chunks = ingest_access_to_insight(sys.argv[1])
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"Sample: {chunks[0].text[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
