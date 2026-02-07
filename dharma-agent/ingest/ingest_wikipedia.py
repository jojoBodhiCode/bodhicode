"""
Ingest Wikipedia Buddhism articles (JSON) into the knowledge base.

Reads JSON article files saved by scrape_wikipedia_buddhism.py, classifies
articles by Buddhist tradition and text type using their Wikipedia categories,
then chunks and enriches metadata using ingest_common.

Expected structure (from scrape_wikipedia_buddhism.py):
  wikipedia-buddhism-data/
    articles/*.json
    _manifest.json

Usage:
  python -m ingest.ingest_wikipedia /path/to/wikipedia-buddhism-data
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ingest_common import TextChunk, chunk_text, enrich_metadata


# ─── Tradition Detection ─────────────────────────────────────────────────────

TRADITION_KEYWORDS = {
    "Theravada": [
        "theravada", "pali canon", "pali", "vipassana", "sutta pitaka",
        "abhidhamma", "thai buddhism", "burmese buddhism", "sri lankan buddhism",
        "theravadin", "early buddhism",
    ],
    "Mahayana": [
        "mahayana", "madhyamaka", "yogacara", "pure land", "chan buddhism",
        "zen", "huayan", "tiantai", "seon", "chinese buddhism",
        "japanese buddhism", "korean buddhism", "nichiren",
        "lotus sutra", "heart sutra", "prajnaparamita",
    ],
    "Vajrayana": [
        "vajrayana", "tibetan buddhism", "tantra", "dzogchen", "mahamudra",
        "kagyu", "nyingma", "gelug", "sakya", "bon", "shingon",
        "tibetan buddhist", "lama", "tulku", "rinpoche", "dalai lama",
        "karmapa", "terma",
    ],
}

# ─── Text Type Classification ────────────────────────────────────────────────

TEXT_TYPE_KEYWORDS = {
    "sutra":      ["sutra", "sutta", "sutras", "suttas", "pali canon texts"],
    "commentary": ["commentary", "commentaries", "exegesis",
                   "abhidharma", "abhidhamma"],
    "history":    ["history of buddhism", "buddhist history", "historical"],
    "biography":  ["buddhist monks", "buddhist nuns", "buddhist leaders",
                   "lamas", "rinpoches", "bhikkhus", "zen masters",
                   "buddhist writers", "indian buddhists"],
    "philosophy": ["buddhist philosophy", "buddhist concepts", "buddhist terms",
                   "buddhist logic", "epistemology"],
    "practice":   ["buddhist meditation", "buddhist practices", "buddhist rituals",
                   "buddhist devotion", "mindfulness"],
    "school":     ["schools of buddhism", "buddhist denominations", "buddhist sects"],
    "temple":     ["buddhist temples", "buddhist monasteries", "stupas", "pagodas"],
    "art":        ["buddhist art", "buddhist architecture", "buddhist symbols",
                   "buddha images", "buddhist iconography"],
}

# Sections to strip from article text (reference/navigation noise)
STRIP_SECTIONS = [
    "See also", "References", "External links", "Further reading",
    "Notes", "Citations", "Bibliography", "Sources",
]


# ─── Classification ──────────────────────────────────────────────────────────

def detect_tradition(categories: List[str], discovery_categories: List[str]) -> str:
    """
    Detect the Buddhist tradition from article and discovery categories.

    Returns "Theravada", "Mahayana", "Vajrayana", or "General Buddhism".
    """
    all_cats = " ".join(categories + discovery_categories).lower()

    scores = {}
    for tradition, keywords in TRADITION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in all_cats)
        if score > 0:
            scores[tradition] = score

    if not scores:
        return "General Buddhism"

    return max(scores, key=scores.get)


def detect_text_type(title: str, categories: List[str]) -> str:
    """
    Classify article text type from its title and categories.

    Returns a type string compatible with the KB metadata schema.
    """
    all_text = (title + " " + " ".join(categories)).lower()

    for text_type in ["sutra", "commentary", "biography", "history",
                      "practice", "philosophy", "school", "temple", "art"]:
        keywords = TEXT_TYPE_KEYWORDS[text_type]
        if any(kw in all_text for kw in keywords):
            return text_type

    return "encyclopedia"


def generate_text_id(title: str) -> str:
    """Generate a text ID: WP:Article_Title."""
    safe_title = title.replace(" ", "_")
    return f"WP:{safe_title}"


# ─── Text cleaning ───────────────────────────────────────────────────────────

def clean_article_text(text: str) -> str:
    """
    Clean Wikipedia article plain text for chunking.

    Strips reference sections, citation brackets, and section header markup.
    """
    # Remove trailing sections (See also, References, etc.)
    for section in STRIP_SECTIONS:
        pattern = rf'\n\s*={2,}\s*{re.escape(section)}\s*={2,}\s*\n'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]

    # Remove citation brackets: [1], [citation needed], [note 1]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[note \d+\]', '', text, flags=re.IGNORECASE)

    # Clean section headers: "== Heading ==" -> "Heading"
    text = re.sub(r'={2,}\s*(.+?)\s*={2,}', r'\1', text)

    # Normalize whitespace
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    return text.strip()


# ─── Main ingestion ──────────────────────────────────────────────────────────

def ingest_wikipedia(source_path: str) -> List[TextChunk]:
    """
    Ingest Wikipedia Buddhism articles from locally cached JSON files.

    Args:
        source_path: Path to the wikipedia-buddhism-data directory
                     (created by scrape_wikipedia_buddhism.py)

    Returns:
        List of TextChunk objects ready for embedding
    """
    base = Path(source_path)

    if not base.exists():
        print(f"  Error: Path not found: {source_path}")
        print(f"  Run scraper first:")
        print(f"    python -m ingest.scrape_wikipedia_buddhism {source_path}")
        return []

    articles_dir = base / "articles"
    if not articles_dir.exists():
        print(f"  No articles directory found in {source_path}")
        return []

    # Load manifest for discovery categories
    manifest_path = base / "_manifest.json"
    discovery_categories: Dict[str, List[str]] = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        discovery_categories = manifest.get("article_categories", {})

    # Find all JSON article files
    json_files = sorted(articles_dir.glob("*.json"))
    if not json_files:
        print(f"  No JSON article files found in {articles_dir}")
        return []

    print(f"  Found {len(json_files)} Wikipedia article files")

    all_chunks: List[TextChunk] = []
    file_count = 0
    skip_count = 0

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                article = json.load(f)

            title = article.get("title", "")
            raw_text = article.get("text", "")

            if not raw_text or len(raw_text) < 200:
                skip_count += 1
                continue

            text = clean_article_text(raw_text)
            if len(text) < 200:
                skip_count += 1
                continue

            # Classify
            article_cats = article.get("categories", [])
            disc_cats = discovery_categories.get(title, [])
            tradition = detect_tradition(article_cats, disc_cats)
            text_type = detect_text_type(title, article_cats)
            text_id = generate_text_id(title)
            source_url = article.get(
                "url",
                f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            )

            # Chunk
            chunks = chunk_text(
                text,
                base_metadata={
                    "title": title,
                    "source_file": json_file.name,
                    "wp_categories": ", ".join(article_cats[:10]),
                    "pageid": str(article.get("pageid", "")),
                },
            )

            # Enrich
            chunks = enrich_metadata(
                chunks,
                tradition=tradition,
                text_id=text_id,
                translator="",
                canonical_collection="Wikipedia",
                text_type=text_type,
                source_url=source_url,
            )

            all_chunks.extend(chunks)
            file_count += 1

        except Exception as e:
            print(f"  Warning: Error processing {json_file.name}: {e}")
            continue

    print(f"  Wikipedia: Ingested {file_count} articles -> {len(all_chunks)} chunks")
    if skip_count:
        print(f"  (Skipped {skip_count} articles with insufficient content)")
    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.ingest_wikipedia /path/to/wikipedia-buddhism-data")
        print("\nDownload first with:")
        print("  python -m ingest.scrape_wikipedia_buddhism /path/to/wikipedia-buddhism-data")
        sys.exit(1)
    chunks = ingest_wikipedia(sys.argv[1])
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        sample = chunks[0].text[:200].encode("ascii", errors="replace").decode()
        print(f"Sample: {sample}...")
        print(f"Metadata: {chunks[0].metadata}")
