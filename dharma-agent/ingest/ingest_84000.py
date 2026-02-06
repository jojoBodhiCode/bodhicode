"""
Ingest 84000.co Tibetan Buddhist canon translations (TEI/XML) into the knowledge base.

84000.co provides English translations of the Tibetan Kangyur and Tengyur
under CC BY-NC-ND license. Their data-tei GitHub repo contains all published
translations in TEI/XML format with rich metadata.

Expected structure (from git clone https://github.com/84000/data-tei.git):
  84000-data-tei/
    translations/
      kangyur/
        translations/*.xml   (actual translations)
        placeholders/*.xml   (not yet translated -- skipped)
      tengyur/
        translations/*.xml
        placeholders/*.xml

Usage:
  python -m ingest.ingest_84000 /path/to/84000-data-tei
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional
from xml.etree import ElementTree as ET

from .ingest_common import TextChunk, chunk_text, enrich_metadata


# TEI namespace
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def extract_tei_text(filepath: Path) -> dict:
    """
    Extract text and metadata from an 84000 TEI/XML file.

    Returns dict with 'title', 'translator', 'text', 'toh_number',
    'collection', 'tradition'.
    """
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  Warning: XML parse error in {filepath.name}: {e}")
        return {}

    # ─── Extract metadata from teiHeader ──────────────────────────────────

    # Title (English main title)
    title = ""
    for title_el in root.findall(".//tei:titleStmt/tei:title", NS):
        if title_el.get("{http://www.w3.org/XML/1998/namespace}lang") == "en" and \
           title_el.get("type") == "mainTitle":
            title = (title_el.text or "").strip()
            break
    # Fallback to any English title
    if not title:
        for title_el in root.findall(".//tei:titleStmt/tei:title", NS):
            if title_el.get("{http://www.w3.org/XML/1998/namespace}lang") == "en":
                title = (title_el.text or "").strip()
                break

    # Tohoku number from sourceDesc
    toh_number = ""
    bibl = root.find(".//tei:sourceDesc/tei:bibl", NS)
    if bibl is not None:
        ref = bibl.find("tei:ref", NS)
        if ref is not None and ref.text:
            toh_number = ref.text.strip()
        # Also try the key attribute
        if not toh_number and bibl.get("key"):
            toh_number = bibl.get("key", "")

    # Translator
    translator = ""
    for author_el in root.findall(".//tei:titleStmt/tei:author", NS):
        role = author_el.get("role", "")
        if role in ("translatorEng", "translatorMain"):
            # Get text content, handling mixed content with <lb/> etc
            translator = "".join(author_el.itertext()).strip()
            # Clean up "Translated by X and team under..."
            if "under the patronage" in translator:
                translator = translator.split("under the patronage")[0].strip()
            translator = translator.replace("Translated by ", "").strip()
            break

    # Determine collection from file path
    collection = "Kangyur"
    path_str = str(filepath).lower()
    if "tengyur" in path_str:
        collection = "Tengyur"

    # Determine tradition
    tradition = "Mahayana"
    fname = filepath.stem.lower()
    content_preview = title.lower() + " " + fname
    if any(kw in content_preview for kw in ["tantra", "vajra", "mandala", "mantra", "sadhana"]):
        tradition = "Vajrayana"

    # ─── Extract translation text body ────────────────────────────────────

    # Find the translation div
    translation_div = root.find(".//tei:body//tei:div[@type='translation']", NS)
    if translation_div is None:
        # Try without specific type
        translation_div = root.find(".//tei:body", NS)
    if translation_div is None:
        return {}

    # Extract all text from paragraphs and verses
    text_parts = []

    for elem in translation_div.iter():
        tag = elem.tag.replace(f"{{{NS['tei']}}}", "")

        if tag == "p":
            text = "".join(elem.itertext()).strip()
            if text and len(text) > 20:
                # Clean whitespace
                text = re.sub(r'\s+', ' ', text)
                text_parts.append(text)

        elif tag == "l":
            # Verse line
            text = "".join(elem.itertext()).strip()
            if text and not text.startswith(("F.", "B")):  # skip folio refs
                text_parts.append(text)

        elif tag == "head":
            text = "".join(elem.itertext()).strip()
            if text:
                text_parts.append(f"\n{text}\n")

    full_text = "\n".join(text_parts)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()

    if len(full_text) < 100:
        return {}

    return {
        "title": title,
        "translator": translator,
        "text": full_text,
        "toh_number": toh_number,
        "collection": collection,
        "tradition": tradition,
    }


def ingest_84000(repo_path: str) -> List[TextChunk]:
    """
    Ingest 84000 TEI/XML translations from the data-tei repo.

    Args:
        repo_path: Path to the cloned 84000/data-tei repository

    Returns:
        List of TextChunk objects ready for embedding
    """
    base = Path(repo_path)

    # Check for the expected repo structure
    translations_dir = base / "translations"
    if not translations_dir.exists():
        # Maybe they pointed directly at the translations folder
        if (base / "kangyur").exists() or (base / "tengyur").exists():
            translations_dir = base
        else:
            print(f"  Error: Could not find translations in {repo_path}")
            print(f"  Expected: {repo_path}/translations/kangyur/translations/*.xml")
            print(f"  Clone with: git clone https://github.com/84000/data-tei.git")
            return []

    # Find actual translation XML files (skip placeholders)
    xml_files = []
    for subdir in ["kangyur/translations", "tengyur/translations",
                    "kangyur", "tengyur"]:
        search_dir = translations_dir / subdir
        if search_dir.exists():
            xml_files.extend(search_dir.glob("*.xml"))

    # Deduplicate
    xml_files = list(set(xml_files))
    # Filter out placeholders
    xml_files = [f for f in xml_files if "placeholder" not in str(f).lower()]

    print(f"  Found {len(xml_files)} translation XML files")

    if not xml_files:
        print(f"  No XML files found. Check the path structure.")
        return []

    all_chunks = []
    file_count = 0

    for filepath in sorted(xml_files):
        try:
            result = extract_tei_text(filepath)
            if not result or not result.get("text"):
                continue

            text_id = result.get("toh_number") or result.get("title", filepath.stem)
            tradition = result.get("tradition", "Mahayana")

            # Determine text type from filename/content
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
                    "source_file": filepath.name,
                    "title": result.get("title", ""),
                },
            )

            translator = result.get("translator", "84000 Translation Group")
            chunks = enrich_metadata(
                chunks,
                tradition=tradition,
                text_id=text_id,
                translator=translator,
                canonical_collection=result.get("collection", "Kangyur"),
                text_type=text_type,
                source_url="https://84000.co",
            )

            all_chunks.extend(chunks)
            file_count += 1

        except Exception as e:
            print(f"  Warning: Error parsing {filepath.name}: {e}")
            continue

    print(f"  84000.co: Ingested {file_count} texts -> {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.ingest_84000 /path/to/84000-data-tei")
        sys.exit(1)
    chunks = ingest_84000(sys.argv[1])
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"Sample: {chunks[0].text[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
