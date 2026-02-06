"""
Ingest SuttaCentral bilara-data into the knowledge base.

SuttaCentral's bilara-data repo (github.com/suttacentral/bilara-data) contains
thousands of suttas in structured JSON format, segmented by sentence with unique IDs.

Expected repo structure:
  bilara-data/
    translation/en/sujato/sutta/
      dn/ mn/ sn/ an/ kp/ dhp/ ud/ iti/ snp/ vv/ pv/ thag/ thig/

Each JSON file maps segment IDs to translated text:
  {"mn1:1.1": "So I have heard.", "mn1:1.2": "At one time...", ...}

Usage:
  python -m ingest.ingest_suttacentral /path/to/bilara-data
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from .ingest_common import TextChunk, chunk_text, enrich_metadata


# Map nikaya abbreviations to full names and canonical collections
NIKAYA_INFO = {
    "dn": {"name": "Digha Nikaya", "collection": "Sutta Pitaka", "abbr": "DN"},
    "mn": {"name": "Majjhima Nikaya", "collection": "Sutta Pitaka", "abbr": "MN"},
    "sn": {"name": "Samyutta Nikaya", "collection": "Sutta Pitaka", "abbr": "SN"},
    "an": {"name": "Anguttara Nikaya", "collection": "Sutta Pitaka", "abbr": "AN"},
    "kp": {"name": "Khuddakapatha", "collection": "Khuddaka Nikaya", "abbr": "Kp"},
    "dhp": {"name": "Dhammapada", "collection": "Khuddaka Nikaya", "abbr": "Dhp"},
    "ud": {"name": "Udana", "collection": "Khuddaka Nikaya", "abbr": "Ud"},
    "iti": {"name": "Itivuttaka", "collection": "Khuddaka Nikaya", "abbr": "Iti"},
    "snp": {"name": "Sutta Nipata", "collection": "Khuddaka Nikaya", "abbr": "Snp"},
    "vv": {"name": "Vimanavatthu", "collection": "Khuddaka Nikaya", "abbr": "Vv"},
    "pv": {"name": "Petavatthu", "collection": "Khuddaka Nikaya", "abbr": "Pv"},
    "thag": {"name": "Theragatha", "collection": "Khuddaka Nikaya", "abbr": "Thag"},
    "thig": {"name": "Therigatha", "collection": "Khuddaka Nikaya", "abbr": "Thig"},
}


def parse_sutta_id(filename: str) -> dict:
    """Extract sutta ID info from filename like 'mn1_translation-en-sujato.json'."""
    stem = Path(filename).stem
    # Pattern: {nikaya}{number}[.{sub}]_translation-en-{translator}
    match = re.match(r'^([a-z]+)(\d+(?:\.\d+)*)(?:[-_](\d+(?:\.\d+)*))?', stem)
    if not match:
        return {"nikaya": "", "number": "", "text_id": stem}

    nikaya = match.group(1)
    number = match.group(2)
    sub = match.group(3) or ""

    info = NIKAYA_INFO.get(nikaya, {"name": nikaya.upper(), "collection": "Sutta Pitaka", "abbr": nikaya.upper()})
    text_id = f"{info['abbr']} {number}"
    if sub:
        text_id += f".{sub}"

    return {
        "nikaya": nikaya,
        "number": number,
        "text_id": text_id,
        "nikaya_name": info["name"],
        "canonical_collection": info["collection"],
    }


def load_sutta_json(filepath: Path) -> str:
    """Load a bilara-data JSON file and reconstruct the full text."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    # Segments are ordered by their IDs (e.g., mn1:1.1, mn1:1.2, ...)
    # Sort by the numeric parts of the ID to ensure correct order
    def sort_key(item):
        key = item[0]
        parts = re.findall(r'\d+', key)
        return [int(p) for p in parts]

    sorted_segments = sorted(data.items(), key=sort_key)

    # Join segments into continuous text, using paragraph breaks at section boundaries
    paragraphs = []
    current_section = None
    current_parts = []

    for seg_id, text in sorted_segments:
        if not text or not text.strip():
            continue

        # Detect section breaks from segment ID structure (e.g., mn1:1.x vs mn1:2.x)
        section_match = re.match(r'[^:]+:(\d+)', seg_id)
        section = section_match.group(1) if section_match else None

        if section != current_section and current_parts:
            paragraphs.append(" ".join(current_parts))
            current_parts = []

        current_section = section
        current_parts.append(text.strip())

    if current_parts:
        paragraphs.append(" ".join(current_parts))

    return "\n\n".join(paragraphs)


def ingest_bilara_data(
    bilara_path: str,
    translator: str = "sujato",
    language: str = "en",
) -> List[TextChunk]:
    """
    Ingest SuttaCentral bilara-data repository.

    Args:
        bilara_path: Path to the bilara-data repo root
        translator: Translator folder name (default: sujato for Bhikkhu Sujato)
        language: Language code (default: en)

    Returns:
        List of TextChunk objects ready for embedding
    """
    base = Path(bilara_path) / "translation" / language / translator / "sutta"

    if not base.exists():
        print(f"  ❌ bilara-data path not found: {base}")
        print(f"  💡 Clone it with: git clone https://github.com/suttacentral/bilara-data.git")
        return []

    all_chunks = []
    file_count = 0

    for nikaya_dir in sorted(base.iterdir()):
        if not nikaya_dir.is_dir():
            continue

        nikaya = nikaya_dir.name
        json_files = list(nikaya_dir.rglob("*.json"))

        for json_file in sorted(json_files):
            try:
                text = load_sutta_json(json_file)
                if not text or len(text.strip()) < 100:
                    continue

                sutta_info = parse_sutta_id(json_file.name)
                text_id = sutta_info.get("text_id", json_file.stem)

                chunks = chunk_text(
                    text,
                    base_metadata={"source_file": str(json_file.relative_to(bilara_path))},
                )

                translator_name = "Bhikkhu Sujato" if translator == "sujato" else translator
                chunks = enrich_metadata(
                    chunks,
                    tradition="Theravada",
                    text_id=text_id,
                    translator=translator_name,
                    canonical_collection=sutta_info.get("canonical_collection", "Sutta Pitaka"),
                    text_type="sutta",
                    source_url=f"https://suttacentral.net/{nikaya}{sutta_info.get('number', '')}/en/{translator}",
                )

                all_chunks.extend(chunks)
                file_count += 1

            except Exception as e:
                print(f"  ⚠️  Error parsing {json_file.name}: {e}")
                continue

    print(f"  ✅ SuttaCentral: Ingested {file_count} suttas → {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m ingest.ingest_suttacentral /path/to/bilara-data")
        sys.exit(1)
    chunks = ingest_bilara_data(sys.argv[1])
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print(f"Sample: {chunks[0].text[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")
