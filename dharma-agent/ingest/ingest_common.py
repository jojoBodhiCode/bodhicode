"""
Shared chunking logic and metadata enrichment for Buddhist text ingestion.

Handles:
  - 300-500 token chunks with 50-100 token overlap
  - Metadata enrichment: tradition, text_id, translator, canonical_collection, type
  - Natural boundary detection (verse groups, paragraphs, section divisions)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TextChunk:
    """A single chunk of Buddhist text with metadata."""
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {"text": self.text, "metadata": self.metadata}


# Rough token estimation: ~4 chars per token for English, ~3 for mixed Pali/Sanskrit
CHARS_PER_TOKEN = 4
DEFAULT_CHUNK_SIZE = 400      # tokens
DEFAULT_CHUNK_OVERLAP = 75    # tokens
MIN_CHUNK_SIZE = 50           # tokens — discard chunks smaller than this


def estimate_tokens(text: str) -> int:
    """Rough token count estimation."""
    return len(text) // CHARS_PER_TOKEN


def split_into_paragraphs(text: str) -> List[str]:
    """Split text at paragraph boundaries (double newlines or blank lines)."""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences, respecting common Buddhist abbreviations."""
    # Avoid splitting on common abbreviations like "e.g.", "i.e.", "Skt.", "Pali."
    text = re.sub(r'\b(e\.g|i\.e|viz|Skt|Pali|cf|etc)\.\s', r'\1<PERIOD> ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace('<PERIOD>', '.') for s in sentences]
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    base_metadata: Optional[dict] = None,
) -> List[TextChunk]:
    """
    Split text into chunks of approximately chunk_size tokens with overlap.

    Tries to split at paragraph boundaries first, then sentence boundaries.
    Each chunk inherits base_metadata and gets a chunk_index added.
    """
    if not text or not text.strip():
        return []

    base_metadata = base_metadata or {}
    chunks = []

    # First try paragraph-level splitting
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return []

    current_parts = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If a single paragraph exceeds chunk_size, split it further
        if para_tokens > chunk_size:
            # Flush current buffer
            if current_parts:
                chunk_text_str = "\n\n".join(current_parts)
                if estimate_tokens(chunk_text_str) >= MIN_CHUNK_SIZE:
                    meta = {**base_metadata, "chunk_index": len(chunks)}
                    chunks.append(TextChunk(text=chunk_text_str, metadata=meta))
                current_parts = []
                current_tokens = 0

            # Split the large paragraph by sentences
            sentences = split_into_sentences(para)
            sent_parts = []
            sent_tokens = 0
            for sent in sentences:
                st = estimate_tokens(sent)
                if sent_tokens + st > chunk_size and sent_parts:
                    chunk_text_str = " ".join(sent_parts)
                    if estimate_tokens(chunk_text_str) >= MIN_CHUNK_SIZE:
                        meta = {**base_metadata, "chunk_index": len(chunks)}
                        chunks.append(TextChunk(text=chunk_text_str, metadata=meta))
                    # Overlap: keep last few sentences
                    overlap_tokens = 0
                    overlap_parts = []
                    for s in reversed(sent_parts):
                        overlap_tokens += estimate_tokens(s)
                        if overlap_tokens > chunk_overlap:
                            break
                        overlap_parts.insert(0, s)
                    sent_parts = overlap_parts
                    sent_tokens = sum(estimate_tokens(s) for s in sent_parts)
                sent_parts.append(sent)
                sent_tokens += st

            if sent_parts:
                chunk_text_str = " ".join(sent_parts)
                if estimate_tokens(chunk_text_str) >= MIN_CHUNK_SIZE:
                    meta = {**base_metadata, "chunk_index": len(chunks)}
                    chunks.append(TextChunk(text=chunk_text_str, metadata=meta))
            continue

        # Normal case: accumulate paragraphs
        if current_tokens + para_tokens > chunk_size and current_parts:
            chunk_text_str = "\n\n".join(current_parts)
            if estimate_tokens(chunk_text_str) >= MIN_CHUNK_SIZE:
                meta = {**base_metadata, "chunk_index": len(chunks)}
                chunks.append(TextChunk(text=chunk_text_str, metadata=meta))

            # Overlap: keep last paragraph(s) for context continuity
            overlap_tokens = 0
            overlap_parts = []
            for p in reversed(current_parts):
                overlap_tokens += estimate_tokens(p)
                if overlap_tokens > chunk_overlap:
                    break
                overlap_parts.insert(0, p)
            current_parts = overlap_parts
            current_tokens = sum(estimate_tokens(p) for p in current_parts)

        current_parts.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_parts:
        chunk_text_str = "\n\n".join(current_parts)
        if estimate_tokens(chunk_text_str) >= MIN_CHUNK_SIZE:
            meta = {**base_metadata, "chunk_index": len(chunks)}
            chunks.append(TextChunk(text=chunk_text_str, metadata=meta))

    return chunks


def chunk_verses(
    verses: List[str],
    group_size: int = 4,
    base_metadata: Optional[dict] = None,
) -> List[TextChunk]:
    """
    Chunk verse-structured texts (e.g., Dhammapada, MMK) into groups.

    Groups verses by group_size with 1-verse overlap between groups.
    """
    base_metadata = base_metadata or {}
    chunks = []

    for i in range(0, len(verses), group_size - 1):  # -1 for overlap
        group = verses[i:i + group_size]
        if not group:
            continue
        chunk_text_str = "\n".join(group)
        if estimate_tokens(chunk_text_str) >= MIN_CHUNK_SIZE:
            meta = {
                **base_metadata,
                "chunk_index": len(chunks),
                "verse_range": f"{i+1}-{i+len(group)}",
            }
            chunks.append(TextChunk(text=chunk_text_str, metadata=meta))

    return chunks


def enrich_metadata(
    chunks: List[TextChunk],
    tradition: str,
    text_id: str,
    translator: str = "",
    canonical_collection: str = "",
    text_type: str = "sutta",
    source_url: str = "",
) -> List[TextChunk]:
    """Add standard metadata fields to all chunks."""
    for chunk in chunks:
        chunk.metadata.update({
            "tradition": tradition,
            "text_id": text_id,
            "translator": translator,
            "canonical_collection": canonical_collection,
            "type": text_type,
        })
        if source_url:
            chunk.metadata["source_url"] = source_url
    return chunks
