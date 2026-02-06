"""
RAG (Retrieval-Augmented Generation) module for the Dharma Scholar agent.

Architecture:
  - Embedding model: nomic-ai/nomic-embed-text-v1.5 (CUDA if available, 8192 token context)
  - Vector store: ChromaDB (local persistent)
  - Integration: feeds retrieved context into llama-server via OpenAI-compatible API

Usage:
    from rag import DharmaRAG
    rag = DharmaRAG()
    rag.index_chunks(chunks)
    context, sources = rag.retrieve("What is emptiness?", k=5)
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Missing 'chromadb'. Install with: pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Missing 'sentence-transformers'. Install with: pip install sentence-transformers")
    sys.exit(1)

from ingest.ingest_common import TextChunk


# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_buddhist_db")
DEFAULT_COLLECTION = "buddhist_texts"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
if torch.cuda.is_available():
    _DEVICE = "cuda:0"
    BATCH_SIZE = 64  # conservative for 2GB VRAM GPUs
else:
    _DEVICE = "cpu"
    BATCH_SIZE = 100


class DharmaRAG:
    """RAG pipeline for Buddhist scholarly texts."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        self._embedder = None
        self._embedding_model_name = embedding_model

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

    @property
    def embedder(self):
        """Lazy-load the embedding model (large download on first use)."""
        if self._embedder is None:
            print(f"  Loading embedding model: {self._embedding_model_name}")
            print(f"     (first run downloads ~274MB, subsequent runs use cache)")
            self._embedder = SentenceTransformer(
                self._embedding_model_name,
                trust_remote_code=True,
            )
            self._embedder = self._embedder.to(_DEVICE)
            print(f"  Embedding model loaded on {_DEVICE.upper()}.")
        return self._embedder

    def _embed_texts(self, texts: List[str], prefix: str = "search_document: ") -> List[List[float]]:
        """
        Embed texts using nomic-embed-text.

        nomic-embed-text supports task-type prefixes:
          - "search_document: " for documents being indexed
          - "search_query: " for queries at retrieval time
        """
        prefixed = [f"{prefix}{t}" for t in texts]
        embeddings = self.embedder.encode(prefixed, show_progress_bar=len(texts) > 50)
        return embeddings.tolist()

    def index_chunks(self, chunks: List[TextChunk], show_progress: bool = True) -> int:
        """
        Index a list of TextChunk objects into ChromaDB.

        Args:
            chunks: List of TextChunk objects with text and metadata
            show_progress: Whether to print progress

        Returns:
            Number of chunks successfully indexed
        """
        if not chunks:
            return 0

        total = len(chunks)
        indexed = 0

        for batch_start in range(0, total, BATCH_SIZE):
            batch = chunks[batch_start:batch_start + BATCH_SIZE]

            texts = [c.text for c in batch]
            metadatas = []
            ids = []

            for i, chunk in enumerate(batch):
                # ChromaDB metadata must be str, int, float, or bool
                meta = {}
                for k, v in chunk.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                metadatas.append(meta)

                # Generate unique ID from metadata
                text_id = chunk.metadata.get("text_id", "unknown")
                chunk_idx = chunk.metadata.get("chunk_index", batch_start + i)
                ids.append(f"{text_id}__chunk_{chunk_idx}")

            try:
                embeddings = self._embed_texts(texts, prefix="search_document: ")

                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                indexed += len(batch)

                if show_progress:
                    print(f"  Indexed {indexed}/{total} chunks", end="\r")

            except Exception as e:
                print(f"\n  Warning: Error indexing batch at {batch_start}: {e}")
                continue

        if show_progress:
            print(f"  Indexed {indexed}/{total} chunks into ChromaDB")

        return indexed

    def retrieve(
        self,
        query: str,
        k: int = 5,
        tradition_filter: Optional[str] = None,
        text_type_filter: Optional[str] = None,
        collection_filter: Optional[str] = None,
    ) -> Tuple[str, List[dict]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query
            k: Number of chunks to retrieve
            tradition_filter: Optional filter by tradition (Theravada/Mahayana/Vajrayana)
            text_type_filter: Optional filter by type (sutta/verse/commentary/essay)
            collection_filter: Optional filter by canonical collection

        Returns:
            Tuple of (formatted_context, list_of_source_metadata)
        """
        if self.collection.count() == 0:
            return "", []

        # Build where clause for metadata filtering
        where = None
        filters = {}
        if tradition_filter:
            filters["tradition"] = tradition_filter
        if text_type_filter:
            filters["type"] = text_type_filter
        if collection_filter:
            filters["canonical_collection"] = collection_filter

        if len(filters) == 1:
            key, val = list(filters.items())[0]
            where = {key: val}
        elif len(filters) > 1:
            where = {"$and": [{k: v} for k, v in filters.items()]}

        try:
            query_embedding = self._embed_texts([query], prefix="search_query: ")

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(k, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"  Warning: RAG retrieval error: {e}")
            return "", []

        if not results or not results["documents"] or not results["documents"][0]:
            return "", []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else [0] * len(documents)

        # Format context for injection into the prompt
        context = self.format_context(documents, metadatas, distances)
        sources = metadatas

        return context, sources

    def format_context(
        self,
        documents: List[str],
        metadatas: List[dict],
        distances: List[float],
    ) -> str:
        """
        Format retrieved chunks with source citations for injection into the prompt.

        Produces a structured context block that the LLM can reference.
        """
        parts = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            # Build a citation header
            text_id = meta.get("text_id", "Unknown")
            tradition = meta.get("tradition", "")
            translator = meta.get("translator", "")
            collection = meta.get("canonical_collection", "")
            similarity = max(0, 1 - dist)  # Convert cosine distance to similarity

            citation = f"[Source {i+1}: {text_id}"
            if tradition:
                citation += f" ({tradition})"
            if translator:
                citation += f", tr. {translator}"
            if collection:
                citation += f", {collection}"
            citation += f", relevance: {similarity:.0%}]"

            parts.append(f"{citation}\n{doc}")

        return "\n\n---\n\n".join(parts)

    def get_stats(self) -> dict:
        """Get corpus statistics."""
        count = self.collection.count()

        stats = {
            "total_chunks": count,
            "db_path": self.db_path,
            "embedding_model": self._embedding_model_name,
        }

        if count > 0:
            # Sample some metadata to get tradition/collection breakdowns
            try:
                sample = self.collection.get(
                    limit=min(count, 1000),
                    include=["metadatas"],
                )
                traditions = {}
                collections = {}
                text_types = {}

                for meta in sample["metadatas"]:
                    t = meta.get("tradition", "Unknown")
                    traditions[t] = traditions.get(t, 0) + 1

                    c = meta.get("canonical_collection", "Unknown")
                    collections[c] = collections.get(c, 0) + 1

                    tt = meta.get("type", "Unknown")
                    text_types[tt] = text_types.get(tt, 0) + 1

                # If we sampled less than total, scale up
                if len(sample["metadatas"]) < count:
                    scale = count / len(sample["metadatas"])
                    traditions = {k: int(v * scale) for k, v in traditions.items()}
                    collections = {k: int(v * scale) for k, v in collections.items()}
                    text_types = {k: int(v * scale) for k, v in text_types.items()}

                stats["by_tradition"] = traditions
                stats["by_collection"] = collections
                stats["by_type"] = text_types

            except Exception:
                pass

        return stats

    def search_direct(self, query: str, k: int = 5) -> List[dict]:
        """
        Direct search for interactive knowledge base exploration.

        Returns list of dicts with 'text', 'metadata', and 'similarity'.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self._embed_texts([query], prefix="search_query: ")

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "metadata": meta,
                "similarity": max(0, 1 - dist),
            })

        return output

    def clear(self):
        """Clear the entire knowledge base."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print("  Knowledge base cleared.")
