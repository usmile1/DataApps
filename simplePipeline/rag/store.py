"""DIY vector store backed by SQLite + numpy.

Stores embedding vectors as BLOBs alongside entity metadata.
Retrieves top-K results by cosine similarity with a threshold cutoff.

At our scale (~200 vectors), brute-force cosine similarity is <1ms.
In production you'd reach for FAISS or Pinecone, but this lets us
see exactly what's happening.
"""

import sqlite3
import struct
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RetrievedExample:
    """A past classification retrieved by similarity search."""
    entity_type: str
    matched_text: str
    confidence: float
    classified_by_layer: str
    pattern_name: str
    document_context: str  # file path / type info
    similarity: float      # cosine similarity score


# Similarity thresholds
STRONG_THRESHOLD = 0.75
WEAK_THRESHOLD = 0.50


class VectorStore:
    """SQLite-backed vector store for RAG retrieval."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        # Cache all vectors in memory for fast search
        self._vectors: Optional[np.ndarray] = None
        self._metadata: Optional[List[dict]] = None

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS rag_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                matched_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                classified_by_layer TEXT NOT NULL,
                pattern_name TEXT NOT NULL,
                document_context TEXT NOT NULL,
                embed_text TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
        """)
        self._conn.commit()

    def add(self, entity_type: str, matched_text: str, confidence: float,
            classified_by_layer: str, pattern_name: str,
            document_context: str, embed_text: str,
            embedding: List[float]) -> None:
        """Store an embedded entity classification."""
        blob = _floats_to_blob(embedding)
        self._conn.execute(
            "INSERT INTO rag_vectors "
            "(entity_type, matched_text, confidence, classified_by_layer, "
            "pattern_name, document_context, embed_text, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entity_type, matched_text, confidence, classified_by_layer,
             pattern_name, document_context, embed_text, blob)
        )
        self._conn.commit()
        # Invalidate cache
        self._vectors = None
        self._metadata = None

    def search(self, query_embedding: List[float], top_k: int = 3,
               threshold: float = WEAK_THRESHOLD) -> List[RetrievedExample]:
        """Find the most similar past classifications.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results.
            threshold: Minimum cosine similarity to include.

        Returns:
            List of RetrievedExample, sorted by similarity (highest first).
            Only includes results above the threshold.
        """
        self._load_cache()

        if self._vectors is None or len(self._vectors) == 0:
            return []

        # Cosine similarity: dot product of normalized vectors
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        similarities = self._vectors @ query_norm

        # Filter by threshold and sort
        results = []
        indices = np.argsort(similarities)[::-1]  # highest first

        for idx in indices[:top_k * 2]:  # check extra in case some are below threshold
            sim = float(similarities[idx])
            if sim < threshold:
                break  # sorted, so all remaining are lower
            meta = self._metadata[idx]
            results.append(RetrievedExample(
                entity_type=meta["entity_type"],
                matched_text=meta["matched_text"],
                confidence=meta["confidence"],
                classified_by_layer=meta["classified_by_layer"],
                pattern_name=meta["pattern_name"],
                document_context=meta["document_context"],
                similarity=sim,
            ))
            if len(results) >= top_k:
                break

        return results

    def count(self) -> int:
        """Number of vectors in the store."""
        row = self._conn.execute("SELECT COUNT(*) FROM rag_vectors").fetchone()
        return row[0]

    def _load_cache(self) -> None:
        """Load all vectors into memory for fast similarity search."""
        if self._vectors is not None:
            return

        rows = self._conn.execute(
            "SELECT entity_type, matched_text, confidence, classified_by_layer, "
            "pattern_name, document_context, embedding FROM rag_vectors"
        ).fetchall()

        if not rows:
            self._vectors = np.array([], dtype=np.float32)
            self._metadata = []
            return

        vectors = []
        metadata = []
        for row in rows:
            vec = _blob_to_floats(row["embedding"])
            # Normalize for cosine similarity
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
            metadata.append({
                "entity_type": row["entity_type"],
                "matched_text": row["matched_text"],
                "confidence": row["confidence"],
                "classified_by_layer": row["classified_by_layer"],
                "pattern_name": row["pattern_name"],
                "document_context": row["document_context"],
            })

        self._vectors = np.array(vectors, dtype=np.float32)
        self._metadata = metadata

    def close(self) -> None:
        self._conn.close()


def _floats_to_blob(floats: List[float]) -> bytes:
    """Pack a list of floats into a binary blob."""
    return struct.pack(f"{len(floats)}f", *floats)


def _blob_to_floats(blob: bytes) -> np.ndarray:
    """Unpack a binary blob into a numpy array of floats."""
    count = len(blob) // 4  # 4 bytes per float32
    return np.array(struct.unpack(f"{count}f", blob), dtype=np.float32)
