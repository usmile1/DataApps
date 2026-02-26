"""Embedding client using Ollama's nomic-embed-text model.

Turns text into a 768-dimensional vector for similarity search.
"""

import requests
from typing import List

OLLAMA_EMBED_API = "http://localhost:11434/api/embeddings"
DEFAULT_MODEL = "nomic-embed-text"


def embed(text: str, model: str = DEFAULT_MODEL) -> List[float]:
    """Embed a single text string into a vector.

    Args:
        text: The text to embed.
        model: Ollama embedding model name.

    Returns:
        A list of floats (768 dimensions for nomic-embed-text).
    """
    resp = requests.post(OLLAMA_EMBED_API, json={
        "model": model,
        "prompt": text,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_batch(texts: List[str], model: str = DEFAULT_MODEL) -> List[List[float]]:
    """Embed multiple texts. Calls Ollama sequentially (no batch API)."""
    return [embed(t, model) for t in texts]
