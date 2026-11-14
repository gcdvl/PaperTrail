"""
dense_retriever.py: Dense embedding-based retrieval for PaperTrail.

Uses sentence-transformers models (default: BAAI/bge-small-en-v1.5) to compute
dense vector representations of text passages and queries, scores candidates via
cosine similarity, and caches passage embeddings on disk to accelerate repeated
eval runs.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.chunker import Chunk
from src.retriever import RetrievalResult

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "embeddings"


def compute_corpus_cache_key(passages: Sequence[Chunk], model_name: str) -> str:
    """
    Computes a deterministic SHA-256 hash representing a specific corpus of passages
    and model configuration.

    Args:
        passages: Sequence of Chunk objects.
        model_name: Name or path of the embedding model.

    Returns:
        Hexadecimal hash string.
    """
    hasher = hashlib.sha256()
    hasher.update(model_name.strip().encode("utf-8"))
    hasher.update(b"\n---CORPUS_CHUNKS_SEPARATOR---\n")
    for chunk in passages:
        hasher.update(chunk.passage_id.encode("utf-8"))
        hasher.update(b":")
        hasher.update(chunk.text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


class DenseRetriever:
    """
    Dense vector retriever using sentence-transformers embeddings and cosine similarity.

    Attributes:
        model_name: Hugging Face model identifier (e.g., 'BAAI/bge-small-en-v1.5').
        cache_dir: Directory to persist numpy-serialized passage embeddings.
        embedder: The underlying sentence-transformers model or stub embedder.
        query_prefix: Optional custom query prefix (auto-detected for BGE models).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        cache_dir: Optional[Union[str, Path]] = None,
        embedder: Optional[Any] = None,
        query_prefix: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.embedder = embedder
        self.query_prefix = query_prefix
        self._memory_cache: Dict[str, np.ndarray] = {}

    def _get_embedder(self) -> Any:
        """Lazily loads the sentence-transformers model if not already provided."""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.model_name)
        return self.embedder

    def _prepare_query_text(self, query: str) -> str:
        """Applies query prefix if specified or if model is a BGE variant."""
        prefix = self.query_prefix
        if prefix is None and "bge" in self.model_name.lower():
            prefix = BGE_QUERY_PREFIX

        if prefix and not query.startswith(prefix):
            return f"{prefix}{query}"
        return query

    def get_passage_embeddings(self, passages: Sequence[Chunk]) -> np.ndarray:
        """
        Retrieves or computes passage embeddings for the given chunk sequence.
        Checks in-memory cache, then on-disk cache, falling back to model inference.

        Args:
            passages: Sequence of Chunk objects.

        Returns:
            2D numpy array of shape (len(passages), embedding_dim).
        """
        if not passages:
            return np.empty((0, 0), dtype=np.float32)

        cache_key = compute_corpus_cache_key(passages, self.model_name)

        # 1. Check in-memory cache
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # 2. Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                embeddings = np.load(cache_file)
                self._memory_cache[cache_key] = embeddings
                return embeddings
            except Exception:
                # If cache corrupted, recalculate below
                pass

        # 3. Compute embeddings with embedder
        embedder = self._get_embedder()
        texts = [chunk.text for chunk in passages]
        raw_embs = embedder.encode(texts)
        embeddings = np.asarray(raw_embs, dtype=np.float32)

        # 4. Save to disk and memory cache
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, embeddings)
        except Exception:
            pass  # Non-fatal if writing disk cache fails
        self._memory_cache[cache_key] = embeddings

        return embeddings

    def retrieve(
        self,
        passages: Sequence[Chunk],
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieves top_k passages ranked by cosine similarity against the query embedding.

        Args:
            passages: Sequence of Chunk objects to search.
            query: User question or search query.
            top_k: Maximum number of ranked results to return.

        Returns:
            List of RetrievalResult objects ranked descending by cosine similarity score.
            Returns [] if query is empty, corpus is empty, or top_k <= 0.
        """
        if not passages or not query or not query.strip() or top_k <= 0:
            return []

        # 1. Get or compute passage embeddings
        passage_embeddings = self.get_passage_embeddings(passages)
        if passage_embeddings.size == 0 or len(passages) == 0:
            return []

        # 2. Embed the query with appropriate prefix
        embedder = self._get_embedder()
        formatted_query = self._prepare_query_text(query.strip())
        raw_query_emb = embedder.encode(formatted_query)
        query_embedding = np.asarray(raw_query_emb, dtype=np.float32).flatten()

        # 3. Compute cosine similarity
        passage_norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_embedding)

        # Avoid zero division
        passage_norms = np.maximum(passage_norms, 1e-10)
        query_norm = max(float(query_norm), 1e-10)

        norm_passages = passage_embeddings / passage_norms
        norm_query = query_embedding / query_norm

        scores = np.dot(norm_passages, norm_query).flatten()

        # 4. Rank descending by cosine similarity
        k = min(top_k, len(passages))
        # Use argsort to get descending ranks
        sorted_indices = np.argsort(-scores)[:k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(sorted_indices, start=1):
            chunk = passages[idx]
            results.append(
                RetrievalResult(
                    passage_id=chunk.passage_id,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    passage=chunk.text,
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        return results


def retrieve_dense(
    passages: Sequence[Chunk],
    query: str,
    top_k: int = 5,
    model_name: str = DEFAULT_DENSE_MODEL,
    cache_dir: Optional[Union[str, Path]] = None,
    embedder: Optional[Any] = None,
    query_prefix: Optional[str] = None,
) -> List[RetrievalResult]:
    """
    Convenience function to perform dense vector retrieval for a query over passages.

    Args:
        passages: Sequence of Chunk objects.
        query: Query string.
        top_k: Cutoff rank.
        model_name: Hugging Face model identifier.
        cache_dir: Disk cache location.
        embedder: Optional pre-instantiated model or stub embedder.
        query_prefix: Optional query prefix.

    Returns:
        List of RetrievalResult objects.
    """
    retriever = DenseRetriever(
        model_name=model_name,
        cache_dir=cache_dir,
        embedder=embedder,
        query_prefix=query_prefix,
    )
    return retriever.retrieve(passages, query, top_k=top_k)
