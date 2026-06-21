# =============================================================================
# src/vector_store.py
# -----------------------------------------------------------------------------
# Builds and queries the FAISS vector index for semantic chunk retrieval.
#
# Responsibilities:
#   - Accept embedding vectors + chunk metadata and build a FAISS index
#   - Store chunk metadata alongside the index for result enrichment
#   - Given a query vector, return the top-K most similar chunks
#   - Provide index management utilities (stats, reset, serialisation)
#
# Why FAISS?
#   - Built by Facebook AI Research — used in production at massive scale
#   - Pure in-memory operation: no database server, no disk I/O required
#   - IndexFlatIP (Inner Product) with normalised vectors = cosine similarity
#   - Millisecond search even with thousands of chunks
#   - Perfect for Streamlit Cloud: stateless, no persistence needed
#
# FAISS Index Type — IndexFlatIP:
#   "Flat"  → Exhaustive search (checks every vector, no approximation)
#   "IP"    → Inner Product similarity metric
#   When vectors are L2-normalised, Inner Product == Cosine Similarity.
#   For our chunk counts (typically < 10,000), exhaustive search is fast
#   enough and gives perfect (non-approximate) results.
# =============================================================================

import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss

from config.settings import TOP_K_RESULTS
from src.embeddings import EMBEDDING_DIMENSION

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VectorStore Class
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Manages a FAISS index and its associated chunk metadata.

    This class encapsulates all FAISS operations so the rest of the
    application never needs to interact with FAISS directly. It stores:
      - A FAISS IndexFlatIP index containing normalised embedding vectors
      - A parallel list of chunk metadata dicts (text, source, page, etc.)

    The index and metadata list are always kept in sync:
        index.ntotal == len(self.chunks)
        index.reconstruct(i) corresponds to self.chunks[i]

    Usage:
        store = VectorStore()
        store.build_index(embeddings_array, chunks_list)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self):
        """
        Initialise an empty VectorStore.

        The index is None until build_index() is called.
        """
        # The FAISS index — None until build_index() is called.
        self._index: Optional[faiss.IndexFlatIP] = None

        # Parallel list of chunk metadata dicts.
        # chunks[i] corresponds to the vector at index position i.
        self._chunks: List[Dict[str, Any]] = []

        # Track whether the index has been built and is ready for search.
        self._is_ready: bool = False

        logger.debug("VectorStore instance created (empty).")

    # -----------------------------------------------------------------------
    # Index Building
    # -----------------------------------------------------------------------

    def build_index(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
    ) -> bool:
        """
        Build a FAISS index from a numpy array of embedding vectors.

        This method:
        1. Validates input shapes and counts
        2. L2-normalises all vectors (required for cosine similarity via IP)
        3. Creates a FAISS IndexFlatIP index
        4. Adds all normalised vectors to the index
        5. Stores the parallel chunk metadata list

        Args:
            embeddings: 2D numpy float32 array of shape (n_chunks, dimension).
                        Produced by embeddings.embed_chunks().
            chunks:     List of chunk metadata dicts from text_chunker.
                        Must have the same length as embeddings.shape[0].

        Returns:
            True if the index was built successfully, False otherwise.
        """
        # --- Input Validation ---
        if embeddings is None or len(embeddings) == 0:
            logger.error("build_index() received empty embeddings array.")
            return False

        if not chunks:
            logger.error("build_index() received empty chunks list.")
            return False

        if embeddings.shape[0] != len(chunks):
            logger.error(
                f"Mismatch: {embeddings.shape[0]} embeddings but "
                f"{len(chunks)} chunks. They must be equal."
            )
            return False

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            logger.error(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
                f"got {embeddings.shape[1]}."
            )
            return False

        try:
            n_vectors = embeddings.shape[0]
            logger.info(
                f"Building FAISS index with {n_vectors} vectors "
                f"(dim={EMBEDDING_DIMENSION})..."
            )

            # --- L2 Normalisation ---
            # Normalise each vector to unit length (magnitude = 1).
            # After normalisation: dot_product(a, b) == cosine_similarity(a, b)
            # This lets us use IndexFlatIP (fast inner product) to get
            # cosine similarity without needing IndexFlatL2.
            normalised = embeddings.copy()
            faiss.normalize_L2(normalised)

            # --- Create FAISS Index ---
            # IndexFlatIP: Flat (exhaustive) search using Inner Product.
            # EMBEDDING_DIMENSION tells FAISS the vector size.
            index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)

            # --- Add Vectors ---
            # add() inserts all vectors into the index.
            # Vectors are assigned integer IDs 0, 1, 2, ... automatically.
            index.add(normalised)

            # --- Store Index and Metadata ---
            self._index  = index
            self._chunks = list(chunks)  # Defensive copy
            self._is_ready = True

            logger.info(
                f"FAISS index built successfully: "
                f"{self._index.ntotal} vectors indexed."
            )
            return True

        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            self._is_ready = False
            return False

    # -----------------------------------------------------------------------
    # Similarity Search
    # -----------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS index for the most similar chunks to a query vector.

        The query vector is normalised before search (same as index vectors)
        to ensure cosine similarity scores are computed correctly.

        Args:
            query_vector: 1D numpy float32 array of shape (EMBEDDING_DIMENSION,).
                          Produced by embeddings.embed_query().
            top_k:        Number of results to return.
                          Defaults to TOP_K_RESULTS from config/settings.py.

        Returns:
            List of chunk dicts enriched with a "score" key (cosine similarity,
            0.0–1.0). Sorted by descending similarity (best match first).
            Returns empty list if search fails or index is not ready.

        Example:
            query_vec = embed_query("What was Q3 revenue?")
            results   = store.search(query_vec, top_k=5)
            for r in results:
                print(r["score"], r["text"][:100])
        """
        if not self._is_ready or self._index is None:
            logger.error(
                "search() called but index is not ready. "
                "Call build_index() first."
            )
            return []

        if query_vector is None:
            logger.error("search() received a None query vector.")
            return []

        k = top_k or TOP_K_RESULTS

        # Clamp k to the number of indexed vectors to avoid FAISS errors.
        k = min(k, self._index.ntotal)

        try:
            # --- Prepare Query Vector ---
            # FAISS search expects a 2D array of shape (n_queries, dimension).
            # We're searching with one query at a time, so reshape to (1, dim).
            query = query_vector.reshape(1, -1).astype(np.float32)

            # Normalise the query vector to unit length.
            # CRITICAL: must use the same normalisation as index vectors.
            faiss.normalize_L2(query)

            # --- FAISS Search ---
            # index.search() returns two arrays:
            #   scores  : shape (1, k) — cosine similarity scores (0.0 to 1.0)
            #   indices : shape (1, k) — integer IDs of matching vectors
            scores, indices = self._index.search(query, k)

            # --- Enrich Results with Metadata ---
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # FAISS returns -1 for indices when fewer than k results exist.
                if idx == -1:
                    continue

                # Retrieve the chunk metadata for this index position.
                chunk = self._chunks[idx].copy()  # Copy to avoid mutation

                # Add the similarity score to the chunk dict.
                # Score is a float32 — cast to Python float for JSON safety.
                chunk["score"] = float(score)

                results.append(chunk)

            logger.debug(
                f"FAISS search returned {len(results)} result(s) "
                f"(top score: {results[0]['score']:.4f})."
                if results else "FAISS search returned 0 results."
            )

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    # -----------------------------------------------------------------------
    # Index Management
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear the index and all stored chunk metadata.

        Call this when the user uploads new documents and the existing
        index needs to be rebuilt from scratch.
        """
        self._index    = None
        self._chunks   = []
        self._is_ready = False
        logger.info("VectorStore reset — index and chunks cleared.")

    def is_ready(self) -> bool:
        """
        Return True if the index has been built and is ready for search.
        """
        return self._is_ready

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a summary of the current index state.

        Used by the Streamlit UI to display an info panel showing how
        many chunks were indexed from how many documents.

        Returns:
            Dict with keys: is_ready, total_vectors, unique_sources,
            chunks_per_source, dimension.
        """
        if not self._is_ready:
            return {
                "is_ready":          False,
                "total_vectors":     0,
                "unique_sources":    [],
                "chunks_per_source": {},
                "dimension":         EMBEDDING_DIMENSION,
            }

        # Count chunks per source file.
        chunks_per_source: Dict[str, int] = {}
        for chunk in self._chunks:
            src = chunk.get("source", "unknown")
            chunks_per_source[src] = chunks_per_source.get(src, 0) + 1

        return {
            "is_ready":          True,
            "total_vectors":     self._index.ntotal,
            "unique_sources":    list(chunks_per_source.keys()),
            "chunks_per_source": chunks_per_source,
            "dimension":         EMBEDDING_DIMENSION,
        }

    def get_chunk_count(self) -> int:
        """Return the total number of indexed chunks."""
        return self._index.ntotal if self._is_ready else 0

    # -----------------------------------------------------------------------
    # Serialisation (Optional — for future persistence)
    # -----------------------------------------------------------------------

    def to_bytes(self) -> Optional[bytes]:
        """
        Serialise the FAISS index to bytes for optional persistence.

        Not used in the main pipeline (everything is in-memory per session)
        but provided for future use cases like saving/loading indexes to disk
        or cloud storage between sessions.

        Returns:
            Raw bytes of the serialised FAISS index, or None on failure.
        """
        if not self._is_ready:
            logger.warning("to_bytes() called on an uninitialised index.")
            return None
        try:
            return faiss.serialize_index(self._index).tobytes()
        except Exception as e:
            logger.error(f"Failed to serialise FAISS index: {e}")
            return None

    def from_bytes(
        self,
        index_bytes: bytes,
        chunks: List[Dict[str, Any]],
    ) -> bool:
        """
        Restore a FAISS index from previously serialised bytes.

        Args:
            index_bytes: Bytes produced by to_bytes().
            chunks:      The chunk metadata list that corresponds to the index.

        Returns:
            True if restoration succeeded, False otherwise.
        """
        try:
            index_array  = np.frombuffer(index_bytes, dtype=np.uint8)
            self._index  = faiss.deserialize_index(index_array)
            self._chunks = list(chunks)
            self._is_ready = True
            logger.info(
                f"FAISS index restored from bytes: "
                f"{self._index.ntotal} vectors."
            )
            return True
        except Exception as e:
            logger.error(f"Failed to restore FAISS index from bytes: {e}")
            return False

    # -----------------------------------------------------------------------
    # Dunder Methods
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._is_ready:
            return (
                f"VectorStore(vectors={self._index.ntotal}, "
                f"dim={EMBEDDING_DIMENSION}, ready=True)"
            )
        return "VectorStore(ready=False)"

    def __len__(self) -> int:
        """Return the number of indexed vectors."""
        return self._index.ntotal if self._is_ready else 0