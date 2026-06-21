# =============================================================================
# src/embeddings.py
# -----------------------------------------------------------------------------
# Generates vector embeddings using SentenceTransformers (local, offline).
#
# Model:   sentence-transformers/all-MiniLM-L6-v2
# Output:  384-dimensional float32 vectors, L2-normalised
# Offline: No API key or network calls required after first model download.
#          The model is cached locally by HuggingFace Hub on first use.
#
# Public API (identical signatures to the Gemini version — no other file
# needs to change):
#   initialise_gemini()   → bool        # name kept for compatibility
#   embed_chunks()        → np.ndarray | None
#   embed_query()         → np.ndarray | None
#   validate_embeddings() → bool
#
# FAISS compatibility:
#   All arrays are float32 and L2-normalised.
#   L2-normalised vectors + IndexFlatIP = cosine similarity (same as before).
#   Only the dimension changes: 768 (Gemini) → 384 (MiniLM).
#   vector_store.py imports EMBEDDING_DIMENSION from here, so it auto-adapts.
#
# First run:
#   The model (~90 MB) is downloaded from HuggingFace and cached at:
#   ~/.cache/huggingface/hub/  (Linux/macOS)
#   C:\Users\<user>\.cache\huggingface\hub\  (Windows)
#   Subsequent runs load from the local cache instantly.
# =============================================================================

import logging
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

# HuggingFace model identifier. Downloaded once, cached locally forever.
# all-MiniLM-L6-v2: fast, lightweight, excellent retrieval quality.
MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Output vector dimension for this model (fixed by the model architecture).
# vector_store.py imports this constant — changing models here is enough.
EMBEDDING_DIMENSION: int = 384

# Chunks to encode per SentenceTransformer batch call.
# MiniLM is fast; 64 per batch balances memory vs throughput.
EMBEDDING_BATCH_SIZE: int = 64


# ---------------------------------------------------------------------------
# Module-level model singleton
# ---------------------------------------------------------------------------
# Loaded once on first use; reused for all subsequent calls in the session.
# Loading a SentenceTransformer takes ~1–2 s; we never want to do it twice.
_model: Optional[SentenceTransformer] = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialise_gemini(api_key: Optional[str] = None) -> bool:
    """
    Load the local SentenceTransformer model into memory.

    The function name is kept as initialise_gemini() so that app.py and
    the processing pipeline require zero changes. The api_key parameter
    is accepted but ignored — no API key is needed for local embeddings.

    On the very first call the model weights (~90 MB) are downloaded from
    HuggingFace Hub and cached locally. Every subsequent call loads from
    the cache in ~1 second.

    Args:
        api_key: Ignored. Kept for interface compatibility only.

    Returns:
        True  if the model loaded successfully.
        False if loading failed (e.g. no internet on first run).
    """
    global _model

    # Already loaded — nothing to do.
    if _model is not None:
        logger.debug("SentenceTransformer model already loaded — reusing.")
        return True

    logger.info(f"Loading embedding model '{MODEL_NAME}'...")
    logger.info(
        "First run: model will be downloaded (~90 MB) and cached locally. "
        "Subsequent runs load from cache instantly."
    )

    try:
        _model = SentenceTransformer(MODEL_NAME)
        logger.info(
            f"Model loaded successfully. "
            f"Output dimension: {EMBEDDING_DIMENSION}."
        )
        return True

    except Exception as e:
        logger.error(
            f"Failed to load SentenceTransformer model '{MODEL_NAME}': {e}\n"
            "If this is your first run, ensure you have an internet connection "
            "so the model can be downloaded and cached."
        )
        _model = None
        return False


def _get_model() -> Optional[SentenceTransformer]:
    """
    Return the loaded model, auto-initialising if needed.

    Returns:
        The SentenceTransformer instance, or None if loading failed.
    """
    global _model
    if _model is None:
        initialise_gemini()
    return _model


# ---------------------------------------------------------------------------
# Chunk Embedding  (document indexing)
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    Embed a list of text chunks for indexing in FAISS.

    Encodes all chunk texts using the local MiniLM model.
    Vectors are L2-normalised so that inner-product search in FAISS
    is equivalent to cosine similarity.

    Args:
        chunks: List of chunk dicts from text_chunker.chunk_pages().
                Each must have a "text" key.

    Returns:
        np.ndarray of shape (n_chunks, 384), dtype float32.
        Returns None if encoding fails or model is unavailable.
    """
    if not chunks:
        logger.warning("embed_chunks() received an empty list.")
        return None

    model = _get_model()
    if model is None:
        logger.error("embed_chunks(): model is not loaded.")
        return None

    texts = [chunk["text"] for chunk in chunks]
    logger.info(f"Encoding {len(texts)} chunks with '{MODEL_NAME}'...")

    try:
        # encode() returns a float32 numpy array of shape (n, 384).
        # normalize_embeddings=True applies L2 normalisation in-place,
        # making inner product == cosine similarity (required for FAISS IndexFlatIP).
        # show_progress_bar=True prints a tqdm bar for large document sets.
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            precision="float32",
            normalize_embeddings=True,
            show_progress_bar=len(texts) > EMBEDDING_BATCH_SIZE,
        )

        logger.info(f"Encoding complete: shape={embeddings.shape}.")
        return embeddings

    except Exception as e:
        logger.error(f"embed_chunks() failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Query Embedding  (inference time)
# ---------------------------------------------------------------------------

def embed_query(query_text: str) -> Optional[np.ndarray]:
    """
    Embed a single user query for similarity search against the FAISS index.

    Uses the same model and normalisation as embed_chunks() so that
    cosine similarity scores between query and chunk vectors are meaningful.

    Note: MiniLM uses symmetric similarity — no separate task_type is
    needed for queries vs documents (unlike Gemini's asymmetric retrieval).

    Args:
        query_text: The user's question string.

    Returns:
        np.ndarray of shape (384,), dtype float32.
        Returns None if encoding fails.
    """
    if not query_text or not query_text.strip():
        logger.warning("embed_query() received an empty string.")
        return None

    model = _get_model()
    if model is None:
        logger.error("embed_query(): model is not loaded.")
        return None

    logger.debug(f"Encoding query ({len(query_text)} chars)...")

    try:
        # encode() with a single string returns shape (384,) — a 1-D array.
        # vector_store.search() reshapes it to (1, 384) before FAISS search.
        vector: np.ndarray = model.encode(
            query_text.strip(),
            convert_to_numpy=True,
            precision="float32",
            normalize_embeddings=True,
        )

        return vector

    except Exception as e:
        logger.error(f"embed_query() failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_embeddings(embeddings: np.ndarray, expected_count: int) -> bool:
    """
    Sanity-check a completed embedding array before passing it to FAISS.

    Verifies shape, dtype, row count, vector dimension, and absence of
    NaN / Inf values that would silently corrupt FAISS similarity scores.

    Args:
        embeddings:     Array returned by embed_chunks().
        expected_count: Number of chunks that were embedded.

    Returns:
        True if valid and safe to pass to VectorStore.build_index().
        False otherwise (error details logged).
    """
    if embeddings is None:
        logger.error("Validation failed: embeddings is None.")
        return False

    if embeddings.ndim != 2:
        logger.error(
            f"Validation failed: expected 2-D array, got {embeddings.ndim}-D."
        )
        return False

    if embeddings.shape[0] != expected_count:
        logger.error(
            f"Validation failed: expected {expected_count} rows, "
            f"got {embeddings.shape[0]}."
        )
        return False

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        logger.error(
            f"Validation failed: expected dim={EMBEDDING_DIMENSION}, "
            f"got {embeddings.shape[1]}."
        )
        return False

    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        logger.error("Validation failed: array contains NaN or Inf values.")
        return False

    logger.debug(
        f"Embeddings valid: shape={embeddings.shape}, dtype={embeddings.dtype}."
    )
    return True
