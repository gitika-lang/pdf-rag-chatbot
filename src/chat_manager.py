# =============================================================================
# src/chat_manager.py
# -----------------------------------------------------------------------------
# Manages all Streamlit session state for the PDF RAG Chatbot.
#
# Responsibilities:
#   - Initialise and structure st.session_state on first load
#   - Add, retrieve, and clear chat messages
#   - Track uploaded files and their processing status
#   - Store and expose the VectorStore and LLMHandler instances
#   - Provide a clean API so app.py never touches session_state directly
#
# Why centralise session state?
#   Streamlit reruns the entire script on every user interaction.
#   st.session_state is the only mechanism for persisting data between reruns.
#   Centralising all state access here means:
#     - No scattered st.session_state["key"] references across files
#     - Easy to debug (all state in one place)
#     - Easy to reset (one reset() call clears everything cleanly)
#     - app.py stays clean and readable
# =============================================================================

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import streamlit as st

from src.vector_store import VectorStore
from src.llm_handler import LLMHandler

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session State Keys
# ---------------------------------------------------------------------------
# Centralise all session_state key names as constants.
# This prevents typos like st.session_state["mesages"] going undetected.

KEY_MESSAGES          = "messages"           # List of chat message dicts
KEY_VECTOR_STORE      = "vector_store"       # VectorStore instance
KEY_LLM_HANDLER       = "llm_handler"        # LLMHandler instance
KEY_PROCESSED_FILES   = "processed_files"    # Set of processed file hashes
KEY_FILE_METADATA     = "file_metadata"      # List of file info dicts
KEY_IS_PROCESSING     = "is_processing"      # Bool: pipeline running?
KEY_CHUNKS            = "chunks"             # List of all text chunks
KEY_TOTAL_CHUNKS      = "total_chunks"       # Int: total chunk count
KEY_INITIALIZED       = "chat_initialized"   # Bool: session set up?


# ---------------------------------------------------------------------------
# Session Initialisation
# ---------------------------------------------------------------------------

def initialise_session() -> None:
    """
    Initialise all required session state keys on first app load.

    This function is idempotent — calling it multiple times is safe.
    It only sets keys that don't already exist, so existing state
    (e.g. chat history from earlier in the session) is preserved.

    Call this at the very top of app.py before rendering any UI.
    """
    # Guard: only run full initialisation once per session.
    if st.session_state.get(KEY_INITIALIZED):
        return

    defaults = {
        KEY_MESSAGES:        [],         # Empty chat history
        KEY_VECTOR_STORE:    None,       # No index yet
        KEY_LLM_HANDLER:     None,       # No model yet
        KEY_PROCESSED_FILES: set(),      # No files processed
        KEY_FILE_METADATA:   [],         # No file info
        KEY_IS_PROCESSING:   False,      # Not currently processing
        KEY_CHUNKS:          [],         # No chunks yet
        KEY_TOTAL_CHUNKS:    0,          # Zero chunks
        KEY_INITIALIZED:     True,       # Mark as initialised
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    logger.info("Session state initialised.")


# ---------------------------------------------------------------------------
# Chat Message Management
# ---------------------------------------------------------------------------

def add_message(
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a new message to the chat history in session state.

    Each message is stored as a dictionary with a consistent structure
    so app.py can render them uniformly regardless of role.

    Message structure:
        {
            "role"      : "user" | "assistant",
            "content"   : str,         # The message text
            "sources"   : [...] | None,# Source chunks (assistant only)
            "metadata"  : {...} | None,# Extra info (timing, model, etc.)
            "timestamp" : str,         # ISO format timestamp
        }

    Args:
        role:     "user" or "assistant".
        content:  The message text to display.
        sources:  List of source chunk dicts for citation display
                  (only relevant for assistant messages).
        metadata: Optional dict with extra info like response time.
    """
    message = {
        "role":      role,
        "content":   content,
        "sources":   sources or [],
        "metadata":  metadata or {},
        "timestamp": datetime.now().isoformat(),
    }

    st.session_state[KEY_MESSAGES].append(message)

    logger.debug(
        f"Message added — role: {role}, "
        f"length: {len(content)} chars, "
        f"sources: {len(sources or [])}."
    )


def get_messages() -> List[Dict[str, Any]]:
    """
    Retrieve the full chat history from session state.

    Returns:
        List of message dicts in chronological order (oldest first).
        Returns empty list if no messages exist yet.
    """
    return st.session_state.get(KEY_MESSAGES, [])


def get_messages_for_llm() -> List[Dict[str, str]]:
    """
    Return chat history in the simplified format expected by build_rag_prompt().

    The LLM only needs role and content — not timestamps, sources, or metadata.
    This strips the extra fields so the prompt stays lean.

    Returns:
        List of {"role": str, "content": str} dicts.

    Example:
        [
            {"role": "user",      "content": "What is the revenue?"},
            {"role": "assistant", "content": "The revenue in Q3 was..."},
        ]
    """
    messages = get_messages()
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]


def clear_chat_history() -> None:
    """
    Clear all chat messages while preserving the vector index and file state.

    This allows the user to start a fresh conversation about the same
    uploaded documents without re-processing the PDFs.
    """
    st.session_state[KEY_MESSAGES] = []
    logger.info("Chat history cleared.")


def get_message_count() -> int:
    """Return the total number of messages in the chat history."""
    return len(st.session_state.get(KEY_MESSAGES, []))


# ---------------------------------------------------------------------------
# Vector Store Management
# ---------------------------------------------------------------------------

def set_vector_store(store: VectorStore) -> None:
    """
    Save a built VectorStore instance to session state.

    Args:
        store: A VectorStore instance that has had build_index() called.
    """
    st.session_state[KEY_VECTOR_STORE] = store
    logger.info(
        f"VectorStore saved to session state "
        f"({store.get_chunk_count()} vectors)."
    )


def get_vector_store() -> Optional[VectorStore]:
    """
    Retrieve the VectorStore instance from session state.

    Returns:
        The VectorStore instance, or None if not yet built.
    """
    return st.session_state.get(KEY_VECTOR_STORE)


def has_vector_store() -> bool:
    """
    Return True if a ready VectorStore exists in session state.
    """
    store = get_vector_store()
    return store is not None and store.is_ready()


# ---------------------------------------------------------------------------
# LLM Handler Management
# ---------------------------------------------------------------------------

def get_or_create_llm_handler() -> Optional[LLMHandler]:
    """
    Return the existing LLMHandler from session state, or create a new one.

    The LLMHandler is expensive to initialise (API client setup), so we
    create it once and reuse it for the entire session. This pattern is
    called "lazy initialisation" — we only create it when first needed.

    Returns:
        An initialised LLMHandler, or None if initialisation fails.
    """
    handler = st.session_state.get(KEY_LLM_HANDLER)

    # Return existing handler if it's already initialised and ready.
    if handler is not None and handler.is_ready():
        return handler

    # Create a new handler and save it to session state.
    logger.info("Creating new LLMHandler instance...")
    new_handler = LLMHandler()

    if new_handler.is_ready():
        st.session_state[KEY_LLM_HANDLER] = new_handler
        logger.info("LLMHandler created and saved to session state.")
        return new_handler
    else:
        logger.error("LLMHandler initialisation failed.")
        return None


# ---------------------------------------------------------------------------
# File Tracking
# ---------------------------------------------------------------------------

def is_file_processed(file_hash: str) -> bool:
    """
    Check whether a file (by hash) has already been processed this session.

    Used to avoid re-embedding a file the user uploads multiple times.

    Args:
        file_hash: MD5 hash of the file's bytes from utils.helpers.get_file_hash().

    Returns:
        True if this file was already processed, False otherwise.
    """
    return file_hash in st.session_state.get(KEY_PROCESSED_FILES, set())


def mark_file_processed(file_hash: str) -> None:
    """
    Record that a file has been successfully processed.

    Args:
        file_hash: MD5 hash of the processed file.
    """
    if KEY_PROCESSED_FILES not in st.session_state:
        st.session_state[KEY_PROCESSED_FILES] = set()
    st.session_state[KEY_PROCESSED_FILES].add(file_hash)


def add_file_metadata(metadata: Dict[str, Any]) -> None:
    """
    Store display metadata for an uploaded file.

    This metadata is shown in the UI sidebar as a summary card
    for each uploaded document.

    Args:
        metadata: Dict from pdf_processor.get_pdf_metadata().
                  Expected keys: filename, page_count, title, author, file_size.
    """
    if KEY_FILE_METADATA not in st.session_state:
        st.session_state[KEY_FILE_METADATA] = []

    # Avoid duplicate entries for the same filename.
    existing_names = [
        m.get("filename") for m in st.session_state[KEY_FILE_METADATA]
    ]
    if metadata.get("filename") not in existing_names:
        st.session_state[KEY_FILE_METADATA].append(metadata)


def get_file_metadata() -> List[Dict[str, Any]]:
    """
    Retrieve the list of file metadata dicts for all uploaded documents.

    Returns:
        List of metadata dicts, one per uploaded file.
    """
    return st.session_state.get(KEY_FILE_METADATA, [])


def get_processed_file_count() -> int:
    """Return the number of files processed this session."""
    return len(st.session_state.get(KEY_PROCESSED_FILES, set()))


# ---------------------------------------------------------------------------
# Chunk Management
# ---------------------------------------------------------------------------

def set_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Save the full list of text chunks to session state.

    Storing chunks lets us rebuild the index if needed without
    re-processing the PDFs.

    Args:
        chunks: List of chunk dicts from text_chunker.chunk_pages().
    """
    st.session_state[KEY_CHUNKS]       = chunks
    st.session_state[KEY_TOTAL_CHUNKS] = len(chunks)
    logger.info(f"{len(chunks)} chunks saved to session state.")


def get_chunks() -> List[Dict[str, Any]]:
    """Return the stored text chunks from session state."""
    return st.session_state.get(KEY_CHUNKS, [])


def get_total_chunk_count() -> int:
    """Return the total number of stored chunks."""
    return st.session_state.get(KEY_TOTAL_CHUNKS, 0)


# ---------------------------------------------------------------------------
# Processing State
# ---------------------------------------------------------------------------

def set_processing(is_processing: bool) -> None:
    """
    Set the processing flag to prevent concurrent pipeline runs.

    When True, the UI should disable the file uploader and chat input
    to prevent the user from triggering another pipeline run while
    one is already in progress.

    Args:
        is_processing: True when the pipeline is running, False when done.
    """
    st.session_state[KEY_IS_PROCESSING] = is_processing


def is_processing() -> bool:
    """
    Return True if the processing pipeline is currently running.
    """
    return st.session_state.get(KEY_IS_PROCESSING, False)


# ---------------------------------------------------------------------------
# Full Session Reset
# ---------------------------------------------------------------------------

def reset_session() -> None:
    """
    Completely reset all session state to the initial empty values.

    Called when the user clicks "Clear Everything" or uploads a completely
    new set of documents that should replace the previous session.

    This is more thorough than clear_chat_history() — it also clears
    the vector store, processed files, and chunk data.
    """
    keys_to_clear = [
        KEY_MESSAGES,
        KEY_VECTOR_STORE,
        KEY_LLM_HANDLER,
        KEY_PROCESSED_FILES,
        KEY_FILE_METADATA,
        KEY_IS_PROCESSING,
        KEY_CHUNKS,
        KEY_TOTAL_CHUNKS,
        KEY_INITIALIZED,
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    logger.info("Full session reset complete.")

    # Re-initialise with clean defaults immediately.
    initialise_session()


# ---------------------------------------------------------------------------
# Session Summary (for debugging / UI display)
# ---------------------------------------------------------------------------

def get_session_summary() -> Dict[str, Any]:
    """
    Return a summary of the current session state for display or debugging.

    Used in the sidebar to show the user a status overview:
    - How many documents are loaded
    - How many chunks are indexed
    - How many messages in the chat

    Returns:
        Dict with human-readable session statistics.
    """
    store   = get_vector_store()
    handler = st.session_state.get(KEY_LLM_HANDLER)

    return {
        "messages":          get_message_count(),
        "files_processed":   get_processed_file_count(),
        "total_chunks":      get_total_chunk_count(),
        "index_ready":       has_vector_store(),
        "index_size":        store.get_chunk_count() if store else 0,
        "llm_ready":         handler.is_ready() if handler else False,
        "is_processing":     is_processing(),
    }