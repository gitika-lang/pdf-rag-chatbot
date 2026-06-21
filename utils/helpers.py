# =============================================================================
# utils/helpers.py
# -----------------------------------------------------------------------------
# Pure utility functions shared across the project.
# No API calls, no Streamlit imports, no side effects.
# Each function does exactly one thing and is independently testable.
# =============================================================================

import re
import time
import hashlib
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean raw text extracted from a PDF.

    PDFs often contain artifacts like:
    - Multiple consecutive blank lines (from page layout)
    - Excessive whitespace between words (from column formatting)
    - Null bytes or non-printable characters

    Args:
        text: Raw string extracted from a PDF page.

    Returns:
        Cleaned string suitable for chunking and embedding.
    """
    if not text:
        return ""

    # Replace null bytes and other non-printable control characters
    # (except newlines \n and tabs \t which carry semantic meaning)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Collapse sequences of more than 2 newlines into exactly 2.
    # This preserves paragraph breaks while removing excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse sequences of spaces/tabs into a single space.
    # Handles PDF column artifacts where words have many spaces between them.
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove leading/trailing whitespace from each line.
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Final strip of the entire string.
    return text.strip()


def truncate_text(text: str, max_chars: int = 300, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum character length for display purposes.

    Used in the UI to show a preview of source chunks without overwhelming
    the user with the full chunk text.

    Args:
        text:      The string to truncate.
        max_chars: Maximum allowed character length (default 300).
        suffix:    String appended when truncation occurs (default "...").

    Returns:
        Original string if short enough, otherwise truncated string + suffix.
    """
    if not text or len(text) <= max_chars:
        return text
    # Cut at the last word boundary before max_chars to avoid mid-word cuts.
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + suffix


# ---------------------------------------------------------------------------
# File Utilities
# ---------------------------------------------------------------------------

def get_file_hash(file_bytes: bytes) -> str:
    """
    Generate a short MD5 hash string for a file's byte content.

    This is used to detect whether a file has already been processed
    in the current session, avoiding redundant re-embedding of the same PDF.

    Args:
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        First 16 characters of the MD5 hex digest (enough to be unique
        for session-level deduplication).
    """
    return hashlib.md5(file_bytes).hexdigest()[:16]


def format_file_size(size_bytes: int) -> str:
    """
    Convert a byte count into a human-readable file size string.

    Examples:
        500       → "500 B"
        2048      → "2.0 KB"
        1048576   → "1.0 MB"

    Args:
        size_bytes: File size in bytes.

    Returns:
        Human-readable string like "2.3 MB".
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 ** 2):.1f} MB"


# ---------------------------------------------------------------------------
# Chunk Formatting
# ---------------------------------------------------------------------------

def format_source_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Format a list of retrieved source chunks into a single context string
    that is injected into the LLM prompt.

    Each chunk dictionary is expected to have:
        - "text"        : The chunk's text content.
        - "source"      : The PDF filename the chunk came from.
        - "page"        : The page number within that PDF.
        - "chunk_index" : The chunk's position in the document.

    Args:
        chunks: List of chunk metadata dicts from the vector store.

    Returns:
        A formatted multi-line string ready to embed in the RAG prompt.

    Example output:
        [Source 1 | File: report.pdf | Page: 3]
        The quarterly revenue increased by 12%...

        [Source 2 | File: report.pdf | Page: 5]
        Operating expenses were reduced by...
    """
    if not chunks:
        return "No relevant context found."

    formatted_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "Unknown")
        page   = chunk.get("page", "?")
        text   = chunk.get("text", "").strip()

        header = f"[Source {i} | File: {source} | Page: {page}]"
        formatted_parts.append(f"{header}\n{text}")

    # Join chunks with a blank line separator for readability in the prompt.
    return "\n\n".join(formatted_parts)


def build_source_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build a deduplicated list of citation objects for display in the UI.

    Multiple retrieved chunks may come from the same file and page.
    This function deduplicates them so the UI doesn't show the same
    source twice.

    Args:
        chunks: List of chunk metadata dicts from the vector store.

    Returns:
        List of unique citation dicts, each with keys:
            - "source"  : Filename.
            - "page"    : Page number as a string.
            - "preview" : Truncated text preview for display.
    """
    seen = set()
    citations = []

    for chunk in chunks:
        source  = chunk.get("source", "Unknown")
        page    = str(chunk.get("page", "?"))
        text    = chunk.get("text", "")
        preview = truncate_text(text, max_chars=200)

        # Use (source, page) as the deduplication key.
        key = (source, page)
        if key not in seen:
            seen.add(key)
            citations.append({
                "source":  source,
                "page":    page,
                "preview": preview,
            })

    return citations


# ---------------------------------------------------------------------------
# Timing Utility
# ---------------------------------------------------------------------------

class Timer:
    """
    A simple context manager for measuring elapsed time.

    Usage:
        with Timer() as t:
            do_something_slow()
        print(f"Took {t.elapsed:.2f}s")
    """

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed: float = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def is_valid_question(text: str, min_length: int = 3) -> bool:
    """
    Check whether a user's input is a valid, non-empty question.

    Prevents the pipeline from running on accidental empty submissions
    or single-character inputs.

    Args:
        text:       The user's raw input string.
        min_length: Minimum number of non-whitespace characters required.

    Returns:
        True if the input is usable, False otherwise.
    """
    if not text:
        return False
    stripped = text.strip()
    return len(stripped) >= min_length


def validate_api_key(api_key: str) -> bool:
    """
    Perform a basic sanity check on the Gemini API key format.

    This is NOT a live API call — it only checks that the key looks
    plausible (non-empty, reasonable length, no obvious placeholder text).

    Args:
        api_key: The API key string from the environment.

    Returns:
        True if the key passes basic validation, False otherwise.
    """
    if not api_key:
        return False
    if api_key in ("YOUR_GEMINI_API_KEY", "your-api-key-here"):
        return False
    # Gemini API keys are typically 39 characters long.
    if len(api_key) < 20:
        return False
    return True