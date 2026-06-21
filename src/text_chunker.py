# =============================================================================
# src/text_chunker.py
# -----------------------------------------------------------------------------
# Splits extracted PDF page text into overlapping chunks for embedding.
#
# Responsibilities:
#   - Accept page-level data from pdf_processor.py
#   - Split long page text into smaller, overlapping chunks
#   - Preserve and propagate page metadata (source, page number) per chunk
#   - Filter out chunks that are too short to be meaningful
#   - Return a flat list of chunk dicts ready for embedding
#
# Why chunking matters:
#   - LLMs have a context window limit — we can't feed an entire PDF at once
#   - Embedding models work best on short, focused passages (not whole pages)
#   - Smaller chunks = more precise retrieval = better answers
#   - Overlap between chunks prevents answers from being lost at boundaries
#
# Why RecursiveCharacterTextSplitter?
#   - Tries natural split points first: \n\n → \n → " " → ""
#   - This keeps sentences and paragraphs intact wherever possible
#   - Falls back to character-level splitting only when necessary
#   - Battle-tested in thousands of production RAG systems
# =============================================================================

import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimum Chunk Length
# ---------------------------------------------------------------------------
# Chunks shorter than this many characters are discarded.
# Very short chunks (e.g. page headers, footers, lone page numbers) add
# noise to the vector store without contributing useful context.
MIN_CHUNK_LENGTH: int = 50


# ---------------------------------------------------------------------------
# Core Chunking Function
# ---------------------------------------------------------------------------

def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split a list of page dictionaries into smaller overlapping text chunks.

    This is the primary entry point for this module. It processes all pages
    from all uploaded PDFs and returns a flat list of chunk dictionaries
    ready to be passed to the embedding module.

    Each output chunk dictionary has this structure:
        {
            "text"        : str,  # The chunk's text content
            "source"      : str,  # Original PDF filename
            "page"        : int,  # Page number the chunk came from
            "file_hash"   : str,  # File hash for deduplication
            "chunk_index" : int,  # Global index of this chunk (0-based)
            "char_count"  : int,  # Character length of this chunk
        }

    Args:
        pages: List of page dicts produced by pdf_processor.extract_text_from_pdf().
               Each dict must have keys: "text", "source", "page", "file_hash".

    Returns:
        Flat list of chunk dicts. Returns empty list if input is empty
        or no valid chunks could be produced.

    Example:
        pages  = pdf_processor.extract_text_from_pdf(file_bytes, "doc.pdf")
        chunks = chunk_pages(pages)
        # chunks is now ready for embeddings.embed_chunks(chunks)
    """
    if not pages:
        logger.warning("chunk_pages() received an empty pages list.")
        return []

    # Initialise the splitter once and reuse it for all pages.
    # This avoids recreating the object (and its compiled regex) per page.
    splitter = _build_splitter()

    all_chunks: List[Dict[str, Any]] = []
    chunk_index = 0  # Global chunk counter across all pages and files

    for page in pages:
        page_text   = page.get("text", "")
        source      = page.get("source", "unknown")
        page_number = page.get("page", 0)
        file_hash   = page.get("file_hash", "")

        # Skip pages that somehow have no text at this stage.
        if not page_text.strip():
            logger.debug(
                f"Skipping empty page {page_number} from '{source}'."
            )
            continue

        # Split this page's text into raw string chunks.
        raw_chunks = splitter.split_text(page_text)

        logger.debug(
            f"Page {page_number} of '{source}': "
            f"split into {len(raw_chunks)} raw chunk(s)."
        )

        # Wrap each raw string chunk in a metadata dictionary.
        for raw_chunk in raw_chunks:
            cleaned_chunk = raw_chunk.strip()

            # Discard chunks that are too short to be meaningful.
            if len(cleaned_chunk) < MIN_CHUNK_LENGTH:
                logger.debug(
                    f"Discarding short chunk ({len(cleaned_chunk)} chars) "
                    f"from page {page_number} of '{source}'."
                )
                continue

            all_chunks.append({
                "text":        cleaned_chunk,
                "source":      source,
                "page":        page_number,
                "file_hash":   file_hash,
                "chunk_index": chunk_index,
                "char_count":  len(cleaned_chunk),
            })
            chunk_index += 1

    # Summary statistics — useful during development and debugging.
    total_pages  = len(pages)
    total_chunks = len(all_chunks)

    if total_chunks == 0:
        logger.warning(
            f"No valid chunks produced from {total_pages} page(s). "
            f"Check that the PDF contains extractable text."
        )
    else:
        avg_chars = sum(c["char_count"] for c in all_chunks) // total_chunks
        logger.info(
            f"Chunking complete: {total_chunks} chunks from "
            f"{total_pages} page(s) | "
            f"Avg chunk size: {avg_chars} chars."
        )

    return all_chunks


# ---------------------------------------------------------------------------
# Splitter Factory (Private)
# ---------------------------------------------------------------------------

def _build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build and return a configured RecursiveCharacterTextSplitter instance.

    Separators are tried in order — the splitter moves to the next separator
    only when a chunk would still exceed chunk_size after splitting on the
    current one.

    Separator priority:
        1. "\\n\\n"  — Paragraph breaks (strongest natural boundary)
        2. "\\n"     — Line breaks
        3. ". "      — Sentence endings
        4. ", "      — Clause boundaries
        5. " "       — Word boundaries
        6. ""        — Character-level (last resort)

    Returns:
        A configured RecursiveCharacterTextSplitter ready to use.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        # length_function defines how "size" is measured.
        # len() counts characters, which maps directly to our CHUNK_SIZE.
        length_function=len,
        # is_separator_regex=False means separators are treated as
        # plain strings, not regular expressions. Safer and faster.
        is_separator_regex=False,
    )


# ---------------------------------------------------------------------------
# Utility: Chunk Statistics
# ---------------------------------------------------------------------------

def get_chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics for a list of chunks.

    Used by the Streamlit UI to display an informational summary after
    processing, e.g. "Processed 3 files → 142 chunks, avg 487 chars".

    Args:
        chunks: List of chunk dicts produced by chunk_pages().

    Returns:
        Dictionary with keys:
            - total_chunks   : int  — Total number of chunks
            - avg_chars      : int  — Average characters per chunk
            - min_chars      : int  — Shortest chunk length
            - max_chars      : int  — Longest chunk length
            - unique_sources : list — Unique PDF filenames in the chunk set
            - chunks_per_source : dict — {filename: chunk_count} breakdown
    """
    if not chunks:
        return {
            "total_chunks":      0,
            "avg_chars":         0,
            "min_chars":         0,
            "max_chars":         0,
            "unique_sources":    [],
            "chunks_per_source": {},
        }

    char_counts = [c["char_count"] for c in chunks]

    # Count chunks per source file.
    chunks_per_source: Dict[str, int] = {}
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        chunks_per_source[source] = chunks_per_source.get(source, 0) + 1

    return {
        "total_chunks":      len(chunks),
        "avg_chars":         sum(char_counts) // len(char_counts),
        "min_chars":         min(char_counts),
        "max_chars":         max(char_counts),
        "unique_sources":    list(chunks_per_source.keys()),
        "chunks_per_source": chunks_per_source,
    }


# ---------------------------------------------------------------------------
# Utility: Filter Chunks by Source
# ---------------------------------------------------------------------------

def filter_chunks_by_source(
    chunks: List[Dict[str, Any]],
    source_filename: str,
) -> List[Dict[str, Any]]:
    """
    Return only the chunks that belong to a specific PDF file.

    Useful when the user uploads multiple PDFs and wants to query only one.
    Not used in the main pipeline but available for future UI features.

    Args:
        chunks:          Full list of chunks from chunk_pages().
        source_filename: The filename to filter by (e.g. "report.pdf").

    Returns:
        Subset of chunks where chunk["source"] == source_filename.
    """
    return [c for c in chunks if c.get("source") == source_filename]