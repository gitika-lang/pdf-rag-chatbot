# =============================================================================
# src/pdf_processor.py
# -----------------------------------------------------------------------------
# Handles all PDF loading and text extraction logic.
#
# Responsibilities:
#   - Accept uploaded file bytes (from Streamlit's file uploader)
#   - Extract text from each page using PyMuPDF (fitz)
#   - Clean extracted text using utils/helpers.py
#   - Return structured page-level data with metadata (filename, page number)
#   - Handle corrupt, empty, or password-protected PDFs gracefully
#
# Why PyMuPDF (fitz)?
#   - Fastest Python PDF parser (C++ core, Python bindings)
#   - Handles complex layouts, embedded fonts, and multi-column text well
#   - Provides page-level access, which we need for citation tracking
#   - Much more reliable than pdfplumber or PyPDF2 for real-world PDFs
# =============================================================================

import io
import logging
from typing import List, Dict, Any

import fitz  # PyMuPDF — installed as the "pymupdf" package

from utils.helpers import clean_text, get_file_hash

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# Using Python's logging module instead of print() is an industry standard.
# It allows log levels (DEBUG, INFO, WARNING, ERROR) and can be routed to
# files or monitoring systems without changing any code.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Data Structure
# ---------------------------------------------------------------------------
# Every page extracted from a PDF is returned as a dictionary with this shape:
#
#   {
#       "text"       : str,   # Cleaned text content of the page
#       "page"       : int,   # 1-based page number (human-friendly)
#       "source"     : str,   # Original filename (e.g. "report.pdf")
#       "file_hash"  : str,   # MD5 hash of the file for deduplication
#       "char_count" : int,   # Character count after cleaning
#   }
#
# This structure flows through the entire pipeline:
#   pdf_processor → text_chunker → embeddings → vector_store


# ---------------------------------------------------------------------------
# Main Extraction Function
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    file_bytes: bytes,
    filename: str,
) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file provided as raw bytes.

    This is the primary entry point for PDF processing. It is designed to
    accept Streamlit's UploadedFile bytes directly:
        file_bytes = uploaded_file.read()

    Args:
        file_bytes: Raw bytes of the PDF file.
        filename:   Original filename shown to the user (e.g. "research.pdf").

    Returns:
        A list of page dictionaries (see data structure above).
        Returns an empty list if extraction fails or the PDF has no text.

    Raises:
        Does NOT raise — all exceptions are caught and logged so the
        Streamlit app can display a user-friendly error instead of crashing.
    """
    if not file_bytes:
        logger.warning(f"Empty file bytes received for '{filename}'.")
        return []

    # Compute a hash of the file for session-level deduplication.
    file_hash = get_file_hash(file_bytes)
    logger.info(f"Processing PDF: '{filename}' | Hash: {file_hash}")

    pages: List[Dict[str, Any]] = []

    try:
        # Open the PDF from memory (bytes) rather than a file path.
        # fitz.open() accepts a stream parameter for in-memory files,
        # which is ideal for Streamlit where files live in memory, not disk.
        pdf_stream = io.BytesIO(file_bytes)
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

        # Check for password protection before attempting page access.
        if pdf_document.needs_pass:
            logger.error(f"PDF '{filename}' is password-protected.")
            pdf_document.close()
            return []

        total_pages = len(pdf_document)
        logger.info(f"PDF '{filename}' has {total_pages} page(s).")

        if total_pages == 0:
            logger.warning(f"PDF '{filename}' contains no pages.")
            pdf_document.close()
            return []

        # Iterate over every page in the document.
        for page_num in range(total_pages):
            page_data = _extract_page(
                pdf_document=pdf_document,
                page_num=page_num,
                filename=filename,
                file_hash=file_hash,
            )
            # Only include pages that have usable text content.
            if page_data is not None:
                pages.append(page_data)

        pdf_document.close()

        # Summary log — useful for debugging during development.
        usable_pages = len(pages)
        logger.info(
            f"Extracted text from {usable_pages}/{total_pages} pages "
            f"in '{filename}'."
        )

        if usable_pages == 0:
            logger.warning(
                f"No text could be extracted from '{filename}'. "
                f"The PDF may be image-based (scanned) and require OCR."
            )

    except fitz.FileDataError as e:
        # Raised when the file is corrupt or not a valid PDF.
        logger.error(f"Corrupt or invalid PDF '{filename}': {e}")
        return []

    except Exception as e:
        # Catch-all for any unexpected errors during processing.
        logger.error(f"Unexpected error processing '{filename}': {e}")
        return []

    return pages


# ---------------------------------------------------------------------------
# Per-Page Extraction (Private Helper)
# ---------------------------------------------------------------------------

def _extract_page(
    pdf_document: fitz.Document,
    page_num: int,
    filename: str,
    file_hash: str,
) -> Dict[str, Any] | None:
    """
    Extract and clean text from a single PDF page.

    This is a private helper (prefixed with _) — it should only be called
    by extract_text_from_pdf(), not by external modules.

    Args:
        pdf_document: An open fitz.Document object.
        page_num:     Zero-based page index.
        filename:     PDF filename for metadata.
        file_hash:    Pre-computed file hash for metadata.

    Returns:
        A page dictionary if the page has usable text, otherwise None.
    """
    try:
        page = pdf_document[page_num]

        # get_text("text") extracts plain text in reading order.
        # Other options: "html", "dict", "blocks" — but "text" is
        # cleanest for our chunking pipeline.
        raw_text = page.get_text("text")

        # Clean the raw extracted text using our utility function.
        cleaned = clean_text(raw_text)

        # Skip pages with no meaningful content (e.g. blank pages,
        # pages that are pure images with no text layer).
        if not cleaned or len(cleaned) < 20:
            logger.debug(
                f"Skipping page {page_num + 1} of '{filename}' "
                f"— insufficient text ({len(cleaned)} chars)."
            )
            return None

        return {
            "text":       cleaned,
            "page":       page_num + 1,   # Convert to 1-based for display
            "source":     filename,
            "file_hash":  file_hash,
            "char_count": len(cleaned),
        }

    except Exception as e:
        logger.error(
            f"Error extracting page {page_num + 1} from '{filename}': {e}"
        )
        return None


# ---------------------------------------------------------------------------
# Multi-File Processing
# ---------------------------------------------------------------------------

def process_multiple_pdfs(
    uploaded_files: List[Any],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Process a list of Streamlit UploadedFile objects.

    Iterates over each uploaded file, extracts text, and aggregates results.
    Files that fail to process are tracked separately so the UI can show
    the user which files succeeded and which had issues.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects from
                        st.file_uploader(..., accept_multiple_files=True).

    Returns:
        A tuple of:
            - all_pages   : Combined list of page dicts from all PDFs.
            - failed_files: List of filenames that failed to process.

    Example:
        pages, failures = process_multiple_pdfs(st.session_state.uploaded_files)
        if failures:
            st.warning(f"Could not process: {', '.join(failures)}")
    """
    all_pages: List[Dict[str, Any]] = []
    failed_files: List[str] = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        try:
            # Read the file bytes from Streamlit's UploadedFile object.
            # .read() returns the complete file content as bytes.
            file_bytes = uploaded_file.read()

            pages = extract_text_from_pdf(file_bytes, filename)

            if pages:
                all_pages.extend(pages)
                logger.info(
                    f"Successfully processed '{filename}': "
                    f"{len(pages)} page(s) extracted."
                )
            else:
                # extract_text_from_pdf returned empty — file had issues.
                failed_files.append(filename)
                logger.warning(f"No content extracted from '{filename}'.")

        except Exception as e:
            failed_files.append(filename)
            logger.error(f"Failed to read uploaded file '{filename}': {e}")

    logger.info(
        f"PDF processing complete: {len(all_pages)} total pages from "
        f"{len(uploaded_files) - len(failed_files)} file(s). "
        f"{len(failed_files)} file(s) failed."
    )

    return all_pages, failed_files


# ---------------------------------------------------------------------------
# Utility: Get PDF Metadata
# ---------------------------------------------------------------------------

def get_pdf_metadata(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract document-level metadata from a PDF (title, author, page count).

    This is used in the UI to display a summary card when a PDF is uploaded,
    giving the user confirmation that their file was read correctly.

    Args:
        file_bytes: Raw bytes of the PDF.
        filename:   Original filename for display.

    Returns:
        Dictionary with keys: filename, page_count, title, author, file_size.
    """
    metadata = {
        "filename":   filename,
        "page_count": 0,
        "title":      "N/A",
        "author":     "N/A",
        "file_size":  len(file_bytes),
    }

    try:
        pdf_stream   = io.BytesIO(file_bytes)
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

        metadata["page_count"] = len(pdf_document)

        # fitz exposes PDF metadata as a dictionary.
        # Keys: title, author, subject, keywords, creator, producer, etc.
        doc_meta = pdf_document.metadata
        if doc_meta:
            metadata["title"]  = doc_meta.get("title", "N/A") or "N/A"
            metadata["author"] = doc_meta.get("author", "N/A") or "N/A"

        pdf_document.close()

    except Exception as e:
        logger.error(f"Could not read metadata from '{filename}': {e}")

    return metadata