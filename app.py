# =============================================================================
# app.py
# -----------------------------------------------------------------------------
# Main entry point for the PDF RAG Chatbot Streamlit application.
#
# Run with:
#     streamlit run app.py
#
# This file orchestrates all modules:
#   - Renders the full Streamlit UI (sidebar + chat window)
#   - Handles file uploads and triggers the processing pipeline
#   - Routes user queries through the RAG pipeline
#   - Displays answers with source citations
#   - Manages all session state via chat_manager.py
#
# Architecture: this file contains ONLY UI logic.
#   All heavy lifting (PDF parsing, embedding, retrieval, LLM calls)
#   is delegated to the src/ modules.
# =============================================================================

import logging
import time
from typing import List, Dict

import streamlit as st

# --- Internal Modules ---
from config.settings import (
    APP_TITLE,
    APP_SUBTITLE,
    CHAT_INPUT_PLACEHOLDER,
    MAX_UPLOAD_SIZE_MB,
    GEMINI_API_KEY,
)
from src.pdf_processor  import process_multiple_pdfs, get_pdf_metadata
from src.text_chunker   import chunk_pages, get_chunk_stats
from src.embeddings     import initialise_gemini, embed_chunks, embed_query, validate_embeddings
from src.vector_store   import VectorStore
from src.chat_manager   import (
    initialise_session,
    add_message,
    get_messages,
    get_messages_for_llm,
    clear_chat_history,
    reset_session,
    set_vector_store,
    get_vector_store,
    has_vector_store,
    get_or_create_llm_handler,
    is_file_processed,
    mark_file_processed,
    add_file_metadata,
    get_file_metadata,
    set_chunks,
    get_total_chunk_count,
    set_processing,
    is_processing,
    get_session_summary,
)
from utils.helpers import (
    validate_api_key,
    format_file_size,
    build_source_citations,
    is_valid_question,
    Timer,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
# Must be the FIRST Streamlit call in the script.
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Inject custom CSS to polish the default Streamlit styling."""
    st.markdown("""
    <style>
        /* ---- Main header ---- */
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .main-header h1 { color: #e2e8f0; font-size: 2rem; font-weight: 700; margin: 0; }
        .main-header p  { color: #94a3b8; font-size: 1rem; margin: 0.5rem 0 0 0; }

        /* ---- Source citation cards ---- */
        .source-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin: 0.4rem 0;
            font-size: 0.85rem;
        }
        .source-card .source-title   { color: #60a5fa; font-weight: 600; margin-bottom: 0.3rem; }
        .source-card .source-preview { color: #94a3b8; line-height: 1.5; }

        /* ---- Status badges ---- */
        .status-badge   { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
        .status-ready   { background: #064e3b; color: #6ee7b7; }
        .status-pending { background: #451a03; color: #fdba74; }

        /* ---- Sidebar section headers ---- */
        .sidebar-section { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin: 1rem 0 0.3rem 0; }

        /* ---- Hide Streamlit branding ---- */
        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }
        header    { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ===========================================================================
# SIDEBAR
# ===========================================================================

def render_sidebar() -> None:
    """
    Render the full sidebar UI:
    1. API key status
    2. PDF file uploader + Process button
    3. Loaded documents summary
    4. Index statistics
    5. Action buttons
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        _render_api_key_status()
        st.divider()

        # --- File Uploader ---
        st.markdown("## 📂 Upload Documents")
        uploaded_files = st.file_uploader(
            label="Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Maximum file size: {MAX_UPLOAD_SIZE_MB} MB per file.",
            disabled=is_processing(),
        )

        if uploaded_files:
            st.caption(f"{len(uploaded_files)} file(s) selected")
            if st.button(
                label="🚀 Process Documents",
                type="primary",
                use_container_width=True,
                disabled=is_processing(),
                help="Extract text, generate embeddings, and build the search index.",
            ):
                _run_processing_pipeline(uploaded_files)

        st.divider()
        _render_file_info()
        _render_index_stats()
        st.divider()
        _render_action_buttons()


def _render_api_key_status() -> None:
    """Show a coloured badge for API key status."""
    if validate_api_key(GEMINI_API_KEY):
        st.markdown(
            '<span class="status-badge status-ready">✓ API Key Configured</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-pending">✗ API Key Missing</span>',
            unsafe_allow_html=True,
        )
        st.error(
            "**Gemini API key not found.**\n\n"
            "Add `GEMINI_API_KEY=your_key` to your `.env` file and restart the app.",
            icon="🔑",
        )


def _render_file_info() -> None:
    """Render a summary card for each successfully processed PDF."""
    file_metadata = get_file_metadata()

    if not file_metadata:
        st.markdown(
            '<p class="sidebar-section">No documents loaded</p>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<p class="sidebar-section">📄 Loaded Documents ({len(file_metadata)})</p>',
        unsafe_allow_html=True,
    )

    for meta in file_metadata:
        with st.expander(f"📄 {meta.get('filename', 'Unknown')}", expanded=False):
            c1, c2 = st.columns(2)
            c1.metric("Pages", meta.get("page_count", "?"))
            c2.metric("Size",  format_file_size(meta.get("file_size", 0)))
            if meta.get("title")  and meta["title"]  != "N/A":
                st.caption(f"**Title:** {meta['title']}")
            if meta.get("author") and meta["author"] != "N/A":
                st.caption(f"**Author:** {meta['author']}")


def _render_index_stats() -> None:
    """Render index statistics when the index is ready."""
    summary = get_session_summary()
    if not summary["index_ready"]:
        return

    st.markdown('<p class="sidebar-section">🔍 Index Statistics</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Chunks",    summary["index_size"])
    c2.metric("Documents", summary["files_processed"])
    st.markdown(
        '<span class="status-badge status-ready">✓ Index Ready</span>',
        unsafe_allow_html=True,
    )


def _render_action_buttons() -> None:
    """Render Clear Chat and Reset Everything buttons."""
    st.markdown("## 🛠️ Actions")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🗑️ Clear Chat", use_container_width=True,
                     help="Clear conversation history (keeps the document index)."):
            clear_chat_history()
            st.success("Chat cleared!")
            st.rerun()

    with c2:
        if st.button("🔄 Reset All", use_container_width=True,
                     help="Clear everything including uploaded documents."):
            reset_session()
            st.success("Session reset!")
            st.rerun()


# ===========================================================================
# PROCESSING PIPELINE
# ===========================================================================

def _run_processing_pipeline(uploaded_files: list) -> None:
    """
    Run the full PDF → Chunks → Embeddings → FAISS pipeline with live
    progress feedback in the sidebar.

    Steps:
        1. Extract text from PDFs       (pdf_processor)
        2. Split text into chunks       (text_chunker)
        3. Generate embeddings          (embeddings)
        4. Build FAISS index            (vector_store)
    """
    if not validate_api_key(GEMINI_API_KEY):
        st.sidebar.error("Cannot process: Gemini API key is missing.")
        return

    set_processing(True)
    progress_container = st.sidebar.container()

    with progress_container:
        st.markdown("---")
        progress_bar = st.progress(0, text="Starting...")
        status_text  = st.empty()

        try:
            # -----------------------------------------------------------
            # STEP 1 — Extract text
            # -----------------------------------------------------------
            status_text.info("📖 Step 1/4: Extracting text from PDFs...")
            progress_bar.progress(10, text="Extracting PDF text...")

            with Timer() as t_extract:
                pages, failed = process_multiple_pdfs(uploaded_files)

            if failed:
                st.sidebar.warning(f"⚠️ Could not process: {', '.join(failed)}")

            if not pages:
                st.sidebar.error(
                    "No text could be extracted. Ensure the PDFs contain "
                    "selectable text (not scanned images)."
                )
                set_processing(False)
                return

            status_text.success(
                f"✅ Step 1/4: Extracted {len(pages)} page(s) in {t_extract}."
            )
            progress_bar.progress(30, text="Text extracted...")

            # Save metadata for each file.
            for uf in uploaded_files:
                if uf.name not in failed:
                    uf.seek(0)
                    meta = get_pdf_metadata(uf.read(), uf.name)
                    add_file_metadata(meta)
                    mark_file_processed(meta.get("file_hash", uf.name))
                    uf.seek(0)

            # -----------------------------------------------------------
            # STEP 2 — Chunk text
            # -----------------------------------------------------------
            status_text.info("✂️  Step 2/4: Splitting text into chunks...")
            progress_bar.progress(35, text="Chunking text...")

            with Timer() as t_chunk:
                chunks = chunk_pages(pages)

            if not chunks:
                st.sidebar.error("Text extracted but could not be chunked.")
                set_processing(False)
                return

            stats = get_chunk_stats(chunks)
            set_chunks(chunks)

            status_text.success(
                f"✅ Step 2/4: {stats['total_chunks']} chunks "
                f"(avg {stats['avg_chars']} chars) in {t_chunk}."
            )
            progress_bar.progress(50, text="Chunks ready...")

            # -----------------------------------------------------------
            # STEP 3 — Generate embeddings
            # -----------------------------------------------------------
            status_text.info(
                f"🧮 Step 3/4: Embedding {len(chunks)} chunks "
                f"(this may take a moment)..."
            )
            progress_bar.progress(55, text="Generating embeddings...")

            if not initialise_gemini():
                st.sidebar.error("Failed to initialise Gemini API.")
                set_processing(False)
                return

            with Timer() as t_embed:
                embeddings = embed_chunks(chunks)

            if embeddings is None or not validate_embeddings(embeddings, len(chunks)):
                st.sidebar.error("Embedding generation failed.")
                set_processing(False)
                return

            status_text.success(
                f"✅ Step 3/4: {embeddings.shape[0]} embeddings "
                f"(dim={embeddings.shape[1]}) in {t_embed}."
            )
            progress_bar.progress(80, text="Embeddings ready...")

            # -----------------------------------------------------------
            # STEP 4 — Build FAISS index
            # -----------------------------------------------------------
            status_text.info("🗂️  Step 4/4: Building search index...")
            progress_bar.progress(85, text="Building FAISS index...")

            with Timer() as t_index:
                store   = VectorStore()
                success = store.build_index(embeddings, chunks)

            if not success:
                st.sidebar.error("Failed to build the search index.")
                set_processing(False)
                return

            set_vector_store(store)
            status_text.success(
                f"✅ Step 4/4: {store.get_chunk_count()} vectors indexed in {t_index}."
            )
            progress_bar.progress(100, text="Complete!")

            # -----------------------------------------------------------
            # Success
            # -----------------------------------------------------------
            time.sleep(0.5)
            progress_container.empty()

            st.sidebar.success(
                f"🎉 **Ready to chat!**\n\n"
                f"- 📄 {len(uploaded_files) - len(failed)} document(s) loaded\n"
                f"- ✂️ {stats['total_chunks']} text chunks\n"
                f"- 🔍 {store.get_chunk_count()} vectors indexed"
            )

            # Add welcome message if chat is empty.
            if not get_messages():
                source_list = ", ".join(
                    f"`{f.name}`" for f in uploaded_files if f.name not in failed
                )
                add_message(
                    role="assistant",
                    content=(
                        f"✅ Your documents are ready! I've processed {source_list} "
                        f"and indexed **{stats['total_chunks']} text chunks**.\n\n"
                        f"Ask me anything about the content — I'll answer strictly "
                        f"from your documents."
                    ),
                )

            st.rerun()

        except Exception as e:
            logger.error(f"Processing pipeline failed: {e}", exc_info=True)
            progress_container.empty()
            st.sidebar.error(f"Unexpected error during processing:\n`{str(e)}`")

        finally:
            set_processing(False)


# ===========================================================================
# CHAT INTERFACE
# ===========================================================================

def render_chat_interface() -> None:
    """
    Render the main chat area:
    - App header
    - Onboarding screen (no documents loaded)
    - Message history with source citations
    - Chat input box
    """
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>📄 PDF RAG Chatbot</h1>
            <p>Ask questions about your uploaded PDF documents</p>
        </div>
    """, unsafe_allow_html=True)

    # Onboarding when no index exists
    if not has_vector_store():
        _render_onboarding()
        return

    # Render all messages
    for message in get_messages():
        role    = message["role"]
        content = message["content"]
        sources = message.get("sources", [])

        with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
            st.markdown(content)
            if role == "assistant" and sources:
                _render_source_citations(sources)

    # Chat input
    _render_chat_input()


def _render_onboarding() -> None:
    """Welcome screen shown before any documents are uploaded."""
    st.markdown("<br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.info(
            "### 👋 Welcome!\n\n"
            "To get started:\n\n"
            "1. **Upload** one or more PDF files using the sidebar\n"
            "2. Click **Process Documents** to index the content\n"
            "3. **Ask questions** in natural language\n\n"
            "The chatbot answers strictly from your documents.",
            icon="📄",
        )
        st.markdown("#### 💡 Example questions:")
        for ex in [
            "What is the main topic of this document?",
            "Summarise the key findings.",
            "What does the document say about [topic]?",
            "List all recommendations mentioned.",
        ]:
            st.markdown(f"- *{ex}*")


def _render_source_citations(sources: List[Dict]) -> None:
    """
    Render collapsible source citation cards below an assistant message.

    Args:
        sources: List of citation dicts with keys: source, page, preview.
    """
    with st.expander(f"📚 Sources ({len(sources)} reference(s))", expanded=False):
        for i, citation in enumerate(sources, start=1):
            st.markdown(f"""
                <div class="source-card">
                    <div class="source-title">
                        [{i}] 📄 {citation.get('source','?')} — Page {citation.get('page','?')}
                    </div>
                    <div class="source-preview">{citation.get('preview','')}</div>
                </div>
            """, unsafe_allow_html=True)


def _render_chat_input() -> None:
    """
    Render the chat input and trigger the RAG pipeline on submission.
    """
    user_input = st.chat_input(
        placeholder=CHAT_INPUT_PLACEHOLDER,
        disabled=is_processing(),
    )

    if not user_input:
        return

    if not is_valid_question(user_input):
        st.warning("Please enter a valid question (at least 3 characters).")
        return

    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    add_message(role="user", content=user_input)

    # Generate and display assistant response
    _handle_user_query(user_input)


def _handle_user_query(query: str) -> None:
    """
    Full RAG pipeline for one user query:
        embed query → FAISS search → stream LLM answer → save to history

    Args:
        query: The user's question.
    """
    with st.chat_message("assistant", avatar="🤖"):

        # --- Validate index ---
        store = get_vector_store()
        if not store or not store.is_ready():
            msg = "⚠️ Document index not ready. Please upload and process PDFs first."
            st.error(msg)
            add_message(role="assistant", content=msg)
            return

        # --- Validate LLM ---
        handler = get_or_create_llm_handler()
        if not handler:
            msg = "⚠️ Could not initialise the AI model. Check your Gemini API key."
            st.error(msg)
            add_message(role="assistant", content=msg)
            return

        # --- Embed query ---
        with st.spinner("🔍 Searching documents..."):
            query_vector = embed_query(query)

        if query_vector is None:
            msg = "⚠️ Could not embed your query. Check your API key and try again."
            st.error(msg)
            add_message(role="assistant", content=msg)
            return

        # --- FAISS search ---
        retrieved_chunks = store.search(query_vector)

        if not retrieved_chunks:
            msg = (
                "I couldn't find relevant information in the uploaded documents. "
                "Try rephrasing or ask about a topic covered in your PDFs."
            )
            st.warning(msg)
            add_message(role="assistant", content=msg)
            return

        # --- Stream answer ---
        chat_history = get_messages_for_llm()

        with Timer() as t:
            full_answer = st.write_stream(
                handler.generate_answer_stream(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    chat_history=chat_history[:-1],  # Exclude current question
                )
            )

        # --- Source citations ---
        citations = build_source_citations(retrieved_chunks)
        if citations:
            _render_source_citations(citations)

        # --- Persist to session history ---
        add_message(
            role="assistant",
            content=full_answer,
            sources=citations,
            metadata={
                "response_time_s":  round(t.elapsed, 2),
                "chunks_retrieved": len(retrieved_chunks),
                "top_score":        round(retrieved_chunks[0].get("score", 0), 4),
            },
        )

        logger.info(
            f"Query answered in {t} | "
            f"chunks={len(retrieved_chunks)} | "
            f"top_score={retrieved_chunks[0].get('score', 0):.4f}"
        )


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """
    Application entry point — called by `streamlit run app.py`.

    Order matters:
    1. inject_custom_css()     — must run before any widget renders
    2. initialise_session()    — must run before any session_state access
    3. render_sidebar()        — file upload + controls
    4. render_chat_interface() — main chat area
    """
    inject_custom_css()
    initialise_session()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()