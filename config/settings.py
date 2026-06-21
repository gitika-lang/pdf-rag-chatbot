# =============================================================================
# config/settings.py
# -----------------------------------------------------------------------------
# Central configuration for the PDF RAG Chatbot.
# All tuneable parameters live here — never scatter magic numbers in code.
# To adjust behaviour (e.g. larger chunks, different model), edit this file only.
# =============================================================================

import os
from dotenv import load_dotenv

# Load environment variables from the .env file at project root.
# This must happen before we read os.environ below.
load_dotenv()


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

# Your Google Gemini API key, loaded from the .env file.
# Never hard-code a key directly in source code.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")


# ---------------------------------------------------------------------------
# Gemini Model Names
# ---------------------------------------------------------------------------

# The generative model used to produce chat answers.
# "gemini-1.5-flash" is fast, cheap, and has a generous free-tier quota —
# ideal for student projects and demos.
GEMINI_CHAT_MODEL: str = "gemini-2.5-flash"

# The embedding model used to convert text chunks and queries into vectors.
# "text-embedding-004" is Google's latest general-purpose embedding model.
GEMINI_EMBEDDING_MODEL = "text-embedding-004"

# ---------------------------------------------------------------------------
# Text Chunking Parameters
# ---------------------------------------------------------------------------

# Maximum number of characters per chunk.
# Smaller chunks → more precise retrieval but less context per chunk.
# Larger chunks → more context but noisier retrieval.
# 1000 characters ≈ ~200 tokens, a good balance for most PDFs.
CHUNK_SIZE: int = 1000

# Number of characters that overlap between consecutive chunks.
# Overlap ensures that sentences split across chunk boundaries are still
# captured in at least one chunk, preventing lost context.
CHUNK_OVERLAP: int = 200


# ---------------------------------------------------------------------------
# FAISS Retrieval Parameters
# ---------------------------------------------------------------------------

# Number of top similar chunks to retrieve for each user query.
# Higher K → more context for the LLM but also more noise.
# 4–6 is the industry-standard sweet spot.
TOP_K_RESULTS: int = 5


# ---------------------------------------------------------------------------
# LLM Generation Parameters
# ---------------------------------------------------------------------------

# Controls randomness in the LLM response.
# 0.0 = fully deterministic (best for factual Q&A).
# 1.0 = very creative/random.
LLM_TEMPERATURE: float = 0.2

# Maximum number of tokens the LLM can generate in a single response.
# 1024 tokens ≈ ~750 words — enough for detailed answers.
LLM_MAX_OUTPUT_TOKENS: int = 1024


# ---------------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------------

# Allowed MIME types / file extensions for uploaded files.
ALLOWED_EXTENSIONS: list[str] = ["pdf"]

# Maximum upload size in megabytes (enforced in the UI layer).
MAX_UPLOAD_SIZE_MB: int = 50


# ---------------------------------------------------------------------------
# UI / Streamlit
# ---------------------------------------------------------------------------

# Name displayed in the browser tab and app header.
APP_TITLE: str = "📄 PDF RAG Chatbot"

# Sub-heading shown below the title.
APP_SUBTITLE: str = "Ask questions about your uploaded PDF documents"

# Placeholder text shown inside the chat input box.
CHAT_INPUT_PLACEHOLDER: str = "Ask a question about your documents..."

# How many chat messages to display before adding a scroll container.
MAX_DISPLAY_MESSAGES: int = 50


# ---------------------------------------------------------------------------
# Prompt Engineering
# ---------------------------------------------------------------------------

# The system-level instruction injected into every Gemini request.
# This shapes the model's persona and constrains it to the retrieved context.
SYSTEM_PROMPT: str = """You are a helpful AI assistant that answers questions \
based strictly on the provided document context.

Guidelines:
- Answer ONLY using the information found in the context below.
- If the context does not contain enough information to answer the question, \
say "I couldn't find relevant information in the uploaded documents."
- Be concise, accurate, and well-structured.
- When helpful, use bullet points or numbered lists.
- Do NOT make up information or use outside knowledge.
"""