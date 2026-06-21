# =============================================================================
# src/llm_handler.py
# -----------------------------------------------------------------------------
# Handles all interactions with the Google Gemini generative model.
#
# SDK:     google-genai  (pip install google-genai)
# Import:  from google import genai
# Model:   gemini-2.5-flash
# Docs:    https://googleapis.github.io/python-genai/
#
# Responsibilities:
#   - Build structured RAG prompts from retrieved chunks + user query
#   - Call Gemini 2.5 Flash for answer generation
#   - Support both streaming and non-streaming response modes
#   - Maintain and format conversation history for multi-turn chat
#   - Handle API errors, safety blocks, and empty responses gracefully
#
# Migration note:
#   The old google-generativeai SDK used genai.configure() + GenerativeModel().
#   The new google-genai SDK uses genai.Client(api_key=...) and
#   client.models.generate_content(model=..., contents=..., config=...).
#   Generation parameters (temperature, system_instruction, safety_settings)
#   now live inside a single types.GenerateContentConfig object.
#
# RAG Prompt Strategy:
#   The prompt is structured in three parts:
#     1. SYSTEM  — Defines the assistant's persona and strict constraints
#     2. CONTEXT — The retrieved chunks injected as grounding evidence
#     3. QUESTION — The user's actual question
#
#   This structure ensures the model answers from the documents,
#   not from its parametric (training) knowledge.
# =============================================================================

import logging
from typing import List, Dict, Any, Optional, Generator

from google import genai
from google.genai import types

from config.settings import (
    GEMINI_CHAT_MODEL,
    GEMINI_API_KEY,
    LLM_TEMPERATURE,
    LLM_MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT,
)
from utils.helpers import format_source_chunks, validate_api_key

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation Configuration
# ---------------------------------------------------------------------------

def _build_safety_settings() -> List[types.SafetySetting]:
    """
    Configure Gemini's content safety filters.

    Gemini has built-in safety filters for harmful content categories.
    For a document Q&A application, we set them to BLOCK_ONLY_HIGH
    to avoid over-blocking legitimate document content (e.g. medical,
    legal, or security documents that may trigger lower thresholds).

    Returns:
        List of SafetySetting objects accepted by GenerateContentConfig.
    """
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        ),
    ]


def _build_generation_config() -> types.GenerateContentConfig:
    """
    Build the Gemini generation configuration object.

    GenerateContentConfig controls how the model samples its output and
    also carries the system instruction and safety settings in the new SDK.

    - temperature        : Randomness (0 = deterministic, 1 = creative)
    - max_output_tokens   : Hard cap on response length
    - top_p               : Nucleus sampling threshold (0.95 = industry default)
    - top_k               : Limits vocabulary at each step (40 = safe default)
    - system_instruction  : The assistant's persona / constraints
    - safety_settings     : Content safety thresholds

    Returns:
        A configured GenerateContentConfig object for use in
        client.models.generate_content() / generate_content_stream().
    """
    return types.GenerateContentConfig(
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
        top_p=0.95,
        top_k=40,
        system_instruction=SYSTEM_PROMPT,
        safety_settings=_build_safety_settings(),
    )


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_rag_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Construct the full RAG prompt to send to Gemini.

    The prompt has three sections (system instruction is passed separately
    via GenerateContentConfig, not embedded in this string):
    ┌─────────────────────────────────────────┐
    │ 1. CONVERSATION HISTORY (optional)      │
    │    (last N turns for multi-turn context)│
    ├─────────────────────────────────────────┤
    │ 2. RETRIEVED CONTEXT                    │
    │    (chunks from FAISS search)           │
    ├─────────────────────────────────────────┤
    │ 3. CURRENT QUESTION                     │
    │    (user's query)                       │
    └─────────────────────────────────────────┘

    Args:
        query:            The user's current question.
        retrieved_chunks: List of chunk dicts from VectorStore.search().
        chat_history:     Optional list of previous {role, content} dicts
                          for multi-turn conversation context.

    Returns:
        A single formatted string ready to pass as `contents` to Gemini.
    """
    sections = []

    # --- Section 1: Conversation History (last 3 turns max) ---
    # Including recent history gives the model context for follow-up questions
    # like "Can you elaborate on that?" or "What about the second point?"
    # We limit to 3 turns (6 messages) to avoid bloating the prompt.
    if chat_history:
        recent_history = chat_history[-6:]  # Last 3 user+assistant pairs
        if recent_history:
            history_lines = ["--- CONVERSATION HISTORY ---"]
            for msg in recent_history:
                role    = msg.get("role", "user").capitalize()
                content = msg.get("content", "").strip()
                history_lines.append(f"{role}: {content}")
            sections.append("\n".join(history_lines))

    # --- Section 2: Retrieved Context ---
    context_str = format_source_chunks(retrieved_chunks)
    sections.append(
        f"--- DOCUMENT CONTEXT ---\n{context_str}"
    )

    # --- Section 3: Current Question ---
    sections.append(
        f"--- QUESTION ---\n{query.strip()}\n\n"
        f"Answer based strictly on the document context above:"
    )

    # Join all sections with double newlines for clear separation.
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# LLM Handler Class
# ---------------------------------------------------------------------------

class LLMHandler:
    """
    Manages Gemini model initialisation and answer generation.

    Wraps the Gemini generative model with RAG-specific prompt construction,
    streaming support, error handling, and response validation.

    Usage:
        handler = LLMHandler()
        if handler.is_ready():
            answer = handler.generate_answer(query, chunks, history)
    """

    def __init__(self):
        """
        Initialise the Google GenAI client.

        Validates the API key and creates a genai.Client instance plus a
        reusable GenerateContentConfig. Sets self._ready = False if
        initialisation fails.
        """
        self._client: Optional[genai.Client] = None
        self._config: Optional[types.GenerateContentConfig] = None
        self._ready: bool = False
        self._initialise_client()

    def _initialise_client(self) -> None:
        """
        Private: Validate the API key and instantiate the GenAI client.
        """
        if not validate_api_key(GEMINI_API_KEY):
            logger.error(
                "LLMHandler: Invalid or missing GEMINI_API_KEY. "
                "Please check your .env file."
            )
            return

        try:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._config = _build_generation_config()
            self._ready  = True
            logger.info(
                f"LLMHandler initialised with model '{GEMINI_CHAT_MODEL}'."
            )

        except Exception as e:
            logger.error(f"LLMHandler: Failed to initialise Gemini client: {e}")
            self._ready = False

    def is_ready(self) -> bool:
        """Return True if the client is initialised and ready to use."""
        return self._ready and self._client is not None

    # -----------------------------------------------------------------------
    # Non-Streaming Answer Generation
    # -----------------------------------------------------------------------

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a complete answer for a user query using retrieved context.

        This non-streaming version waits for the full response before
        returning. Use generate_answer_stream() for a better UX where
        the user sees text appearing progressively.

        Args:
            query:            The user's question.
            retrieved_chunks: Top-K chunks from FAISS search.
            chat_history:     Previous conversation turns for context.

        Returns:
            The model's answer as a plain string.
            Returns a user-friendly error message if generation fails.
        """
        if not self.is_ready():
            return (
                "⚠️ The AI model is not initialised. "
                "Please check your Gemini API key in the .env file."
            )

        if not retrieved_chunks:
            return (
                "I couldn't find relevant information in the uploaded "
                "documents to answer your question. Please try rephrasing "
                "or upload documents that contain the relevant information."
            )

        # Build the complete RAG prompt.
        prompt = build_rag_prompt(query, retrieved_chunks, chat_history)

        logger.debug(
            f"Sending prompt to Gemini ({len(prompt)} chars, "
            f"{len(retrieved_chunks)} chunks)..."
        )

        try:
            response = self._client.models.generate_content(
                model=GEMINI_CHAT_MODEL,
                contents=prompt,
                config=self._config,
            )
            answer = self._extract_text(response)

            logger.info(f"Answer generated: {len(answer)} chars.")
            return answer

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return self._handle_generation_error(e)

    # -----------------------------------------------------------------------
    # Streaming Answer Generation
    # -----------------------------------------------------------------------

    def generate_answer_stream(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream the model's answer token-by-token as a Python generator.

        Used with Streamlit's st.write_stream() to display text progressively.
        The user sees words appearing in real time instead of waiting for
        the complete response — dramatically improves perceived performance.

        Args:
            query:            The user's question.
            retrieved_chunks: Top-K chunks from FAISS search.
            chat_history:     Previous conversation turns for context.

        Yields:
            String tokens/chunks as they arrive from the API.
            Yields a single error message string if generation fails.

        Usage in Streamlit:
            with st.chat_message("assistant"):
                response = st.write_stream(
                    handler.generate_answer_stream(query, chunks, history)
                )
        """
        if not self.is_ready():
            yield (
                "⚠️ The AI model is not initialised. "
                "Please check your Gemini API key."
            )
            return

        if not retrieved_chunks:
            yield (
                "I couldn't find relevant information in the uploaded "
                "documents to answer your question."
            )
            return

        prompt = build_rag_prompt(query, retrieved_chunks, chat_history)

        try:
            # generate_content_stream() returns an iterator of partial
            # GenerateContentResponse objects (server-sent events under the hood).
            stream = self._client.models.generate_content_stream(
                model=GEMINI_CHAT_MODEL,
                contents=prompt,
                config=self._config,
            )

            for chunk in stream:
                token = self._extract_chunk_text(chunk)
                if token:
                    yield token

        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            yield self._handle_generation_error(e)

    # -----------------------------------------------------------------------
    # Response Extraction Helpers (Private)
    # -----------------------------------------------------------------------

    def _extract_text(self, response: Any) -> str:
        """
        Safely extract text from a Gemini non-streaming response object.

        Handles edge cases:
        - Safety blocks (response.prompt_feedback)
        - Empty candidates list
        - Missing text in parts

        Args:
            response: The GenerateContentResponse from
                      client.models.generate_content().

        Returns:
            Extracted text string, or a user-friendly fallback message.
        """
        try:
            # Check for prompt-level safety blocks first.
            feedback = getattr(response, "prompt_feedback", None)
            if feedback is not None:
                block_reason = getattr(feedback, "block_reason", None)
                if block_reason:
                    logger.warning(
                        f"Prompt blocked by safety filter: {block_reason}"
                    )
                    return (
                        "⚠️ Your query was flagged by the content safety filter. "
                        "Please rephrase your question."
                    )

            # Access the primary candidate's text.
            candidates = getattr(response, "candidates", None)
            if candidates:
                candidate = candidates[0]

                # Check for candidate-level finish reasons.
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason and str(finish_reason) not in ("1", "STOP"):
                    logger.warning(f"Unexpected finish reason: {finish_reason}")

                # Extract text from the content parts.
                content = getattr(candidate, "content", None)
                if content and getattr(content, "parts", None):
                    return "".join(
                        part.text
                        for part in content.parts
                        if hasattr(part, "text") and part.text
                    )

            # Fallback: try the .text convenience property.
            text_value = getattr(response, "text", None)
            if text_value:
                return text_value

            logger.warning("Gemini response contained no text content.")
            return (
                "I was unable to generate a response. "
                "Please try asking your question differently."
            )

        except Exception as e:
            logger.error(f"Error extracting response text: {e}")
            return "An error occurred while processing the response."

    def _extract_chunk_text(self, chunk: Any) -> str:
        """
        Safely extract text from a single streaming response chunk.

        Streaming chunks can be partial and may not always have text
        (e.g. the final chunk may only contain finish_reason metadata).

        Args:
            chunk: A partial GenerateContentResponse from streaming.

        Returns:
            Text string from the chunk, or empty string if none.
        """
        try:
            text_value = getattr(chunk, "text", None)
            if text_value:
                return text_value

            candidates = getattr(chunk, "candidates", None)
            if candidates:
                content = getattr(candidates[0], "content", None)
                if content and getattr(content, "parts", None):
                    return "".join(
                        p.text for p in content.parts
                        if hasattr(p, "text") and p.text
                    )
        except Exception:
            # Silently skip malformed streaming chunks.
            pass
        return ""

    def _handle_generation_error(self, error: Exception) -> str:
        """
        Convert a Gemini API exception into a user-friendly error message.

        Different exception types need different messages:
        - Auth errors    → tell user to check API key
        - Quota errors   → tell user they've hit rate limits
        - Network errors → suggest retrying
        - Other errors   → generic message

        Args:
            error: The exception caught during generate_content().

        Returns:
            A user-friendly error string to display in the chat.
        """
        error_str = str(error).lower()

        if "api_key" in error_str or "authentication" in error_str or "401" in error_str:
            return (
                "⚠️ **API Key Error**: Your Gemini API key appears to be "
                "invalid or expired. Please check your `.env` file and "
                "ensure `GEMINI_API_KEY` is set correctly."
            )

        if "quota" in error_str or "429" in error_str or "rate" in error_str:
            return (
                "⚠️ **Rate Limit**: You've exceeded the Gemini API rate limit. "
                "Please wait a moment and try again. Consider upgrading your "
                "API plan if this happens frequently."
            )

        if "timeout" in error_str or "deadline" in error_str:
            return (
                "⚠️ **Timeout**: The request to Gemini timed out. "
                "Please try again — this is usually a temporary issue."
            )

        if "network" in error_str or "connection" in error_str:
            return (
                "⚠️ **Network Error**: Could not connect to the Gemini API. "
                "Please check your internet connection and try again."
            )

        if "404" in error_str or "not found" in error_str:
            return (
                "⚠️ **Model Error**: The model "
                f"'{GEMINI_CHAT_MODEL}' could not be found. "
                "Please verify the model name is correct and available "
                "for your API key."
            )

        # Generic fallback for unexpected errors.
        logger.error(f"Unhandled Gemini error: {error}")
        return (
            "⚠️ An unexpected error occurred while generating the response. "
            "Please try again or rephrase your question."
        )

    def __repr__(self) -> str:
        status = "ready" if self._ready else "not ready"
        return f"LLMHandler(model='{GEMINI_CHAT_MODEL}', status={status})"