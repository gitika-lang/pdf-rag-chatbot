# 📄 PDF RAG Chatbot

A production-quality **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload PDF documents and ask questions about their content. Answers are grounded in the actual text of your documents, with source citations shown for every response.

---

## ✨ Features

- Upload one or more PDF files and query them simultaneously
- Full RAG pipeline: extract → chunk → embed → index → retrieve → generate
- Source citations with page numbers and similarity scores
- Conversational chat interface with history
- Sidebar status dashboard (indexed files, chunk count, model info)
- Robust error handling throughout

---

## 🏗️ Architecture

```
pdf-rag-chatbot/
├── app.py                  # Streamlit entry point (UI + orchestration)
├── config/
│   └── settings.py         # Single source of all configuration
├── src/
│   ├── pdf_processor.py    # PyMuPDF text extraction
│   ├── text_chunker.py     # LangChain RecursiveCharacterTextSplitter
│   ├── embeddings.py       # Google text-embedding-004 wrapper
│   ├── vector_store.py     # FAISS index build + search
│   ├── llm_handler.py      # Gemini 1.5 Flash answer generation
│   └── chat_manager.py     # Session state & conversation history
├── utils/
│   └── helpers.py          # Shared utility functions
├── requirements.txt
├── .env.example
└── README.md
```

### RAG Workflow

```
Upload PDF
    │
    ▼
PDFProcessor.extract_text()     ← PyMuPDF / fitz
    │
    ▼
TextChunker.chunk_pages()       ← LangChain RecursiveCharacterTextSplitter
    │
    ▼
EmbeddingGenerator.embed_texts() ← Google text-embedding-004
    │
    ▼
VectorStore.add_chunks()        ← FAISS IndexFlatIP
    │
  (query time)
    │
    ▼
EmbeddingGenerator.embed_query()
    │
    ▼
VectorStore.search()            ← Top-K nearest neighbours
    │
    ▼
LLMHandler.generate_answer()    ← Gemini 1.5 Flash
    │
    ▼
Streamlit chat UI + citations
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-org/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Open .env and set GEMINI_API_KEY=<your key>
```

Get a free Gemini API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## ⚙️ Configuration

All tuneable parameters live in `config/settings.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Google Gemini API key |
| `GEMINI_MODEL_NAME` | `gemini-1.5-flash` | Gemini model |
| `EMBEDDING_MODEL_NAME` | `models/text-embedding-004` | Embedding model |
| `TOP_K_CHUNKS` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `LLM_TEMPERATURE` | `0.2` | LLM response temperature |
| `MAX_OUTPUT_TOKENS` | `2048` | Max tokens in LLM response |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max PDF upload size |

---

## 📦 Dependencies

```
streamlit
google-generativeai
faiss-cpu
pymupdf
langchain
langchain-text-splitters
python-dotenv
numpy
```

See `requirements.txt` for pinned versions.

---

## 🛠️ Development Notes

- **Adding a new LLM:** Implement the same interface as `LLMHandler` and swap it in `app.py`.
- **Persistent index:** `VectorStore` is in-memory per session. To persist across sessions, call `vs.save(path)` / `vs.load(path)` (implement in `vector_store.py`).
- **Multi-user deployments:** Move `VectorStore` and `ChatManager` state to a database or Redis instead of Streamlit session state.
- **Scanned PDFs:** Add an OCR step in `PDFProcessor` (e.g., `pytesseract`) before chunking.

---

## 📜 License

MIT © Your Organisation
