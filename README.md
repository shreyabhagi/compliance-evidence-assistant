# Compliance Evidence Assistant

A Streamlit app that answers questions over uploaded Word (.docx) compliance, audit, policy, regulatory, and clinical-safety documents. Answers are generated only from the uploaded documents and every claim is cited back to its source file.

All AI processing runs locally through [Ollama](https://ollama.com) — no external API calls, no per-query cost.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- The two Ollama models this app uses, pulled locally:

  ```
  ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
  ollama pull phi3:3.8b
  ```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run Microeai_RAG.py
```

This opens the app in your browser. Upload one or more `.docx` files in the sidebar, then ask questions in the chat box.

## Configuration (optional)

The app works out of the box with sensible defaults. To override any of them, set environment variables before running — see the top of `Microeai_RAG.py` for the full list (chunk size, retrieval settings, LLM temperature/context window, etc.).

## Notes

- Uploaded documents are indexed into a local ChromaDB store (`chroma_db/`, created on first run) — this is not committed to the repo.
- Each browser session is isolated: one session never sees another session's uploaded documents.
