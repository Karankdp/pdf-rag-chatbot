# 📄 Chat with your PDFs — RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDFs and ask questions grounded in their content — with cited sources and relevance scoring.

**Live demo:** _add your Streamlit Cloud URL here after deploying_

## Architecture

```
PDF Upload → Text Extraction (pypdf) → Chunking (sliding window, sentence-aware)
→ Embeddings (gemini-embedding-001) → Vector Store (ChromaDB)
→ Retrieval (top-k similarity search) → Generation (gemini-3.6-flash)
```

## Features

- Multi-PDF upload and processing, entirely in-browser
- Source-cited answers with visual relevance scoring per retrieved chunk
- Example question chips for a fast first interaction
- Session stats (chunks indexed, questions asked)
- Downloadable chat history
- Manual evaluation harness (keyword-hit-rate scoring against a labeled Q&A set) — scored 100% on a 6-question test set over a sample document

## Tech stack

| Layer | Choice |
|---|---|
| LLM (generation) | Google Gemini (`gemini-3.6-flash`) |
| Embeddings | Google Gemini (`gemini-embedding-001`) |
| Vector store | ChromaDB |
| PDF parsing | pypdf |
| Frontend | Streamlit |

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

You'll be prompted for a Gemini API key in the sidebar (get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account
3. Create a new app pointing at `streamlit_app.py`
4. In **Advanced settings → Secrets**, add:
   ```
   GEMINI_API_KEY = "your-key-here"
   ```
5. Deploy

## Evaluation

A manual evaluation harness checks retrieval + generation quality against a hand-labeled set of question/expected-keyword pairs, scoring the keyword hit rate of generated answers. See `eval.py` for the harness — swap in your own document's Q&A pairs to re-run it.

## Possible extensions

- Persistent vector storage across sessions (currently in-memory per session)
