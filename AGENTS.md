# AGENTS.md

## Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Dev Server
```bash
python3 app.py
```

### Test/Lint/Build
No automated tests or linting configured. Build not applicable for Python script.

## Tech Stack
- **Language**: Python 3.11+
- **Framework**: Gradio for UI
- **AI/ML**: OpenAI SDK with GitHub Models, sentence-transformers for embeddings
- **Vector DB**: FAISS (in-memory)
- **Data**: WhatsApp chat text parsing and chunking

## Architecture
- `app.py` - Gradio web UI and main application
- `rag/` - Core RAG pipeline modules (embeddings, vector store, processing)
- `data/` - Sample WhatsApp chat files

## Code Style
- Type hints using `from __future__ import annotations`
- Snake_case for functions/variables
- Environment variables via `.env` file
- Modular design with separation of concerns