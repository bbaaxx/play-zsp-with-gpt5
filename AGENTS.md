# WhatsApp RAG (ES) - Development Guide

## Setup Commands
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Development Commands
- **Build**: No separate build step required (Python interpreted)
- **Lint**: `ruff check .` (included with Gradio)
- **Test**: No test framework configured
- **Dev Server**: `python app.py` (runs on http://127.0.0.1:7860)

## Tech Stack
- **Backend**: Python 3.11+, Gradio web framework
- **RAG**: Custom implementation with FAISS vector store, sentence-transformers
- **LLM**: OpenAI API via GitHub Models
- **Dependencies**: NumPy, pandas, httpx for data processing

## Architecture
- `app.py`: Gradio web UI and main application logic
- `rag/`: Core RAG implementation (parser, embeddings, vector store)
- `data/`: Sample WhatsApp export files

## Code Style
- Uses type hints and dataclasses
- Spanish language support throughout
- Environment-based configuration via `.env`
- Follows Python PEP 8 conventions
