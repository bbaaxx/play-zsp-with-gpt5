# WhatsApp RAG (ES)

Prototipo minimalista RAG para conversaciones de WhatsApp en español.

## Requisitos
- Python 3.11+
- macOS/Linux
- Token GitHub (`GITHUB_TOKEN`) con permiso `models:read`

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuración
Crea `.env` basado en `.env.example`:
```
GITHUB_TOKEN=ghp_xxx
GH_MODELS_BASE_URL=https://models.github.ai/inference
CHAT_MODEL=openai/gpt-4o
EMBEDDING_MODEL=openai/text-embedding-3-small
USE_LOCAL_EMBEDDINGS=0
```
Si tu plan no permite embeddings remotos, usa `USE_LOCAL_EMBEDDINGS=1` (requiere `sentence-transformers`).

## Ejecutar
```bash
python app.py
```
Abre `http://127.0.0.1:7860`.

## Flujo
1. Carga un TXT de WhatsApp.
2. Indexa (parser → chunking → embeddings → FAISS en memoria).
3. Chatea en español. Se envía al LLM solo el contexto recuperado.

## Archivo de ejemplo
`data/sample_whatsapp.txt` contiene un chat de prueba.

## Notas
- Similaridad: FAISS `IndexFlatIP` con normalización L2 (cosine).
- LLM: GitHub Models vía SDK OpenAI.
- Privacidad: el archivo se procesa localmente; solo se envían fragmentos recuperados.

