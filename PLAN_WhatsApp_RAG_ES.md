## Plan de Ejecución — WhatsApp RAG (ES)

### 0) Requisitos previos
- Python 3.11+
- macOS (probado), zsh
- GitHub Personal Access Token (PAT) con `models:read` → guardar como `GITHUB_TOKEN`.

### 1) Estructura mínima
- `app.py` — Gradio UI
- `rag/core.py` — pipeline RAG
- `rag/embeddings.py` — proveedor de embeddings (GH Models o local)
- `rag/vector_store.py` — FAISS helpers
- `requirements.txt`
- `.env.example`
- `data/sample_whatsapp.txt`

### 2) Dependencias
Crear y activar venv, luego instalar:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
`requirements.txt` recomendado:
```
gradio==4.44.0
faiss-cpu==1.8.0.post1
numpy==1.26.4
pandas==2.2.2
python-dotenv==1.0.1
openai==1.40.0
regex==2024.5.15
rapidfuzz==3.9.3
# fallback local embeddings
sentence-transformers==3.0.1
```

### 3) Configuración
Crear `.env` a partir de `.env.example`:
```
GITHUB_TOKEN=ghp_xxx
GH_MODELS_BASE_URL=https://models.github.ai/inference
CHAT_MODEL=openai/gpt-4o
EMBEDDING_MODEL=openai/text-embedding-3-small
USE_LOCAL_EMBEDDINGS=0
```
Notas:
- Si embeddings vía GitHub Models no funciona en tu plan, poner `USE_LOCAL_EMBEDDINGS=1`.
- Para modelos alternativos (coste/limitaciones), cambiar `CHAT_MODEL` (p. ej., `mistralai/mistral-small` si está en catálogo).

### 4) Implementación — detalles clave
- Parsing WhatsApp:
  - Detectar variantes de formato: "[dd/mm/yyyy, hh:mm] Nombre: texto" y "dd/mm/yy, hh:mm - Nombre: texto".
  - Normalizar timestamps a ISO8601 (UTC) y conservar `sender`, `text`, `line_no`.
- Chunking:
  - Ventanas de 20–40 mensajes o ~700 tokens; solapamiento de 10.
- Embeddings:
  - Por defecto, `openai/text-embedding-3-small` vía API OpenAI-compatible. Si no, `sentence-transformers` con `intfloat/multilingual-e5-small`.
  - Normalizar vectores (L2) antes de FAISS.
- FAISS:
  - `IndexFlatIP` + normalización = cosine.
  - Guardar metadatos en arrays paralelos (Python) o estructura en memoria; opcional pickle a disco.
- Recuperación:
  - top_k=5; MMR opcional (λ=0.5) para diversidad.
- LLM (GitHub Models):
  - SDK `openai` con `base_url`=`$GH_MODELS_BASE_URL` y `api_key`=`$GITHUB_TOKEN`.
  - Endpoint: `POST /chat/completions` (el SDK lo maneja).
  - Modelo: `openai/gpt-4o` por defecto. `temperature=0.2`, `max_tokens=800`.
  - Prompts en español; citar remitente y fecha.
- UI (Gradio):
  - Componentes: `File`, `Chatbot`, `Dropdown` (modelo), `Slider` (top_k), `Buttons` (Indexar/Reindexar/Limpiar), estado.
  - Flujo: subir archivo → indexar → chatear.

### 5) Comandos de desarrollo
- Ejecutar app:
```
python app.py
```
- Acceso UI: `http://127.0.0.1:7860`.

### 6) Validación rápida (smoke test)
- Cargar `data/sample_whatsapp.txt`.
- Ver estado de indexado (nº mensajes/chunks > 0).
- Preguntar en español “¿Cuándo quedamos para cenar?” y verificar respuesta con citas `[Nombre — fecha hora]`.

### 7) Manejo de límites/errores
- Retries exponenciales (HTTP 429) y mensajes amigables en UI.
- Alternar a modelo más económico si se exceden límites.
- Si embeddings remotos fallan → activar fallback local automáticamente si `USE_LOCAL_EMBEDDINGS=1`.

### 8) Extensiones opcionales
- Persistencia de índice a disco (`.faiss` + metadatos `.json`).
- Filtros por `sender` y rango de fechas en UI.
- Re-ranking con MMR.

### 9) Seguridad/privacidad
- Procesar archivo localmente; enviar solo fragmentos recuperados al LLM.
- Avisar sobre PII; botón “borrar sesión” que limpia índice en memoria.

### 10) Snippets críticos
- Cliente LLM (Python):
```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["GITHUB_TOKEN"],
    base_url=os.environ.get("GH_MODELS_BASE_URL", "https://models.github.ai/inference")
)

resp = client.chat.completions.create(
    model=os.environ.get("CHAT_MODEL", "openai/gpt-4o"),
    messages=[
        {"role": "system", "content": "Eres un asistente en español..."},
        {"role": "user", "content": "Hola"}
    ],
    temperature=0.2,
    max_tokens=200
)
print(resp.choices[0].message.content)
```
- Embeddings (fallback local):
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("intfloat/multilingual-e5-small")

def embed(texts):
    X = model.encode(texts, normalize_embeddings=True)
    return np.asarray(X, dtype="float32")
```

### 11) Entrega
- Confirmar que `PRD_WhatsApp_RAG_ES.md` y este plan están en el raíz.
- Incluir `requirements.txt`, `.env.example`, `data/sample_whatsapp.txt`.
- Verificar arranque local sin errores.
