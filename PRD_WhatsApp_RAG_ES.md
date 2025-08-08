## WhatsApp RAG (ES) — PRD y Especificación Técnica

### Objetivo
Construir un prototipo minimalista de RAG en español sobre exportes de chats de WhatsApp. Permite:
- Subir un archivo de conversación (export estándar de WhatsApp, sin medios).
- Indexar en memoria con FAISS y consultar en lenguaje natural en español.
- Recuperar fragmentos relevantes y generar respuestas citando mensajes y participantes.
- UI simple con Gradio.
- Inferencia LLM vía GitHub Models.

### Alcance (MVP)
- Ingesta de 1 archivo por sesión (TXT de WhatsApp; opcional: JSON propio).
- Pipeline RAG: parsing → normalización → chunking → embeddings → FAISS (in-memory) → recuperación → prompts en español → respuesta del LLM.
- UI: carga de archivo, estado de indexado, chat, reindexar, limpiar historial.
- Todo en proceso local; sin persistencia obligatoria (opcional guardar índice en disco).

### No alcance (MVP)
- Multi-usuario, auth, persistencia robusta, analíticas avanzadas.
- Moderación de contenido/PII automática.
- Streaming multimedia o análisis de adjuntos.

### Historias de usuario (resumidas)
- Como usuaria, subo un TXT de WhatsApp y pregunto “¿Cuándo quedamos para cenar?” y obtengo respuesta con citas y fechas.
- Como usuaria, filtro por participante “Mamá” y consulto “¿Qué recetas recomendó?”
- Como usuaria, reindexo un nuevo archivo y continuo chateando.

### Requisitos no funcionales
- Idioma: español completo (UI, prompts, respuestas).
- Arranque local en <5 min; sin dependencias externas salvo GitHub Models.
- Manejo de archivos hasta ~10 MB (aprox. decenas de miles de mensajes) en máquina de desarrollo.

---

### Diseño del sistema (alto nivel)
- Python 3.11+
- Embeddings: OpenAI `text-embedding-3-small` (multilingüe) preferido. Intento vía GitHub Models; fallback local si embeddings no disponible.
- Vector DB: FAISS in-memory (`IndexFlatIP` con normalización para cosine).
- UI: Gradio (chat + panel lateral de estado/opciones).
- LLM chat: GitHub Models (compat. OpenAI) `openai/gpt-4o` (o variante de bajo costo si hay límites), endpoint base `https://models.github.ai/inference` usando `POST /chat/completions` con `GITHUB_TOKEN`.

Referencias:
- Chat completions: ver [GitHub Models Changelog](https://github.blog/changelog/2025-05-15-github-models-api-now-available/) y [Integrate AI models](https://docs.github.com/en/github-models/integrating-ai-models-into-your-development-workflow).
- Prototipado general: [Prototyping with AI models](https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models).

Nota embeddings: GitHub Models habla API tipo OpenAI; si `/embeddings` no está habilitado en tu cuenta, se usará un modelo local multilingüe (por ejemplo `intfloat/multilingual-e5-small`) solo para la fase de indexación. El LLM de generación seguirá siendo GitHub Models.

---

### Especificación de datos
- Entrada TXT (WhatsApp export estándar, ejemplo de línea):
  - "[12/10/2023, 21:15] Juan: ¿Salimos mañana?"
  - "12/10/23, 21:15 - Juan: ¿Salimos mañana?" (variantes por región/versión)
- Campos derivados por mensaje:
  - `chat_id` (hash del archivo o nombre de carga)
  - `timestamp` (UTC normalizado)
  - `sender` (string)
  - `text` (string)
  - `line_no` (int; para trazabilidad)
- Normalización:
  - Unificar codificación UTF-8, normalizar espacios, preservar tildes/ñ.
  - Remover líneas de sistema (“Mensajes y llamadas están cifrados...”).
  - Opción para excluir mensajes muy cortos/ruido.

### Chunking (orientado a diálogo)
- Estrategia por ventana de turnos: agrupar bloques de 20–40 mensajes o ~700 tokens con solapamiento ~10 mensajes.
- Metadatos por chunk:
  - `chunk_id`, `chat_id`, `start_ts`, `end_ts`, `participants_en_chunk` (set), `line_span`.
  - `text_window` (texto concatenado con formato “ts — sender: text”).

### Embeddings
- Preferido: `openai/text-embedding-3-small` (1536 dims, multilingüe).
- Si no disponible desde GitHub Models en tu plan, fallback local: `intfloat/multilingual-e5-small` (≈384–768 dims según versión).
- Pre-procesamiento: minificar emojis si necesario, mantener español.
- Almacén: FAISS `IndexFlatIP` con vectores L2-normalizados para simular cosine similarity.

### Recuperación
- Top-k = 5 (configurable), MMR opcional (λ=0.5).
- Filtros por metadatos (opcionales): por `sender` o rango de fecha.
- Re-ranking opcional (off por defecto para MVP).

### Generación (LLM vía GitHub Models)
- Base URL: `https://models.github.ai/inference` (OpenAI-compatible). Usar SDK OpenAI Python con `base_url` y `api_key=GITHUB_TOKEN`.
- Modelo por defecto: `openai/gpt-4o` (ajustable en `.env`).
- Parámetros: `temperature=0.2`, `max_tokens=800` (ajustables a límites de plan).
- Idioma: español forzado en system prompt.

Prompt template (sistema y usuario):
- System (español):
  """
  Eres un asistente en español. Responde de forma breve, correcta y sin inventar. Usa solo la evidencia del chat.
  Cuando cites, incluye remitente y fecha (p. ej., [Juan — 2023-10-12 21:15]). Si falta evidencia, dilo explícitamente.
  """
- User (con contexto):
  """
  Contexto recuperado (fragmentos del chat):
  {{contexto_con_formato}}

  Pregunta: {{pregunta_usuario}}

  Instrucciones: Responde en español, cita 1–3 fragmentos relevantes con referencia [remitente — fecha].
  """

Formato de contexto por fragmento:
- "[2023-10-12 21:15] Juan: ¿Salimos mañana?" (incluir hasta ~3–5 líneas por chunk top-k).

### UI (Gradio)
- Controles:
  - Carga de archivo (TXT; opcional JSON interno).
  - Botones: “Indexar”, “Reindexar”, “Limpiar chat”.
  - Select: modelo de LLM (si hay varias opciones), top-k.
  - Filtros opcionales: participante, fecha.
- Vistas:
  - Estado de indexación (nº mensajes, nº chunks, tiempo, tamaño FAISS).
  - Chat en español (historial y respuestas con citas).
- i18n: etiquetas y placeholders en español.

### Seguridad y privacidad
- Procesamiento local del archivo. Solo los fragmentos recuperados (no todo el chat) se envían al LLM.
- Aviso: contenido podría incluir PII; el usuario es responsable del archivo cargado.

### Límites / riesgos
- Rate limits de GitHub Models (ver [Rate limits](https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models#rate-limits)). Reintentos exponenciales y uso de modelo más económico si aplica.
- Archivos muy grandes: advertir y sugerir submuestreo o indexación incremental.
- Embeddings vía GitHub Models pueden no estar disponibles en todos los planes: activar fallback local.

### Métricas de aceptación (MVP)
- Indexa archivo TXT de ejemplo (<5 MB) sin errores; muestra conteos coherentes.
- Responde en español a 5 consultas de prueba con ≥1 cita correcta (remitente+fecha) en ≥80% de casos.
- UI operativa: carga, indexa, chatea, reindexa, limpia.

### Entregables de implementación (esperados)
- `app.py` (Gradio UI).
- `rag/core.py` (parser, chunking, embeddings, retrieval, prompts).
- `rag/embeddings.py` (proveedor GH Models o fallback local).
- `rag/vector_store.py` (FAISS helpers in-memory, normalización y MMR opcional).
- `requirements.txt`, `.env.example`, `README.md`.
- `data/sample_whatsapp.txt` (para pruebas locales).

### Configuración (env)
- `GITHUB_TOKEN` (PAT con permiso `models:read`).
- `GH_MODELS_BASE_URL` (por defecto `https://models.github.ai/inference`).
- `CHAT_MODEL` (por defecto `openai/gpt-4o`).
- `EMBEDDING_MODEL` (por defecto `openai/text-embedding-3-small`).
- `USE_LOCAL_EMBEDDINGS` (0/1).

---

### Criterios de “hecho”
- Documentos y scripts listos para ejecutar localmente con un archivo de ejemplo en español.
- UI en español y prompts en español; respuestas citadas.
- Sin dependencias de infraestructura externa salvo GitHub Models para el chat.
