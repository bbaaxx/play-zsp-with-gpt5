# WhatsApp RAG (ES)

Una aplicación de Recuperación Aumentada por Generación (RAG) especializada en analizar conversaciones exportadas de WhatsApp en español. Combina embeddings semánticos con modelos de lenguaje grandes (LLMs) para responder preguntas sobre el contexto de las conversaciones.

## Características

- **Parser Robusto**: Maneja múltiples formatos de exportación de WhatsApp (24h, 12h con am/pm, con/sin segundos)
- **Embeddings Multilingües**: Soporte para embeddings remotos (OpenAI) y locales (sentence-transformers)
- **Múltiples Proveedores LLM**: Soporte para GitHub Models y LM Studio con fallback automático
- **Búsqueda Avanzada**: Implementa Maximum Marginal Relevance (MMR) para diversificar resultados
- **Filtrado Granular**: Filtros por remitente, rango de fechas y relevancia
- **Interfaz Web**: UI intuitiva con Gradio para cargar archivos y hacer consultas
- **Respuestas Contextualizadas**: Cita fragmentos específicos con remitente y fecha

## ✨ Nuevo: Módulo de Análisis de Datos

El sistema ahora incluye un módulo avanzado de análisis de datos (`rag.analysis.ChatDataFrame`) que permite análisis estructurado de mensajes de WhatsApp usando pandas:

- **📊 Análisis temporal**: actividad por día/hora, rangos de fechas
- **🔍 Filtrado avanzado**: por autor, contenido, patrones regex
- **📈 Estadísticas**: conteo de mensajes, longitud promedio, autores más activos
- **🔎 Búsqueda contextual**: palabras clave con mensajes de contexto
- **💾 Exportación**: CSV para análisis externos
- **⚡ Optimización**: tipos de datos eficientes para consultas rápidas

```python
from rag import ChatDataFrame

# Cargar y analizar mensajes
analyzer = ChatDataFrame()
analyzer.load_from_file("mi_chat.txt")

# Filtrar por autor y contenido
juan_msgs = analyzer.filter_by_author("Juan")
time_msgs = analyzer.filter_by_content("hora")

# Estadísticas del chat
stats = analyzer.get_message_stats()
print(f"Total: {stats['total_messages']} mensajes")
```

## Requisitos

- Python 3.11+
- Token de GitHub (para GitHub Models) o servidor LM Studio local
- Archivos TXT exportados de WhatsApp

## Instalación

1. Clona el repositorio:
```bash
git clone <repo-url>
cd whatsapp-rag-es
```

2. Crea un entorno virtual:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# o
.venv\Scripts\activate.bat  # Windows
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Configura las variables de entorno:
```bash
cp .env.example .env
# Edita .env con tu configuración
```

## Configuración de LLM

### Opción 1: GitHub Models (Remoto)
```env
GITHUB_TOKEN=tu_token_aqui
CHAT_MODEL=openai/gpt-4o
```

### Opción 2: LM Studio (Local)

**Solo Chat:**
```env
LMSTUDIO_ENABLED=1
LMSTUDIO_CHAT_MODEL=llama-3.2-3b-instruct
```

**Chat + Embeddings:**
```env
LMSTUDIO_ENABLED=1
LMSTUDIO_CHAT_MODEL=llama-3.2-3b-instruct
LMSTUDIO_EMBEDDINGS_ENABLED=1
LMSTUDIO_EMBEDDING_MODEL=nomic-embed-text-v1
```

### Configuración Híbrida
Ambos proveedores pueden estar habilitados simultáneamente. El sistema intentará usar LM Studio primero (si está habilitado) y fallback a GitHub Models automáticamente.

## Tests
```bash
source .venv/bin/activate
PYTHONPATH=. pytest
```

## Uso

1. Inicia la aplicación:
```bash
python app.py
```

2. Abre http://127.0.0.1:7860 en tu navegador

3. Carga un archivo TXT exportado de WhatsApp

4. Usa el botón "Estado LLM" para verificar qué proveedores están disponibles

5. Haz preguntas sobre la conversación en español

## Exportar Conversaciones de WhatsApp

### Android:
1. Abre la conversación en WhatsApp
2. Toca los tres puntos → "Más" → "Exportar chat"
3. Selecciona "Sin multimedia"

### iOS:
1. Abre la conversación en WhatsApp
2. Toca el nombre del contacto/grupo → "Exportar chat"
3. Selecciona "Sin multimedia"

## Configuración

### Variables de Entorno

**GitHub Models:**
- `GITHUB_TOKEN`: Token para GitHub Models
- `GH_MODELS_BASE_URL`: URL base de GitHub Models
- `CHAT_MODEL`: Modelo LLM a usar (default: openai/gpt-4o)

**LM Studio:**
- `LMSTUDIO_ENABLED`: Habilitar LM Studio para chat (0/1)
- `LMSTUDIO_HOST`: Host del servidor LM Studio (default: localhost)
- `LMSTUDIO_PORT`: Puerto del servidor LM Studio (default: 1234)
- `LMSTUDIO_CHAT_MODEL`: Modelo de chat en LM Studio (ej: llama-3.2-3b-instruct)
- `LMSTUDIO_EMBEDDINGS_ENABLED`: Habilitar embeddings LM Studio (0/1)
- `LMSTUDIO_EMBEDDING_MODEL`: Modelo de embeddings en LM Studio (ej: nomic-embed-text-v1)
- `LMSTUDIO_TIMEOUT`: Timeout para requests (default: 60.0s)

**Embeddings:**
- `EMBEDDING_MODEL`: Modelo de embeddings (default: openai/text-embedding-3-small)
- `USE_LOCAL_EMBEDDINGS`: Usar embeddings locales (0/1)

**Otros:**
- `GRADIO_ANALYTICS_ENABLED`: Habilitar analytics de Gradio (0/1)

### Parámetros de Búsqueda

- **Top-k**: Número de fragmentos a recuperar (1-10)
- **MMR**: Activar Maximum Marginal Relevance para diversidad
- **λ (Lambda)**: Balance relevancia/diversidad en MMR (0.0-1.0)
- **fetch_k**: Candidatos iniciales para MMR (5-50)

## Arquitectura

```
app.py              # Interfaz web Gradio
rag/
├── core.py         # Pipeline RAG principal
├── embeddings.py   # Proveedores de embeddings
├── llm_providers.py # Proveedores LLM (nuevo)
└── vector_store.py # Almacén vectorial FAISS
data/               # Archivos de ejemplo
```

### Proveedores LLM

El sistema incluye una arquitectura modular para múltiples proveedores LLM:

- **LMStudioProvider**: Se conecta a servidor LM Studio local via HTTP
- **GitHubModelsProvider**: Usa la API de GitHub Models
- **LLMManager**: Maneja múltiples proveedores con fallback automático

## Desarrollo

### Comandos

- **Lint**: `ruff check .`
- **Test**: `python test_integration.py`
- **Dev**: `python app.py`

### Flujo de Datos

1. **Parsing**: Extrae mensajes del archivo TXT de WhatsApp
2. **Chunking**: Agrupa mensajes en ventanas deslizantes
3. **Embedding**: Convierte chunks a vectores semánticos
4. **Indexado**: Almacena en índice FAISS para búsqueda rápida
5. **Recuperación**: Busca chunks relevantes por similitud/MMR
6. **Generación**: Combina contexto con LLM para respuesta final

### Configuración de LM Studio

1. Descarga e instala [LM Studio](https://lmstudio.ai/)
2. Carga los modelos que necesites:
   - **Chat Model**: Llama-3.2-3b-instruct, Mistral-7B-Instruct, etc.
   - **Embedding Model** (opcional): nomic-embed-text-v1, bge-large-en-v1.5, etc.
3. Inicia el servidor local en el puerto 1234
4. Configura las variables de entorno apropiadas:
   - `LMSTUDIO_CHAT_MODEL`: debe coincidir con el modelo de chat cargado
   - `LMSTUDIO_EMBEDDING_MODEL`: debe coincidir con el modelo de embeddings cargado
5. Verifica la conexión con el botón "Estado LLM"

**Nota**: Puedes usar diferentes modelos para chat y embeddings. Es común usar un modelo pequeño y rápido para embeddings (como nomic-embed-text-v1) y uno más grande para chat.

## Manejo de Errores

El sistema incluye manejo robusto de errores:

- **Timeout**: Configurable por proveedor LLM
- **Fallback**: Automático entre proveedores
- **Reconexión**: Detección automática de disponibilidad
- **Logging**: Detallado para diagnóstico

## Limitaciones

- Solo archivos TXT de WhatsApp (no JSON o otros formatos)
- LM Studio requiere servidor local ejecutándose
- Memoria limitada para conversaciones muy grandes
- No procesa multimedia (solo texto)

## Contribuir

1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## Archivo de ejemplo
`data/sample_whatsapp.txt` contiene un chat de prueba.

## Notas
- Similaridad: FAISS `IndexFlatIP` con normalización L2 (cosine).
- LLM: GitHub Models vía SDK OpenAI con fallback a LM Studio.
- Privacidad: el archivo se procesa localmente; solo se envían fragmentos recuperados.

## Soporte

Para preguntas y soporte, abre un issue en el repositorio.