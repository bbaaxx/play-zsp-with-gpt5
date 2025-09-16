# Configuration Examples

Este directorio contiene ejemplos de configuración para diferentes entornos de despliegue del sistema WhatsApp RAG.

## Archivos de Configuración

### Variables de Entorno

- **`development.env`**: Configuración para desarrollo local
- **`production.env`**: Configuración para producción

### Configuración RAG

- **`rag_config.yaml`**: Configuración completa del sistema RAG

## Uso de Configuraciones

### Variables de Entorno

```bash
# Desarrollo
cp examples/config/development.env .env
# Editar .env con tus valores

# Producción  
cp examples/config/production.env .env
# Configurar valores de producción
```

### Configuración YAML

```python
import yaml
from pathlib import Path

# Cargar configuración
config_path = Path("examples/config/rag_config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Usar en el código
embedding_model = config['embedding']['model_name']
chunk_size = config['text_processing']['chunk_size']
```

## Configuraciones por Entorno

### Desarrollo Local

```bash
# Variables mínimas necesarias
OPENAI_API_KEY=tu_github_token
GITHUB_TOKEN=tu_github_token
LOG_LEVEL=DEBUG
GRADIO_DEBUG=true
```

**Características:**
- Logs detallados para debugging
- Sin autenticación
- Límites relajados para pruebas
- Cache deshabilitado
- Auto-reload activado

### Staging/Testing

```bash
# Configuración intermedia
OPENAI_API_KEY=token_de_staging
LOG_LEVEL=INFO
GRADIO_AUTH_MESSAGE="Staging - Solo testing"
ENABLE_METRICS=true
```

**Características:**  
- Configuración similar a producción
- Métricas habilitadas
- Autenticación básica
- Logs estructurados

### Producción

```bash
# Configuración optimizada para producción
OPENAI_API_KEY=token_produccion
LOG_LEVEL=WARNING
GRADIO_USERNAME=admin
GRADIO_PASSWORD=contraseña_segura
ENABLE_METRICS=true
MAX_REQUESTS_PER_MINUTE=60
```

**Características:**
- Autenticación obligatoria
- Rate limiting estricto
- Logs optimizados
- Métricas y monitoreo
- Configuración de seguridad

## Configuración del Sistema RAG

### Modelos de Embedding

```yaml
# Para español optimizado
embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  
# Para mejor calidad (más lento)
embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Para velocidad (menor calidad)  
embedding:
  model_name: "sentence-transformers/paraphrase-MiniLM-L3-v2"
```

### Estrategias de Chunking

```yaml
# Documentos largos
text_processing:
  chunk_size: 1500
  chunk_overlap: 300
  
# Mensajes cortos de WhatsApp  
text_processing:
  chunk_size: 500
  chunk_overlap: 100
  
# Máxima precisión
text_processing:
  chunk_size: 800
  chunk_overlap: 200
```

### Configuración LLM

```yaml
# Respuestas creativas
llm:
  temperature: 0.9
  top_p: 0.95
  
# Respuestas consistentes
llm:
  temperature: 0.3
  top_p: 0.8
  
# Respuestas deterministas
llm:
  temperature: 0.0
  top_p: 1.0
```

## Configuración Específica por Caso de Uso

### Análisis de Conversaciones Largas

```yaml
text_processing:
  chunk_size: 1200
  chunk_overlap: 250

rag:
  top_k: 8
  max_context_length: 6000
  
llm:
  max_tokens: 3000
```

### Búsqueda Rápida de Información

```yaml
text_processing:
  chunk_size: 800
  chunk_overlap: 150

rag:
  top_k: 3
  similarity_threshold: 0.8
  
performance:
  max_workers: 6
  embedding_cache_size: 2000
```

### Análisis Detallado con Context Amplio

```yaml
rag:
  top_k: 10
  similarity_threshold: 0.6
  max_context_length: 8000
  enable_reranking: true
  
llm:
  max_tokens: 4000
  temperature: 0.5
```

## Seguridad y Privacidad

### Anonimización de Datos

```yaml
whatsapp_parser:
  anonymize_users: true
  anonymize_phone_numbers: true
  
security:
  filter_sensitive_info: true
  redact_phone_numbers: true
  redact_emails: true
```

### Control de Acceso

```bash
# Autenticación básica
GRADIO_USERNAME=admin
GRADIO_PASSWORD=contraseña_compleja_123!

# Rate limiting
MAX_REQUESTS_PER_MINUTE=30
MAX_CONCURRENT_REQUESTS=5

# Tamaño de archivos
MAX_FILE_SIZE_MB=5
```

## Optimización de Rendimiento

### Para CPU Limitadas

```yaml
performance:
  max_workers: 2
  chunk_processing_batch_size: 50
  
embedding:
  batch_size: 16
  
cache:
  enabled: true
  max_size: 500
```

### Para Servidores Potentes

```yaml  
performance:
  max_workers: 8
  chunk_processing_batch_size: 200
  
embedding:
  batch_size: 64
  device: "cuda"  # Si hay GPU
  
cache:
  enabled: true
  max_size: 5000
```

### Optimización de Memoria

```yaml
performance:
  embedding_cache_size: 500
  gc_threshold: 500
  
vector_store:
  # Usar índice comprimido
  index_type: "IndexFlatIP" 
  
cache:
  type: "filesystem"  # En lugar de memoria
```

## Monitoreo y Observabilidad

### Métricas Básicas

```yaml
monitoring:
  enable_metrics: true
  track_query_performance: true
  track_embedding_performance: true
```

### Métricas Avanzadas

```bash
# Variables adicionales
ENABLE_METRICS=true
METRICS_PORT=9090
PROMETHEUS_ENDPOINT=/metrics

# Logging estructurado
LOG_FORMAT=json
LOG_INCLUDE_TRACE_ID=true
```

### Integración con Observabilidad

```yaml
logging:
  level: "INFO"
  format: "json"
  
monitoring:
  enable_metrics: true
  metrics_port: 9090
  alert_on_errors: true
  alert_threshold_error_rate: 0.05
```

## Configuración de Almacenamiento

### Almacenamiento Local

```yaml
data:
  data_directory: "./data"
  index_directory: "./indices"  
  cache_directory: "./cache"
```

### Almacenamiento en la Nube (S3)

```bash
# Variables para S3
S3_BUCKET=whatsapp-rag-data
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
```

### Base de Datos para Metadata

```bash
# PostgreSQL para metadata
DATABASE_URL=postgresql://user:pass@localhost:5432/whatsapp_rag
DATABASE_POOL_SIZE=10
```

## Validación de Configuración

### Script de Validación

```python
def validate_config(config_path):
    """Valida archivo de configuración"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Validar secciones requeridas
    required_sections = ['embedding', 'rag', 'llm']
    for section in required_sections:
        assert section in config, f"Sección requerida: {section}"
    
    # Validar valores
    assert config['rag']['top_k'] > 0, "top_k debe ser > 0"
    assert config['embedding']['batch_size'] > 0, "batch_size debe ser > 0"
    
    print("✓ Configuración válida")

# Uso
validate_config("examples/config/rag_config.yaml")
```

### Verificación de Variables de Entorno

```bash
#!/bin/bash
# check_env.sh

required_vars=("OPENAI_API_KEY" "GITHUB_TOKEN")

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: Variable requerida $var no está definida"
        exit 1
    fi
done

echo "✓ Todas las variables requeridas están definidas"
```

## Mejores Prácticas

1. **Separar configuraciones por entorno** usando archivos diferentes
2. **Usar variables de entorno para secretos** (tokens, contraseñas)
3. **Validar configuración al inicio** de la aplicación  
4. **Documentar todas las opciones** y sus valores por defecto
5. **Usar configuración jerárquica** (defecto < archivo < env vars)
6. **Rotar secretos regularmente** en producción
7. **Monitorear cambios de configuración** en sistemas críticos