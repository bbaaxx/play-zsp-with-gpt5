# Docker Deployment

Este directorio contiene archivos para desplegar WhatsApp RAG usando Docker.

## Configuración Rápida

### 1. Configurar Variables de Entorno

Copia el archivo `.env.example` del directorio raíz y configura:

```bash
cp ../../.env.example .env
```

Edita `.env` con tus credenciales:
```
OPENAI_API_KEY=tu_token_de_github
GITHUB_TOKEN=tu_token_de_github
```

### 2. Construcción y Ejecución

```bash
# Construir y ejecutar
docker-compose up --build -d

# Ver logs
docker-compose logs -f whatsapp-rag

# Detener
docker-compose down
```

La aplicación estará disponible en `http://localhost:7860`

## Despliegue con Proxy (Producción)

Para usar nginx como proxy reverso:

```bash
# Generar certificados SSL (reemplaza con tus dominios)
mkdir ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout ssl/key.pem -out ssl/cert.pem

# Ejecutar con proxy
docker-compose --profile proxy up -d
```

## Configuración Avanzada

### Variables de Entorno

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `OPENAI_API_KEY` | Token de GitHub para modelos | Requerido |
| `GITHUB_TOKEN` | Token de GitHub (mismo que OPENAI_API_KEY) | Requerido |
| `MODEL_NAME` | Modelo a usar | gpt-4o-mini |
| `OPENAI_API_BASE` | URL base de la API | https://models.inference.ai.azure.com |

### Volúmenes

- `data/`: Archivos de WhatsApp (solo lectura)
- `app_logs/`: Logs de la aplicación (persistentes)

### Recursos

Para ajustar recursos del contenedor, añade a `docker-compose.yml`:

```yaml
services:
  whatsapp-rag:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          memory: 1G
```

## Monitoreo

### Health Checks

El contenedor incluye health checks automáticos cada 30 segundos.

```bash
# Ver estado de salud
docker-compose ps
docker inspect --format='{{json .State.Health}}' whatsapp-rag-whatsapp-rag-1
```

### Logs

```bash
# Logs en tiempo real
docker-compose logs -f

# Logs específicos del servicio
docker-compose logs whatsapp-rag

# Logs con filtro de tiempo
docker-compose logs --since="1h" whatsapp-rag
```

## Escalabilidad

Para múltiples instancias:

```yaml
services:
  whatsapp-rag:
    scale: 3
    # ... resto de configuración
```

```bash
docker-compose up --scale whatsapp-rag=3 -d
```