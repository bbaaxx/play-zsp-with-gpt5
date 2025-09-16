# WhatsApp RAG (ES) - Ejemplos de Uso

Esta carpeta contiene scripts de demostración para aprender a usar el sistema WhatsApp RAG. Cada ejemplo muestra diferentes aspectos del sistema, desde uso básico hasta configuraciones avanzadas.

## 📋 Prerrequisitos

### Instalación Básica
```bash
# Desde el directorio raíz del proyecto
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Para ejemplos con LLM
```bash
# Configurar token de GitHub Models (opcional pero recomendado)
export GITHUB_TOKEN=tu_github_token_aqui
```

### Datos de Prueba
Los ejemplos pueden usar:
- `data/sample_whatsapp.txt` - Archivo de muestra incluido
- Tu propio archivo TXT exportado de WhatsApp

## 🚀 Scripts Disponibles

### 1. `basic_usage.py` - Uso Básico del Sistema RAG

**¿Qué demuestra?**
- Carga e indexación de mensajes de WhatsApp
- Parseo automático de diferentes formatos de fecha/hora
- Creación de chunks y vectorización
- Recuperación de documentos relevantes

**Cómo ejecutar:**
```bash
cd examples
python basic_usage.py
```

**Salida esperada:**
- Estadísticas de mensajes parseados
- Información sobre chunks e índice vectorial
- Resultados de consultas de ejemplo sin LLM
- Contexto recuperado para cada pregunta

**Cuándo usarlo:**
- Primera vez usando el sistema
- Verificar que tu archivo de WhatsApp se parsea correctamente
- Entender el proceso de indexación

---

### 2. `llm_example.py` - RAG Completo con LLM

**¿Qué demuestra?**
- Pipeline RAG completo incluyendo generación de respuestas
- Integración con GitHub Models/OpenAI
- Construcción de prompts para el LLM
- Respuestas en español con citaciones

**Prerrequisitos:**
```bash
export GITHUB_TOKEN=tu_token  # Requerido
```

**Cómo ejecutar:**
```bash
cd examples
python llm_example.py
```

**Salida esperada:**
- Respuestas completas en español del LLM
- Contexto recuperado para cada pregunta
- Citaciones con remitente y fecha
- Detalles de configuración del modelo

**Cuándo usarlo:**
- Después de verificar el funcionamiento básico
- Para ver el sistema completo en acción
- Cuando necesitas respuestas generadas, no solo recuperación

---

### 3. `custom_data_example.py` - Datos Personalizados y Configuraciones Avanzadas

**¿Qué demuestra?**
- Uso con tus propios archivos de WhatsApp
- Análisis estadístico del chat
- Filtrado por remitente y fecha
- Configuraciones avanzadas (MMR, parámetros de recuperación)
- Diferentes algoritmos de búsqueda

**Cómo ejecutar:**
```bash
cd examples

# Con tu propio archivo
python custom_data_example.py /ruta/a/tu/whatsapp_export.txt

# O con el archivo de ejemplo
python custom_data_example.py
```

**Salida esperada:**
- Estadísticas detalladas del chat (mensajes, participantes, fechas)
- Comparación de diferentes métodos de recuperación
- Ejemplos de filtrado avanzado
- Configuraciones personalizables

**Cuándo usarlo:**
- Cuando quieres usar tus propios datos
- Para entender opciones avanzadas de configuración
- Analizar estadísticas de tus chats

---

## 📁 Formatos de Archivo Soportados

El sistema reconoce varios formatos de exportación de WhatsApp:

### Formato Estándar (24h)
```
[12/10/2023, 21:15] Juan: ¿Salimos mañana?
[12/10/2023, 21:16] María: Sí, ¿a qué hora?
```

### Formato con Guión
```
12/10/23, 21:22 - María: Perfecto
12/10/23, 21:25 - Pedro: Listo
```

### Formato con AM/PM
```
[26/05/25, 3:18:25 p.m.] Ana: Hola
[26/05/25, 8:30 a.m.] Carlos: Buenos días
```

## ⚙️ Variables de Entorno

### Requeridas para LLM
- `GITHUB_TOKEN` - Token para acceder a GitHub Models

### Opcionales
- `CHAT_MODEL` - Modelo LLM (por defecto: `openai/gpt-4o`)
- `GH_MODELS_BASE_URL` - URL base para GitHub Models
- `HOST` / `PORT` - Para la aplicación web (app.py)

## 🔧 Parámetros de Configuración

### Recuperación de Documentos
- `top_k` (1-20): Número de fragmentos a recuperar
- `use_mmr` (bool): Usar Maximum Marginal Relevance para diversidad
- `lambda_` (0.0-1.0): Balance entre relevancia (1.0) y diversidad (0.0)
- `fetch_k` (5-100): Candidatos iniciales para MMR

### Filtros
- `senders`: Lista de remitentes específicos
- `date_from_iso` / `date_to_iso`: Rango de fechas en formato ISO

### Chunking (en código)
- `window_size`: Mensajes por chunk (por defecto: 30)
- `window_overlap`: Solapamiento entre chunks (por defecto: 10)

## 🐛 Solución de Problemas

### "No se encontraron mensajes válidos"
- Verifica que el archivo sea un TXT exportado de WhatsApp
- Revisa que las fechas estén en formato dd/mm/yyyy o dd/mm/yy
- Asegúrate de que la codificación sea UTF-8

### "GITHUB_TOKEN no configurado"
```bash
# Obtener token en https://github.com/settings/tokens
export GITHUB_TOKEN=tu_token_github
```

### "Error al leer archivo"
- Prueba con encoding diferente (el script lo intenta automáticamente)
- Verifica permisos de lectura del archivo
- Confirma que la ruta sea correcta

### "No se encontraron fragmentos relevantes"
- Prueba diferentes consultas
- Reduce `top_k` o ajusta `lambda_` para MMR
- Verifica que el contenido sea relevante a tu pregunta

## 🔗 Siguientes Pasos

1. **Ejecuta `basic_usage.py`** para verificar funcionamiento
2. **Configura GITHUB_TOKEN** y prueba `llm_example.py`
3. **Usa `custom_data_example.py`** con tus propios datos
4. **Ejecuta `python app.py`** para la interfaz web completa
5. **Lee los archivos de documentación** en el directorio raíz

## 📚 Recursos Adicionales

- `README.md` - Documentación general del proyecto
- `USER_GUIDE.md` - Guía detallada de usuario
- `API_REFERENCE.md` - Referencia de la API
- `ARCHITECTURE.md` - Arquitectura del sistema

---

¿Tienes problemas? Revisa los logs de salida de cada script para información detallada sobre errores y configuración.