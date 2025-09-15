# WhatsApp RAG Application - Complete User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [WhatsApp Chat Export Process](#whatsapp-chat-export-process)
6. [Using the Application](#using-the-application)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

## Introduction

The WhatsApp RAG (Retrieval-Augmented Generation) application allows you to upload WhatsApp chat exports and interact with them using natural language queries in Spanish. The system uses advanced AI techniques to parse your chat history, create semantic embeddings, and answer questions about your conversations.

**Key Features:**
- Parse WhatsApp text exports with multiple date/time formats
- Semantic search through chat history using embeddings
- AI-powered question answering in Spanish
- Support for both remote (GitHub Models) and local embedding models
- Interactive web interface built with Gradio

## System Requirements

### Hardware
- **Minimum:** 4GB RAM, 2GB free disk space
- **Recommended:** 8GB RAM, 4GB free disk space (for local embeddings)
- **CPU:** Any modern processor (GPU not required)

### Software
- **Python:** 3.11 or higher
- **Operating System:** Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Internet Connection:** Required for GitHub Models API (optional for local-only mode)

## Installation & Setup

### Step 1: Clone or Download the Application

If using Git:
```bash
git clone <repository-url>
cd whatsapp-rag
```

Or download and extract the ZIP file to your desired location.

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Dependencies installed:**
- `gradio==4.44.0` - Web interface framework
- `numpy==1.26.4` - Numerical computing
- `pandas==2.2.2` - Data manipulation
- `python-dotenv==1.0.1` - Environment variable management
- `openai==1.40.0` - OpenAI/GitHub Models API client
- `sentence-transformers==3.0.1` - Local embedding models
- `faiss-cpu` - Vector similarity search (automatically installed)
- Additional utilities for text processing

### Step 4: Verify Installation

```bash
python3 app.py --help
```

If no errors appear, the installation was successful.

## Configuration

### Environment Variables Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your preferred text editor:

### GitHub Models API Configuration (Recommended)

For the best experience with AI-powered responses, configure GitHub Models:

```bash
# Required: Your GitHub Personal Access Token
GITHUB_TOKEN=your_github_token_here

# Optional: GitHub Models API endpoint (default shown)
GH_MODELS_BASE_URL=https://models.github.ai/inference

# Optional: Chat model to use (default shown)
CHAT_MODEL=openai/gpt-4o

# Optional: Embedding model for semantic search (default shown)
EMBEDDING_MODEL=openai/text-embedding-3-small

# Optional: Use local embeddings instead of remote (0=remote, 1=local)
USE_LOCAL_EMBEDDINGS=0
```

**How to get a GitHub Token:**
1. Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with at least `read:user` scope
3. Copy the token to your `.env` file

### Local-Only Configuration

If you prefer not to use external APIs or don't have internet access:

```bash
# Use local embeddings only
USE_LOCAL_EMBEDDINGS=1

# GitHub token can be empty or omitted
GITHUB_TOKEN=

# Local embedding model (will be downloaded on first use)
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small
```

**Note:** Local mode will download ~120MB for the embedding model on first use.

### Advanced Configuration

```bash
# Server configuration (optional)
HOST=127.0.0.1          # Server host (default: localhost)
PORT=7860               # Server port (default: 7860)
GRADIO_SHARE=0          # Enable public sharing (0=no, 1=yes)
```

## WhatsApp Chat Export Process

### Step 1: Export Chat from WhatsApp

**On Mobile:**
1. Open WhatsApp
2. Go to the chat you want to export
3. Tap the contact/group name at the top
4. Scroll down and tap "Export Chat"
5. Choose "Without Media" (media files are not supported)
6. Save or share the `.txt` file

**On WhatsApp Web/Desktop:**
1. Open the chat
2. Click the three dots menu (⋮)
3. Select "More" → "Export chat"
4. Choose "Without media"
5. Save the `.txt` file to your computer

### Step 2: Verify Export Format

The application supports multiple WhatsApp export formats:

**Standard Format:**
```
[12/10/2023, 21:15] Juan: ¿Salimos mañana?
[12/10/2023, 21:16] María: Sí, ¿a qué hora te viene bien?
```

**Alternative Format:**
```
12/10/23, 21:22 - María: Sí, para tres personas por favor.
12/10/23, 21:25 - Pedro: Listo, 20:30 en La Trattoria.
```

**With AM/PM indicators:**
```
[26/05/25, 3:18:25 p.m.] Ana: Perfecto, nos vemos allí
[26/05/25, 3:20:15 a.m.] Carlos: ¿Tan temprano?
```

### Step 3: File Preparation

- **File size:** The application can handle large files (tested with 10MB+ exports)
- **Encoding:** UTF-8 encoding is automatically detected
- **Language:** Optimized for Spanish, but works with other languages
- **Privacy:** All processing is done locally; only retrieved context is sent to AI models

## Using the Application

### Starting the Application

1. **Activate your virtual environment** (if not already activated):
   ```bash
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

2. **Launch the application:**
   ```bash
   python3 app.py
   ```

3. **Access the web interface:**
   - Open your browser and go to: `http://127.0.0.1:7860`
   - The interface should load automatically

### Web Interface Overview

The application interface consists of several sections:

#### 1. File Upload Section
- **WhatsApp TXT File:** Click to browse and select your exported chat file
- Supports drag-and-drop functionality

#### 2. Configuration Controls
- **Top-k slider:** Number of relevant chat segments to retrieve (1-10)
  - Higher values provide more context but may include less relevant information
  - Recommended: 5 for balanced results
- **LLM Model field:** AI model to use for generating responses
  - Default: `openai/gpt-4o`
  - You can specify other GitHub Models like `openai/gpt-3.5-turbo`

#### 3. Action Buttons
- **Indexar (Index):** Process and index the uploaded file for the first time
- **Reindexar (Reindex):** Reprocess the same file (useful after configuration changes)
- **Limpiar chat (Clear Chat):** Clear conversation history and reset the system

#### 4. Status Display
- Shows indexing results: number of messages, chunks, and index size
- Displays error messages or processing status

#### 5. Chat Interface
- **Chat History:** Displays your questions and AI responses
- **Question Input:** Type your questions in Spanish
- **Enviar (Send):** Submit your question

### Step-by-Step Usage Workflow

#### Step 1: Upload and Index Your WhatsApp Chat

1. Click the **file upload area** and select your WhatsApp `.txt` export
2. Click **"Indexar"** to process the file
3. Wait for the status message showing successful indexing:
   ```
   Indexado OK — mensajes: 1,247, chunks: 84, tamaño índice: 84
   ```

**What happens during indexing:**
- Messages are parsed from the WhatsApp export format
- Text is cleaned and normalized (removes system messages, handles special characters)
- Messages are grouped into overlapping chunks of ~30 messages each
- Each chunk is converted to a semantic embedding vector
- A searchable index is created for fast retrieval

#### Step 2: Configure Search Parameters

- **Adjust Top-k value** based on your needs:
  - `1-3`: Very focused, specific answers
  - `4-6`: Balanced context and relevance (recommended)
  - `7-10`: Comprehensive context, may include less relevant information

#### Step 3: Ask Questions

Type questions in Spanish in the input field. Examples:

**Conversation Analysis:**
- "¿De qué hablamos la semana pasada?"
- "¿Cuándo quedamos para cenar?"
- "¿Qué planes teníamos para el fin de semana?"

**Participant-Specific Questions:**
- "¿Qué dijo María sobre el trabajo?"
- "¿Cuándo fue la última vez que Pedro escribió?"

**Topic-Based Queries:**
- "¿Hablamos de películas?"
- "¿Qué restaurantes mencionamos?"
- "¿Hubo alguna discusión sobre viajes?"

**Time-Based Questions:**
- "¿Qué pasó en octubre?"
- "¿De qué hablamos ayer?"

#### Step 4: Interpret AI Responses

The AI provides responses with:
- **Direct answers** to your questions
- **Source citations** in the format `[Sender — Date Time]`
- **Relevant context** from the retrieved chat segments

Example response:
```
Según los mensajes recuperados, quedaron para cenar el viernes a las 20:30 en La Trattoria. 
María confirmó la reserva para tres personas [María — 2023-10-12 21:22] y Pedro se 
encargó de hacer la reserva [Pedro — 2023-10-12 21:25].
```

### Advanced Features

#### Custom Model Configuration

You can experiment with different AI models by changing the "Modelo LLM" field:

**Available GitHub Models:**
- `openai/gpt-4o` (recommended, most capable)
- `openai/gpt-4o-mini` (faster, good performance)
- `openai/gpt-3.5-turbo` (faster, basic capabilities)
- `meta-llama/Llama-3.2-11B-Vision-Instruct` (alternative approach)

#### Embedding Model Selection

For semantic search, you can configure different embedding models in `.env`:

**Remote Options (via GitHub Models):**
- `openai/text-embedding-3-small` (default, good balance)
- `openai/text-embedding-3-large` (more accurate, slower)
- `openai/text-embedding-ada-002` (legacy, still effective)

**Local Options:**
- `intfloat/multilingual-e5-small` (default local, supports Spanish)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

#### Batch Processing

For processing multiple chat files:

1. Process the first file normally
2. Use "Limpiar chat" to reset
3. Upload and index the next file
4. The system maintains separate indexes for each session

#### Privacy and Security

- **Local Processing:** Message parsing and chunking happen locally
- **Remote Queries:** Only retrieved context segments are sent to AI models, not entire conversations
- **No Storage:** Chat data is not permanently stored; it exists only during your session
- **HTTPS:** All API communications use encrypted connections

## Troubleshooting

### Common Issues

#### 1. "No se detectaron mensajes" (No messages detected)

**Causes:**
- WhatsApp export format not recognized
- File encoding issues
- Empty or corrupted file

**Solutions:**
- Verify the file starts with lines like `[dd/mm/yyyy, hh:mm] Name: Message`
- Try re-exporting from WhatsApp
- Check the first few lines are shown correctly in the status message

#### 2. "GITHUB_TOKEN ausente" (GitHub token missing)

**Causes:**
- GitHub token not configured
- Token invalid or expired

**Solutions:**
- Set `USE_LOCAL_EMBEDDINGS=1` in `.env` for local-only mode
- Verify your GitHub token is valid
- Check token permissions include required scopes

#### 3. "Error al llamar al modelo" (Model call error)

**Causes:**
- Network connectivity issues
- API quota exceeded
- Invalid model name

**Solutions:**
- Check internet connection
- Verify GitHub Models access
- Try a different model name
- Wait and retry (may be temporary rate limiting)

#### 4. Application won't start

**Causes:**
- Missing dependencies
- Python version incompatibility
- Port already in use

**Solutions:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version
python3 --version  # Should be 3.11+

# Use different port
PORT=7861 python3 app.py
```

#### 5. Slow performance

**Causes:**
- Large chat files
- Using local embeddings on slow hardware
- High top-k values

**Solutions:**
- Use remote embeddings (faster)
- Reduce top-k value to 3-5
- Process smaller chat segments

### Debug Mode

For detailed error information, run with debug logging:

```bash
export LOG_LEVEL=DEBUG
python3 app.py
```

### Performance Optimization

#### For Large Files (>1000 messages):
- Use remote embeddings (`USE_LOCAL_EMBEDDINGS=0`)
- Start with top-k=3 for faster responses
- Consider processing chat segments separately

#### For Slower Systems:
- Set `USE_LOCAL_EMBEDDINGS=1` and use `intfloat/multilingual-e5-small`
- Reduce top-k to 3
- Close other applications while processing

## FAQ

### General Usage

**Q: What file formats are supported?**
A: Only WhatsApp `.txt` exports. The application automatically detects various WhatsApp export formats in multiple languages.

**Q: Can I upload multiple chat files?**
A: You can process one file per session. Use "Limpiar chat" to reset and upload a new file.

**Q: Is my chat data secure?**
A: Yes. Processing happens locally, and only small relevant segments are sent to AI models for generating responses.

**Q: Does it work with group chats?**
A: Yes, both individual and group chat exports are supported.

### Technical Questions

**Q: Which is better: local or remote embeddings?**
A: Remote embeddings (GitHub Models) are faster and often more accurate. Local embeddings provide complete privacy and work offline.

**Q: How much data is sent to external APIs?**
A: Only the retrieved context segments (typically 5-10 message snippets) are sent, not your entire conversation history.

**Q: Can I use it without internet?**
A: Yes, set `USE_LOCAL_EMBEDDINGS=1` and leave `GITHUB_TOKEN` empty. You'll get semantic search but no AI-generated responses.

**Q: What languages are supported?**
A: Optimized for Spanish, but works with other languages. The AI responds in Spanish by default.

### Performance & Limits

**Q: How large files can I process?**
A: The application can handle very large exports (tested with 10MB+, ~50,000 messages), though processing time increases with size.

**Q: Why are responses sometimes irrelevant?**
A: Try adjusting the top-k value. Lower values (2-3) give more focused results, higher values (7-10) provide more context.

**Q: Can I improve response quality?**
A: Yes, try different models (`gpt-4o` is most capable), adjust top-k values, and ask more specific questions.

### Customization

**Q: Can I change the system language?**
A: The interface is in Spanish, but you can modify the system prompts in `rag/core.py` for other languages.

**Q: How do I add custom models?**
A: Add any OpenAI-compatible model name to the "Modelo LLM" field, or modify the default in your `.env` file.

**Q: Can I export my search results?**
A: Currently not supported, but you can copy-paste responses from the chat interface.

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the application status messages for specific error details
2. Review your `.env` configuration
3. Try the troubleshooting steps above
4. Consult the technical documentation in `rag/` modules

The application is designed to be robust and user-friendly. Most issues can be resolved with proper configuration and file preparation.