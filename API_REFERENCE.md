# WhatsApp RAG API Reference

This document provides a comprehensive reference for the public classes, methods, and functions in the RAG system modules: `core.py`, `embeddings.py`, and `vector_store.py`.

## Table of Contents

- [rag.core](#ragcore)
  - [ChatMessage](#chatmessage)
  - [Chunk](#chunk)
  - [RAGPipeline](#ragpipeline)
  - [Utility Functions](#utility-functions)
- [rag.embeddings](#ragembeddings)
  - [EmbeddingProvider](#embeddingprovider)
- [rag.vector_store](#ragvector_store)
  - [InMemoryFAISS](#inmemoryfa)
  - [Utility Functions](#utility-functions-1)

---

## rag.core

The core module contains the main RAG pipeline implementation, WhatsApp message parsing, and chunking functionality.

### ChatMessage

A dataclass representing a single WhatsApp message.

```python
@dataclass
class ChatMessage:
    chat_id: str
    timestamp: datetime
    sender: str
    text: str
    line_no: int
```

#### Fields

- **chat_id** (`str`): Unique identifier for the chat/conversation
- **timestamp** (`datetime`): When the message was sent
- **sender** (`str`): Name of the person who sent the message
- **text** (`str`): The message content
- **line_no** (`int`): Line number in the original export file

#### Example

```python
from datetime import datetime
from rag.core import ChatMessage

message = ChatMessage(
    chat_id="abc123",
    timestamp=datetime(2023, 10, 12, 21, 15),
    sender="Juan",
    text="¿Salimos mañana?",
    line_no=42
)
```

### Chunk

A dataclass representing a window of messages for RAG processing.

```python
@dataclass
class Chunk:
    chunk_id: str
    chat_id: str
    start_ts: datetime
    end_ts: datetime
    participants: List[str]
    line_span: Tuple[int, int]
    text_window: str
```

#### Fields

- **chunk_id** (`str`): Unique identifier for this chunk
- **chat_id** (`str`): Chat identifier this chunk belongs to
- **start_ts** (`datetime`): Timestamp of first message in chunk
- **end_ts** (`datetime`): Timestamp of last message in chunk
- **participants** (`List[str]`): Sorted list of unique senders in this chunk
- **line_span** (`Tuple[int, int]`): Start and end line numbers from original file
- **text_window** (`str`): Formatted text representation of all messages in chunk

#### Example

```python
from rag.core import Chunk
from datetime import datetime

chunk = Chunk(
    chunk_id="chunk123",
    chat_id="abc123",
    start_ts=datetime(2023, 10, 12, 21, 0),
    end_ts=datetime(2023, 10, 12, 21, 30),
    participants=["Ana", "Juan"],
    line_span=(40, 65),
    text_window="[2023-10-12 21:15] Juan: ¿Salimos mañana?\n[2023-10-12 21:16] Ana: Sí, perfecto"
)
```

### RAGPipeline

The main class for building and querying a RAG system over WhatsApp messages.

```python
class RAGPipeline:
    def __init__(self) -> None
```

#### Constructor

Creates a new RAG pipeline instance with default embedding provider and empty vector store.

**Parameters:** None

**Example:**

```python
from rag.core import RAGPipeline

pipeline = RAGPipeline()
```

#### index_messages

```python
def index_messages(self, messages: List[ChatMessage]) -> None
```

Process and index a list of messages for retrieval.

**Parameters:**
- **messages** (`List[ChatMessage]`): Messages to index

**Returns:** `None`

**Side Effects:**
- Updates `self.chunks` with generated chunks
- Creates and populates `self.vector_store` with embeddings
- If messages list is empty, sets `self.vector_store` to `None`

**Example:**

```python
from rag.core import parse_whatsapp_txt, RAGPipeline

# Parse messages from WhatsApp export
with open("chat_export.txt", "r", encoding="utf-8") as f:
    content = f.read()

messages = parse_whatsapp_txt(content)
pipeline = RAGPipeline()
pipeline.index_messages(messages)
print(f"Indexed {len(pipeline.chunks)} chunks")
```

#### retrieve

```python
def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]
```

Search for the most relevant chunks given a query.

**Parameters:**
- **query** (`str`): Search query in natural language
- **top_k** (`int`, optional): Maximum number of results to return. Default: 5

**Returns:** `List[Dict[str, Any]]` - List of metadata dictionaries for matching chunks, each containing:
- `chunk_id`: Unique chunk identifier
- `chat_id`: Chat identifier  
- `start_ts`: ISO timestamp string of first message
- `end_ts`: ISO timestamp string of last message
- `participants`: List of participant names
- `line_span`: Tuple of (start_line, end_line)
- `text_window`: Formatted text content

**Example:**

```python
results = pipeline.retrieve("¿Qué dijeron sobre la reunión?", top_k=3)
for result in results:
    print(f"Participants: {result['participants']}")
    print(f"Time: {result['start_ts']} - {result['end_ts']}")
    print(f"Content: {result['text_window'][:100]}...")
    print("---")
```

#### format_context

```python
def format_context(self, metas: List[Dict[str, Any]]) -> str
```

Format retrieved chunk metadata into a readable context string.

**Parameters:**
- **metas** (`List[Dict[str, Any]]`): List of metadata dictionaries from `retrieve()`

**Returns:** `str` - Formatted context string with first 5 lines from each chunk

**Example:**

```python
results = pipeline.retrieve("planes para mañana")
context = pipeline.format_context(results)
print("Context for LLM:")
print(context)
```

### Utility Functions

#### parse_whatsapp_txt

```python
def parse_whatsapp_txt(content: str, chat_id: Optional[str] = None) -> List[ChatMessage]
```

Parse WhatsApp chat export text into structured messages.

**Parameters:**
- **content** (`str`): Raw WhatsApp export file content
- **chat_id** (`Optional[str]`): Custom chat ID. If None, generates hash from content

**Returns:** `List[ChatMessage]` - Parsed messages

**Supported Formats:**
- `[12/10/2023, 21:15] Juan: ¿Salimos mañana?` (24h format)
- `12/10/23, 21:15 - Juan: ¿Salimos mañana?` (with dash separator)
- `[26/05/25, 3:18:25 p.m.] Ana: Perfecto` (12h format with seconds)

**Example:**

```python
from rag.core import parse_whatsapp_txt

export_content = """[12/10/2023, 21:15] Juan: ¿Salimos mañana?
[12/10/2023, 21:16] Ana: Sí, perfecto
12/10/23, 21:17 - Juan: ¿A qué hora?"""

messages = parse_whatsapp_txt(export_content, chat_id="family_group")
print(f"Parsed {len(messages)} messages")
for msg in messages:
    print(f"{msg.sender}: {msg.text}")
```

#### chunk_messages

```python
def chunk_messages(
    messages: List[ChatMessage], 
    window_size: int = 30, 
    window_overlap: int = 10
) -> List[Chunk]
```

Split messages into overlapping chunks for better retrieval.

**Parameters:**
- **messages** (`List[ChatMessage]`): Input messages to chunk
- **window_size** (`int`, optional): Messages per chunk. Default: 30
- **window_overlap** (`int`, optional): Messages to overlap between chunks. Default: 10

**Returns:** `List[Chunk]` - List of message chunks

**Example:**

```python
from rag.core import chunk_messages

messages = parse_whatsapp_txt(content)
chunks = chunk_messages(messages, window_size=20, window_overlap=5)
print(f"Created {len(chunks)} chunks from {len(messages)} messages")

for chunk in chunks[:3]:  # Show first 3 chunks
    print(f"Chunk {chunk.chunk_id}: {chunk.participants}")
    print(f"Lines {chunk.line_span[0]}-{chunk.line_span[1]}")
```

#### build_user_prompt

```python
def build_user_prompt(context_snippets: str, question: str) -> str
```

Build a formatted prompt for the LLM with context and question.

**Parameters:**
- **context_snippets** (`str`): Retrieved context from `format_context()`
- **question** (`str`): User's question

**Returns:** `str` - Formatted prompt ready for LLM

**Example:**

```python
from rag.core import build_user_prompt

results = pipeline.retrieve("¿Cuándo es la reunión?")
context = pipeline.format_context(results)
prompt = build_user_prompt(context, "¿Cuándo es la reunión del proyecto?")
print("LLM Prompt:")
print(prompt)
```

---

## rag.embeddings

The embeddings module provides text embedding functionality with remote API and local model fallback.

### EmbeddingProvider

Provider for text embeddings with GitHub Models API and sentence-transformers fallback.

```python
class EmbeddingProvider:
    def __init__(self) -> None
```

#### Constructor

Initialize embedding provider with environment-based configuration.

**Environment Variables:**
- `USE_LOCAL_EMBEDDINGS`: Set to "1" to force local embeddings
- `EMBEDDING_MODEL`: Remote model name (default: "openai/text-embedding-3-small")  
- `LOCAL_EMBEDDING_MODEL`: Local model name (default: "intfloat/multilingual-e5-small")
- `GH_MODELS_BASE_URL`: GitHub Models API base URL
- `GITHUB_TOKEN`: Authentication token for GitHub Models

**Raises:**
- **RuntimeError**: If sentence-transformers is not installed and local fallback is required

**Example:**

```python
import os
from rag.embeddings import EmbeddingProvider

# Configure for remote embeddings
os.environ["GITHUB_TOKEN"] = "your_github_token"
embedder = EmbeddingProvider()

# Configure for local embeddings only
os.environ["USE_LOCAL_EMBEDDINGS"] = "1"
local_embedder = EmbeddingProvider()
```

#### embed_texts

```python
def embed_texts(self, texts: List[str]) -> np.ndarray
```

Embed a list of texts into L2-normalized vectors.

**Parameters:**
- **texts** (`List[str]`): List of texts to embed

**Returns:** `np.ndarray` - 2D array of shape (len(texts), embedding_dim) with float32 L2-normalized vectors

**Behavior:**
1. Attempts remote embeddings via GitHub Models API (if configured)
2. Falls back to local sentence-transformers model on failure
3. Always returns L2-normalized vectors for cosine similarity

**Example:**

```python
from rag.embeddings import EmbeddingProvider

embedder = EmbeddingProvider()
texts = [
    "¿Cuándo es la reunión?",
    "El proyecto va bien",
    "Necesitamos más tiempo"
]

embeddings = embedder.embed_texts(texts)
print(f"Shape: {embeddings.shape}")  # (3, 1536) for text-embedding-3-small
print(f"Normalized: {np.linalg.norm(embeddings[0])}")  # Should be ~1.0
```

---

## rag.vector_store

The vector store module provides in-memory vector search with FAISS or numpy fallback.

### InMemoryFAISS

In-memory vector store with cosine similarity search.

```python
class InMemoryFAISS:
    def __init__(self, dim: int) -> None
```

#### Constructor

Initialize vector store for embeddings of specified dimension.

**Parameters:**
- **dim** (`int`): Embedding dimension (must match embeddings added later)

**Example:**

```python
from rag.vector_store import InMemoryFAISS

# For OpenAI text-embedding-3-small (1536 dimensions)
store = InMemoryFAISS(dim=1536)

# For multilingual-e5-small (384 dimensions)  
local_store = InMemoryFAISS(dim=384)
```

#### add

```python
def add(
    self, 
    ids: List[str], 
    embeddings: np.ndarray, 
    metadatas: List[Dict[str, Any]]
) -> None
```

Add vectors with IDs and metadata to the store.

**Parameters:**
- **ids** (`List[str]`): Unique identifiers for each vector
- **embeddings** (`np.ndarray`): 2D array of vectors, shape (n_vectors, dim)
- **metadatas** (`List[Dict[str, Any]]`): Metadata for each vector

**Returns:** `None`

**Raises:**
- **ValueError**: If embedding dimensions don't match store dimension

**Example:**

```python
import numpy as np
from rag.vector_store import InMemoryFAISS

store = InMemoryFAISS(dim=384)

# Add some vectors
ids = ["doc1", "doc2", "doc3"]
embeddings = np.random.randn(3, 384).astype("float32")
metadatas = [
    {"title": "Documento 1", "content": "Contenido del primer documento"},
    {"title": "Documento 2", "content": "Contenido del segundo documento"},
    {"title": "Documento 3", "content": "Contenido del tercer documento"}
]

store.add(ids, embeddings, metadatas)
print(f"Store size: {store.size()}")
```

#### search

```python
def search(
    self, 
    query_embeddings: np.ndarray, 
    top_k: int = 5
) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]
```

Search for similar vectors using cosine similarity.

**Parameters:**
- **query_embeddings** (`np.ndarray`): Query vectors, shape (n_queries, dim)
- **top_k** (`int`, optional): Number of results per query. Default: 5

**Returns:** `Tuple[np.ndarray, List[List[Dict[str, Any]]]]`
- **scores** (`np.ndarray`): Similarity scores, shape (n_queries, top_k)
- **metadatas** (`List[List[Dict[str, Any]]]`): Metadata for each result

**Example:**

```python
# Search with single query
query_emb = embedder.embed_texts(["buscar documento importante"])
scores, results = store.search(query_emb, top_k=3)

print(f"Found {len(results[0])} results")
for i, (score, meta) in enumerate(zip(scores[0], results[0])):
    print(f"{i+1}. Score: {score:.3f}")
    print(f"   Title: {meta.get('title')}")
    print(f"   Content: {meta.get('content')[:50]}...")

# Batch search with multiple queries
batch_queries = embedder.embed_texts([
    "documento importante", 
    "contenido relevante"
])
batch_scores, batch_results = store.search(batch_queries, top_k=2)
```

#### size

```python
def size(self) -> int
```

Get the number of vectors in the store.

**Returns:** `int` - Number of stored vectors

**Example:**

```python
print(f"Vector store contains {store.size()} documents")
```

### Utility Functions

#### l2_normalize

```python
def l2_normalize(vectors: np.ndarray) -> np.ndarray
```

L2-normalize vectors for cosine similarity computation.

**Parameters:**
- **vectors** (`np.ndarray`): Input vectors to normalize

**Returns:** `np.ndarray` - L2-normalized vectors with same shape, dtype float32

**Example:**

```python
import numpy as np
from rag.vector_store import l2_normalize

vectors = np.random.randn(10, 384)
normalized = l2_normalize(vectors)
print(f"Norms: {np.linalg.norm(normalized, axis=1)}")  # Should be ~1.0
```

---

## Complete Usage Example

Here's a comprehensive example showing how to use all components together:

```python
import os
from rag.core import RAGPipeline, parse_whatsapp_txt, build_user_prompt, SYSTEM_PROMPT

# Set up environment
os.environ["GITHUB_TOKEN"] = "your_github_token_here"

# Initialize RAG pipeline
pipeline = RAGPipeline()

# Load and parse WhatsApp export
with open("chat_export.txt", "r", encoding="utf-8") as f:
    content = f.read()

messages = parse_whatsapp_txt(content, chat_id="family_chat")
print(f"Parsed {len(messages)} messages")

# Index messages for search
pipeline.index_messages(messages)
print(f"Created {len(pipeline.chunks)} searchable chunks")

# Query the system
question = "¿Qué planes tenemos para el fin de semana?"
results = pipeline.retrieve(question, top_k=5)

# Format context for LLM
context = pipeline.format_context(results)
user_prompt = build_user_prompt(context, question)

print("=== Context Retrieved ===")
print(context)
print("\n=== LLM Prompt ===")
print(user_prompt)

# The prompt can now be sent to any LLM along with SYSTEM_PROMPT
```

This API reference covers all public interfaces in the RAG system. Each component is designed to work independently or as part of the complete pipeline, providing flexibility for different use cases.