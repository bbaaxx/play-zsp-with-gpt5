"""
WhatsApp message parsing and processing with RAG pipeline management.
"""

from __future__ import annotations

import os
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from datetime import datetime
from rag.core import RAGPipeline, build_user_prompt, SYSTEM_PROMPT

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """Result of document indexing operation."""
    success: bool
    message: str
    n_messages: int = 0
    n_chunks: int = 0
    index_size: int = 0
    backend: str = ""


@dataclass
class ChatResponse:
    """Response from chat interaction."""
    answer: str
    error: Optional[str] = None


class ChatProcessor:
    """Handles WhatsApp message processing and RAG operations."""
    
    def __init__(self, vector_backend: Optional[str] = None, **vector_store_kwargs):
        """Initialize ChatProcessor with optional vector backend configuration."""
        self._pipeline: Optional[RAGPipeline] = None
        self._vector_backend = vector_backend or os.environ.get("VECTOR_BACKEND", "faiss").lower()
        self._vector_store_kwargs = vector_store_kwargs
        self.chat_history: List[Tuple[str, str]] = []
        self._analysis_summary: Optional[str] = None
        self._dynamic_system_prompt: Optional[str] = None
    
    def get_pipeline(self) -> RAGPipeline:
        """Get or create RAG pipeline instance."""
        if self._pipeline is None:
            vector_store_kwargs = self._vector_store_kwargs.copy()
            
            if self._vector_backend == "qdrant":
                vector_store_kwargs.update({
                    "url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
                    "api_key": os.environ.get("QDRANT_API_KEY"),
                    "collection_name": os.environ.get("QDRANT_COLLECTION_NAME", "whatsapp_rag"),
                })
                # Remove None values
                vector_store_kwargs = {k: v for k, v in vector_store_kwargs.items() if v is not None}
            
            self._pipeline = RAGPipeline(
                vector_backend=self._vector_backend,
                **vector_store_kwargs
            )
        return self._pipeline
    
    def index_messages(self, messages) -> IndexingResult:
        """Index WhatsApp messages for RAG retrieval."""
        if not messages:
            return IndexingResult(
                success=False,
                message="No hay mensajes para indexar",
                n_messages=0
            )
        
        logger.info("Indexando %d mensajes", len(messages))
        
        pipeline = self.get_pipeline()
        pipeline.index_messages(messages)
        
        n_msgs = len(messages)
        n_chunks = len(pipeline.chunks)
        size = pipeline.vector_store.size() if pipeline.vector_store else 0
        
        # Determine backend type for display
        if pipeline.vector_store:
            backend = type(pipeline.vector_store).__name__.replace("VectorStore", "").replace("InMemory", "")
        else:
            backend = "none"
        
        logger.info(
            "Indexado completado — mensajes=%d, chunks=%d, indice=%d, backend=%s",
            n_msgs, n_chunks, size, backend
        )
        
        return IndexingResult(
            success=True,
            message=f"Indexado OK — mensajes: {n_msgs}, chunks: {n_chunks}, tamaño índice: {size} (backend: {backend})",
            n_messages=n_msgs,
            n_chunks=n_chunks,
            index_size=size,
            backend=backend
        )
    
    def clear_state(self):
        """Clear all chat state and reset pipeline."""
        self._pipeline = None
        self.chat_history = []
        self._analysis_summary = None
        self._dynamic_system_prompt = None
        logger.info("Estado del procesador de chat limpiado")
    
    def check_llm_status(self) -> str:
        """Check the status of all LLM and embedding providers."""
        pipeline = self.get_pipeline()
        
        # LLM providers
        llm_providers = pipeline.llm_manager.list_providers()
        status_lines = ["=== Estado de Proveedores ===\n"]
        
        if llm_providers:
            status_lines.append("🤖 Proveedores LLM (Chat):")
            available_count = 0
            for provider in llm_providers:
                status = "✅ Disponible" if provider["available"] else "❌ No disponible"
                status_lines.append(f"  - {provider['name']}: {status}")
                if provider["available"]:
                    available_count += 1
            status_lines.append(f"  Disponibles: {available_count}/{len(llm_providers)}\n")
        else:
            status_lines.append("🤖 Proveedores LLM: No configurados\n")
        
        # Embedding providers status
        status_lines.append("🔍 Proveedores Embeddings:")
        embedder = pipeline.embedder
        
        # Check LM Studio embeddings
        if embedder.use_lm_studio and embedder._lm_studio_provider:
            lm_available = embedder._lm_studio_provider.is_available()
            status = "✅ Disponible" if lm_available else "❌ No disponible"
            status_lines.append(f"  - LM Studio Embeddings: {status}")
        else:
            status_lines.append("  - LM Studio Embeddings: No habilitado")
        
        # Check GitHub Models embeddings
        if embedder._remote_token and embedder._remote_base_url:
            status_lines.append("  - GitHub Models Embeddings: ✅ Configurado")
        else:
            status_lines.append("  - GitHub Models Embeddings: ❌ No configurado")
        
        # Check local embeddings
        try:
            embedder._ensure_local_model()
            status_lines.append("  - Local Embeddings: ✅ Disponible")
        except Exception:
            status_lines.append("  - Local Embeddings: ❌ No disponible")
        
        return "\n".join(status_lines)
    
    def _get_legacy_llm_client(self) -> Optional[OpenAI]:
        """Get legacy LLM client for backwards compatibility."""
        if OpenAI is None:
            return None
        token = os.environ.get("GITHUB_TOKEN")
        base_url = os.environ.get("GH_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
        if not token:
            return None
        
        logger.info("Creando cliente LLM legacy (base_url=%s)", base_url)
        return OpenAI(api_key=token, base_url=base_url)  # type: ignore[arg-type]
    
    def chat(
        self,
        user_msg: str,
        top_k: int = 5,
        model_name: Optional[str] = None,
        use_mmr: bool = True,
        lambda_: float = 0.5,
        fetch_k: int = 25,
        senders: Optional[List[str]] = None,
        date_from_iso: Optional[str] = None,
        date_to_iso: Optional[str] = None,
    ) -> ChatResponse:
        """Process chat message and generate response using RAG."""
        if not user_msg.strip():
            return ChatResponse(answer="", error="Escribe una pregunta.")
        
        pipeline = self.get_pipeline()
        logger.info(
            "Consulta recibida — top_k=%s, MMR=%s, λ=%.2f, fetch_k=%s, senders=%s, dfrom=%s, dto=%s",
            top_k, use_mmr, lambda_, fetch_k, senders, date_from_iso, date_to_iso,
        )
        
        # Retrieve relevant context
        retrieved = pipeline.retrieve(
            user_msg,
            top_k=top_k,
            use_mmr=use_mmr,
            fetch_k=fetch_k,
            lambda_=lambda_,
            senders=senders,
            date_from_iso=date_from_iso,
            date_to_iso=date_to_iso,
        )
        context = pipeline.format_context(retrieved)
        logger.info("Recuperados %d fragmentos", len(retrieved))

        # Try using the new LLM manager first
        try:
            # Use dynamic system prompt if available
            system_prompt = self.get_system_prompt()
            answer = pipeline.generate_answer(context, user_msg, system_prompt)
            logger.info("Respuesta LLM generada (%d chars)", len(answer))
        except Exception as e:
            logger.warning("Fallo en proveedores LLM nuevos: %s", e)
            
            # Fallback to legacy client for backwards compatibility
            client = self._get_legacy_llm_client()
            if client is None:
                answer = (
                    f"No hay proveedores LLM disponibles. Error: {e}\n"
                    "Contexto recuperado:\n" + context
                )
            else:
                try:
                    user_prompt = build_user_prompt(context, user_msg)
                    system_prompt = self.get_system_prompt()
                    
                    # Strip vendor prefix from model name for inference API
                    # GitHub Models catalog returns "openai/gpt-4o" but inference expects "gpt-4o"
                    inference_model = model_name or os.environ.get("CHAT_MODEL", "gpt-4o")
                    original_model = inference_model
                    if "/" in inference_model:
                        inference_model = inference_model.split("/", 1)[1]
                    
                    logger.info("Legacy client using model: '%s' (original: '%s')", inference_model, original_model)
                    
                    # Try with appropriate token parameter based on model
                    completion_params = {
                        "model": inference_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.2,
                    }
                    
                    # Use max_completion_tokens for newer models
                    if "gpt-5" in inference_model or "o1" in inference_model or "o3" in inference_model or "o4" in inference_model:
                        completion_params["max_completion_tokens"] = 800
                    else:
                        completion_params["max_tokens"] = 800
                    
                    resp = client.chat.completions.create(**completion_params)
                    answer = resp.choices[0].message.content or "(sin contenido)"
                    logger.info("Respuesta LLM legacy generada (%d chars)", len(answer))
                except Exception as legacy_e:
                    answer = f"Error en todos los proveedores LLM: {legacy_e}"
                    logger.exception("Fallo en cliente LLM legacy")

        self.chat_history.append((user_msg, answer))
        return ChatResponse(answer=answer)
    
    def get_chat_history(self) -> List[Tuple[str, str]]:
        """Get current chat history."""
        return self.chat_history.copy()
    
    def add_analysis_to_vector_store(self, analysis_text: str, analysis_type: str = "analysis"):
        """Store analysis results in the vector database."""
        pipeline = self.get_pipeline()
        
        # Create metadata for the analysis document
        analysis_metadata = {
            "chunk_id": f"{analysis_type}_{len(pipeline.chunks)}",
            "chat_id": "analysis",
            "start_ts": datetime.now().isoformat(),
            "end_ts": datetime.now().isoformat(), 
            "participants": ["system"],
            "line_span": [-1, -1],
            "text_window": analysis_text,
            "document_type": analysis_type
        }
        
        # Generate embeddings for the analysis text
        analysis_embeddings = pipeline.embedder.embed_texts([analysis_text])
        
        # Add to vector store if it exists
        if pipeline.vector_store is not None:
            pipeline.vector_store.add(
                ids=[analysis_metadata["chunk_id"]],
                embeddings=analysis_embeddings,
                metadatas=[analysis_metadata]
            )
            logger.info("Added %s to vector store", analysis_type)
        else:
            logger.warning("No vector store available to add analysis")
    
    def update_system_prompt_with_analysis(self, analysis_summary: str):
        """Update the system prompt to include analysis context."""
        self._analysis_summary = analysis_summary
        
        base_prompt = SYSTEM_PROMPT
        
        # Create enhanced system prompt with analysis context
        self._dynamic_system_prompt = (
            f"{base_prompt}\n\n"
            "CONTEXTO DEL ANÁLISIS PREVIO:\n"
            f"{analysis_summary}\n\n"
            "Utiliza este contexto del análisis cuando sea relevante para responder preguntas, "
            "especialmente sobre patrones, tendencias, participantes destacados, o características generales de la conversación."
        )
        
        logger.info("Updated system prompt with analysis context")
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt (base or enhanced with analysis)."""
        return self._dynamic_system_prompt or SYSTEM_PROMPT