from __future__ import annotations

import os
import logging
import gradio as gr
from typing import List, Tuple, Optional
from dotenv import load_dotenv

from business import FileManager, ChatProcessor, AnalyticsEngine

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# Load .env file with override=True to take precedence over existing env vars
load_dotenv(override=True)
# Set environment variable to disable tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Monkey patch for a Gradio JSON schema bug where additionalProperties can be boolean
try:
    from gradio_client import utils as _gc_utils  # type: ignore

    _orig_json_schema_to_python_type = getattr(_gc_utils, "_json_schema_to_python_type", None)
    _orig_get_type = getattr(_gc_utils, "get_type", None)

    if callable(_orig_json_schema_to_python_type):
        def _safe_json_schema_to_python_type(schema, defs):  # type: ignore
            if isinstance(schema, bool):
                return "Any"
            return _orig_json_schema_to_python_type(schema, defs)

        _gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type  # type: ignore

    if callable(_orig_get_type):
        def _safe_get_type(schema):  # type: ignore
            if isinstance(schema, bool):
                return "Any"
            return _orig_get_type(schema)

        _gc_utils.get_type = _safe_get_type  # type: ignore
except Exception:
    pass

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)
logger = logging.getLogger("app")


class AppState:
    def __init__(self) -> None:
        self.file_manager = FileManager()
        self.chat_processor = ChatProcessor()
        self.analytics_engine = AnalyticsEngine()
        # Connect analytics engine to chat processor for integration
        self.analytics_engine.set_chat_processor(self.chat_processor)
        self.loaded_text: Optional[str] = None
        self.chat_dataframe = None


STATE = AppState()


def index_file(file_obj) -> str:
    # Process the file
    result = STATE.file_manager.process_whatsapp_file(file_obj)
    if not result.success:
        return result.message
    
    # Store the processed data
    STATE.loaded_text = result.content
    STATE.chat_dataframe = result.chat_dataframe
    
    # Parse messages from content and index them
    from rag.core import parse_whatsapp_txt
    messages = parse_whatsapp_txt(result.content)
    
    # Index messages using chat processor
    indexing_result = STATE.chat_processor.index_messages(messages)
    
    if not indexing_result.success:
        return indexing_result.message
    
    if indexing_result.n_messages == 0:
        preview = "\n".join(result.content.splitlines()[:3])
        return (
            "Indexado OK — mensajes: 0, chunks: 0, tamaño índice: 0\n"
            "No se detectaron mensajes. Verifica que el archivo sea un export estándar de WhatsApp (TXT).\n"
            f"Primeras líneas leídas:\n{preview}"
        )
    
    return indexing_result.message


def clear_state():
    STATE.chat_processor.clear_state()
    STATE.analytics_engine.clear_analysis_state()
    STATE.loaded_text = None
    STATE.chat_dataframe = None
    return [], "Estado limpiado."


def initialize_chat_interface():
    """Initialize chat interface with existing analysis context if available."""
    try:
        # Check if we have analysis results that should enhance the system prompt
        if STATE.analytics_engine.last_adaptive_summary:
            summary = STATE.analytics_engine.last_adaptive_summary
            STATE.chat_processor.update_system_prompt_with_analysis(summary)
            logger.info("Initialized chat with adaptive analysis context")
        elif STATE.analytics_engine.last_analysis_summary:
            summary = STATE.analytics_engine.last_analysis_summary
            STATE.chat_processor.update_system_prompt_with_analysis(summary)
            logger.info("Initialized chat with basic analysis context")
        else:
            logger.info("No existing analysis context found")
    except Exception as e:
        logger.exception("Error initializing chat interface: %s", e)


def update_chat_processor_config(provider: str, model: str):
    """Update chat processor configuration based on selected provider/model."""
    # This function can be expanded to update the provider configuration
    # For now, it's a placeholder that could set environment variables or
    # reconfigure the chat processor's LLM manager
    pass


def update_embeddings_config(provider: str, model: str):
    """Update embeddings configuration based on selected provider/model."""
    # Update environment variables for the embedding provider
    if provider == "github_models":
        os.environ["USE_LOCAL_EMBEDDINGS"] = "0"
        os.environ["LMSTUDIO_EMBEDDINGS_ENABLED"] = "0" 
        os.environ["EMBEDDING_MODEL"] = model
    elif provider == "lm_studio":
        os.environ["USE_LOCAL_EMBEDDINGS"] = "0"
        os.environ["LMSTUDIO_EMBEDDINGS_ENABLED"] = "1"
        os.environ["LMSTUDIO_EMBEDDING_MODEL"] = model
    elif provider == "local":
        os.environ["USE_LOCAL_EMBEDDINGS"] = "1"
        os.environ["LMSTUDIO_EMBEDDINGS_ENABLED"] = "0"
        os.environ["LOCAL_EMBEDDING_MODEL"] = model
    
    # Clear the pipeline to force reinitialization with new config
    STATE.chat_processor._pipeline = None


def update_analysis_config(provider: str, model: str):
    """Update analysis configuration based on selected provider/model.""" 
    # Update environment variables for analysis LLM
    if provider == "github_models":
        os.environ["ANALYSIS_MODEL"] = model
        os.environ["ANALYSIS_PROVIDER"] = "github_models"
        os.environ["CHAT_MODEL"] = model  # Also update for compatibility
    elif provider == "lm_studio":
        os.environ["ANALYSIS_MODEL"] = model
        os.environ["ANALYSIS_PROVIDER"] = "lm_studio"
        os.environ["CHAT_MODEL"] = model  # Also update for compatibility
        os.environ["LMSTUDIO_ENABLED"] = "1"
    
    # Update the analytics engine configuration
    STATE.analytics_engine.update_config(provider, model)


def check_llm_status():
    """Check the status of all LLM and embedding providers."""
    return STATE.chat_processor.check_llm_status()


def chat(
    user_msg: str,
    top_k: int,
    model_name: str,
    use_mmr: bool = True,
    lambda_: float = 0.5,
    fetch_k: int = 25,
    senders: Optional[List[str]] = None,
    date_from_iso: Optional[str] = None,
    date_to_iso: Optional[str] = None,
) -> Tuple[List[Tuple[str, str]], str]:
    
    response = STATE.chat_processor.chat(
        user_msg=user_msg,
        top_k=top_k,
        model_name=model_name,
        use_mmr=use_mmr,
        lambda_=lambda_,
        fetch_k=fetch_k,
        senders=senders,
        date_from_iso=date_from_iso,
        date_to_iso=date_to_iso,
    )
    
    if response.error:
        return STATE.chat_processor.get_chat_history(), response.error
    
    return STATE.chat_processor.get_chat_history(), ""


def analyze_chat(progress=gr.Progress()) -> Tuple[str, str]:
    """Realiza un análisis inteligente del chat usando smolagents."""
    detailed_results = STATE.analytics_engine.analyze_chat_basic(STATE.chat_dataframe, progress)
    summary = STATE.analytics_engine.get_last_basic_summary() or "No se pudo generar resumen."
    return summary, detailed_results


def analyze_chat_adaptive(progress=gr.Progress()) -> Tuple[str, str]:
    """Realiza análisis adaptativo de dos etapas."""
    detailed_results = STATE.analytics_engine.analyze_chat_adaptive(STATE.chat_dataframe, progress)
    summary = STATE.analytics_engine.get_last_adaptive_summary() or "No se pudo generar resumen."
    return summary, detailed_results


def get_analysis_summary() -> str:
    """Obtiene un resumen rápido del último análisis."""
    return STATE.analytics_engine.get_analysis_summary()


def _fetch_github_models() -> List[str]:
    """Fetch available models from GitHub Models API."""
    try:
        import httpx
    except ImportError:
        logger.error("httpx not available for fetching GitHub models")
        return _get_github_models_fallback()
    
    token = os.environ.get("GITHUB_TOKEN")
    
    if not token:
        # Return hardcoded fallback list if no token
        return _get_github_models_fallback()
    
    try:
        # Use the correct GitHub API endpoint for models
        url = "https://api.github.com/models"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
            models = []
            
            # Handle the GitHub models API response format
            if isinstance(data, list):
                # Direct model list from GitHub API
                for model in data:
                    if isinstance(model, dict):
                        model_id = model.get("name") or model.get("id")
                        if model_id:
                            # Filter for text generation models (exclude embedding models)
                            model_type = model.get("type", "").lower()
                            tags = model.get("tags", [])
                            
                            # Include models that are for text generation/chat
                            if (not model_type or 
                                model_type in ["text-generation", "conversational", "chat"] or
                                "chat" in tags or
                                "text-generation" in tags or
                                "conversational" in tags):
                                
                                # Skip embedding models specifically
                                if not (model_type == "embedding" or "embedding" in tags):
                                    models.append(model_id)
                            
            elif isinstance(data, dict):
                # Handle wrapped response
                model_list = data.get("data", data.get("models", []))
                if isinstance(model_list, list):
                    for model in model_list:
                        if isinstance(model, dict):
                            model_id = model.get("name") or model.get("id")
                            if model_id:
                                model_type = model.get("type", "").lower()
                                tags = model.get("tags", [])
                                
                                if (not model_type or 
                                    model_type in ["text-generation", "conversational", "chat"] or
                                    "chat" in tags or
                                    "text-generation" in tags or
                                    "conversational" in tags):
                                    
                                    if not (model_type == "embedding" or "embedding" in tags):
                                        models.append(model_id)
            
            # Sort models and remove duplicates
            models = sorted(list(set(models)))
            
            # If no models found via API, return fallback
            if not models:
                logger.warning("No models returned from GitHub API, using fallback list")
                return _get_github_models_fallback()
            
            logger.info(f"Successfully fetched {len(models)} models from GitHub Models API")
            return models
            
    except Exception as e:
        logger.error(f"Error fetching GitHub Models: {e}")
        # Return hardcoded fallback list on error
        return _get_github_models_fallback()


def _get_github_models_fallback() -> List[str]:
    """Get fallback list of known GitHub Models."""
    return [
        # OpenAI models
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        
        # Microsoft Phi models
        "Phi-3-medium-128k-instruct",
        "Phi-3-medium-4k-instruct", 
        "Phi-3-mini-128k-instruct",
        "Phi-3-mini-4k-instruct",
        "Phi-3-small-128k-instruct",
        "Phi-3-small-8k-instruct",
        "Phi-3.5-mini-instruct",
        "Phi-3.5-MoE-instruct",
        
        # Meta Llama models
        "Llama-3.2-11B-Vision-Instruct",
        "Llama-3.2-90B-Vision-Instruct",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Meta-Llama-3-70B-Instruct",
        "Meta-Llama-3-8B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-8B-Instruct",
        
        # Mistral models
        "Mistral-7B-Instruct-v0.1",
        "Mistral-7B-Instruct-v0.3",
        "Mistral-large",
        "Mistral-large-2407",
        "Mistral-Nemo",
        "Mistral-small",
        
        # Cohere models  
        "Cohere-command-r",
        "Cohere-command-r-plus",
        
        # AI21 models
        "jamba-1.5-large",
        "jamba-1.5-mini"
    ]


def get_available_models(provider: str) -> List[str]:
    """Get available models for a given provider."""
    if provider == "github_models":
        return _fetch_github_models()
    elif provider == "lm_studio":
        # For LM Studio, we can't dynamically query models, so provide default
        return ["local-chat-model", "custom-model"]
    return []


def _fetch_github_embedding_models() -> List[str]:
    """Fetch available embedding models from GitHub Models API."""
    try:
        import httpx
    except ImportError:
        logger.error("httpx not available for fetching GitHub embedding models")
        return _get_github_embedding_models_fallback()
    
    token = os.environ.get("GITHUB_TOKEN")
    
    if not token:
        # Return hardcoded fallback list if no token
        return _get_github_embedding_models_fallback()
    
    try:
        # Use the correct GitHub API endpoint for models
        url = "https://api.github.com/models"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
        
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
            models = []
            
            # Handle the GitHub models API response format
            if isinstance(data, list):
                # Direct model list from GitHub API
                for model in data:
                    if isinstance(model, dict):
                        model_id = model.get("name") or model.get("id")
                        if model_id:
                            # Filter for embedding models
                            model_type = model.get("type", "").lower()
                            tags = model.get("tags", [])
                            
                            # Include models that are specifically for embedding
                            if (model_type == "embedding" or 
                                "embedding" in tags or
                                "embedding" in model_id.lower()):
                                models.append(model_id)
                            
            elif isinstance(data, dict):
                # Handle wrapped response
                model_list = data.get("data", data.get("models", []))
                if isinstance(model_list, list):
                    for model in model_list:
                        if isinstance(model, dict):
                            model_id = model.get("name") or model.get("id")
                            if model_id:
                                model_type = model.get("type", "").lower()
                                tags = model.get("tags", [])
                                
                                if (model_type == "embedding" or 
                                    "embedding" in tags or
                                    "embedding" in model_id.lower()):
                                    models.append(model_id)
            
            # Sort models and remove duplicates
            models = sorted(list(set(models)))
            
            # If no embedding models found via API, return fallback
            if not models:
                logger.warning("No embedding models returned from GitHub API, using fallback list")
                return _get_github_embedding_models_fallback()
            
            logger.info(f"Successfully fetched {len(models)} embedding models from GitHub Models API")
            return models
            
    except Exception as e:
        logger.error(f"Error fetching GitHub embedding models: {e}")
        return _get_github_embedding_models_fallback()


def _get_github_embedding_models_fallback() -> List[str]:
    """Get fallback list of known GitHub embedding models."""
    return [
        # OpenAI embedding models
        "text-embedding-3-small",
        "text-embedding-3-large", 
        "text-embedding-ada-002"
    ]


def get_available_embedding_models(provider: str) -> List[str]:
    """Get available embedding models for a given provider."""
    if provider == "github_models":
        return _fetch_github_embedding_models()
    elif provider == "lm_studio":
        return ["local-embedding-model", "custom-embedding-model"]
    elif provider == "local":
        return [
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
    return []


def update_models_dropdown(provider: str) -> gr.Dropdown:
    """Update models dropdown when provider changes."""
    models = get_available_models(provider)
    return gr.Dropdown(
        choices=models,
        value=models[0] if models else "",
        label="Modelo"
    )


def update_embedding_models_dropdown(provider: str) -> gr.Dropdown:
    """Update embedding models dropdown when provider changes."""
    models = get_available_embedding_models(provider)
    return gr.Dropdown(
        choices=models,
        value=models[0] if models else "",
        label="Modelo de Embeddings"
    )


def refresh_models_dropdown(provider: str) -> gr.Dropdown:
    """Refresh models dropdown to fetch latest models from API."""
    models = get_available_models(provider)
    return gr.Dropdown(
        choices=models,
        value=models[0] if models else "",
        label="Modelo"
    )


def refresh_embedding_models_dropdown(provider: str) -> gr.Dropdown:
    """Refresh embedding models dropdown to fetch latest models from API.""" 
    models = get_available_embedding_models(provider)
    return gr.Dropdown(
        choices=models,
        value=models[0] if models else "",
        label="Modelo de Embeddings"
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="WhatsApp RAG (ES)") as demo:
        gr.Markdown("""
        ### WhatsApp RAG (ES)
        Carga un TXT exportado de WhatsApp, indexa y consulta en español. Solo se envían al LLM los fragmentos recuperados.
        """)

        with gr.Row():
            file_input = gr.File(label="Archivo TXT de WhatsApp", file_count="single", type="filepath")
        
        # Provider and model selection section
        with gr.Accordion("⚙️ Configuración de Proveedores", open=False):
            with gr.Row():
                # Embeddings configuration
                with gr.Column():
                    gr.Markdown("**1. Embeddings**")
                    embed_provider = gr.Dropdown(
                        choices=["github_models", "lm_studio", "local"],
                        value="local",
                        label="Proveedor de Embeddings"
                    )
                    embed_model = gr.Dropdown(
                        choices=get_available_embedding_models("local"),
                        value="intfloat/multilingual-e5-small",
                        label="Modelo de Embeddings"
                    )
                    embed_refresh_btn = gr.Button("🔄", size="sm", variant="secondary")
                
                # Analysis configuration  
                with gr.Column():
                    gr.Markdown("**2. Análisis**")
                    analysis_provider = gr.Dropdown(
                        choices=["github_models", "lm_studio"],
                        value="github_models",
                        label="Proveedor para Análisis"
                    )
                    analysis_model = gr.Dropdown(
                        choices=get_available_models("github_models"),
                        value="gpt-4o",
                        label="Modelo para Análisis"
                    )
                    analysis_refresh_btn = gr.Button("🔄", size="sm", variant="secondary")
                
                # Chat configuration
                with gr.Column():
                    gr.Markdown("**3. Chat**")
                    chat_provider = gr.Dropdown(
                        choices=["github_models", "lm_studio"],
                        value="github_models", 
                        label="Proveedor para Chat"
                    )
                    chat_model = gr.Dropdown(
                        choices=get_available_models("github_models"),
                        value="gpt-4o",
                        label="Modelo para Chat"
                    )
                    chat_refresh_btn = gr.Button("🔄", size="sm", variant="secondary")

        with gr.Row():
            topk = gr.Slider(1, 10, value=5, step=1, label="Top-k")
            mmr = gr.Checkbox(value=True, label="MMR")
            lambda_box = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="λ (MMR)")
            fetchk = gr.Slider(5, 50, value=25, step=1, label="fetch_k")
        with gr.Row():
            sender_filter = gr.Textbox(label="Filtrar por remitente(s), separados por coma", placeholder="María, Juan")
            date_from = gr.Textbox(label="Desde (ISO)", placeholder="2025-06-01T00:00")
            date_to = gr.Textbox(label="Hasta (ISO)", placeholder="2025-06-30T23:59")
        with gr.Row():
            index_btn = gr.Button("Indexar")
            reindex_btn = gr.Button("Reindexar")
            clear_btn = gr.Button("Limpiar chat")
            status_btn = gr.Button("Estado LLM")
        with gr.Row():
            analyze_btn = gr.Button("🔍 Analizar Conversación", variant="secondary")
            adaptive_btn = gr.Button("🎯 Análisis Adaptativo", variant="primary")
            summary_btn = gr.Button("📊 Resumen Análisis", size="sm")
        status = gr.Textbox(label="Estado", interactive=False)

        # Crear pestañas para chat y análisis
        with gr.Tabs():
            with gr.Tab("💬 Chat RAG"):
                chatbot = gr.Chatbot(height=400)
                user_input = gr.Textbox(label="Tu pregunta (español)")
                send_btn = gr.Button("Enviar")
            
            with gr.Tab("📈 Análisis Básico"):
                with gr.Column():
                    analysis_summary = gr.Textbox(
                        label="🎯 Resumen Ejecutivo",
                        lines=8,
                        interactive=False,
                        show_copy_button=True,
                        placeholder="El resumen ejecutivo aparecerá aquí después del análisis..."
                    )
                    analysis_output = gr.Textbox(
                        label="📊 Análisis Detallado", 
                        lines=25,
                        interactive=False,
                        show_copy_button=True,
                        placeholder="Haz clic en 'Analizar Conversación' para ver insights básicos sobre el chat..."
                    )
            
            with gr.Tab("🎯 Análisis Adaptativo"):
                with gr.Column():
                    adaptive_summary = gr.Textbox(
                        label="🎯 Resumen Ejecutivo",
                        lines=8,
                        interactive=False,
                        show_copy_button=True,
                        placeholder="El resumen ejecutivo aparecerá aquí después del análisis..."
                    )
                    adaptive_analysis_output = gr.Textbox(
                        label="📊 Análisis Detallado", 
                        lines=35,
                        interactive=False,
                        show_copy_button=True,
                        placeholder="Haz clic en 'Análisis Adaptativo' para un análisis de dos etapas con agentes especializados..."
                    )

        def do_index(file, embed_prov, embed_mod):
            # Update embedding configuration before indexing
            update_embeddings_config(embed_prov, embed_mod)
            return index_file(file)

        index_btn.click(fn=do_index, inputs=[file_input, embed_provider, embed_model], outputs=[status])
        reindex_btn.click(fn=do_index, inputs=[file_input, embed_provider, embed_model], outputs=[status])
        clear_btn.click(fn=clear_state, inputs=[], outputs=[chatbot, status])
        status_btn.click(fn=check_llm_status, inputs=[], outputs=[status])
        def do_analyze_chat(analysis_prov, analysis_mod, progress=gr.Progress()):
            # Update analysis configuration before running
            update_analysis_config(analysis_prov, analysis_mod)
            return analyze_chat(progress)
        
        def do_analyze_chat_adaptive(analysis_prov, analysis_mod, progress=gr.Progress()):
            # Update analysis configuration before running
            update_analysis_config(analysis_prov, analysis_mod)
            return analyze_chat_adaptive(progress)

        analyze_btn.click(
            fn=do_analyze_chat,
            inputs=[analysis_provider, analysis_model],
            outputs=[analysis_summary, analysis_output],
            show_progress=True
        )
        adaptive_btn.click(
            fn=do_analyze_chat_adaptive,
            inputs=[analysis_provider, analysis_model],
            outputs=[adaptive_summary, adaptive_analysis_output],
            show_progress=True
        )
        summary_btn.click(fn=get_analysis_summary, inputs=[], outputs=[status])

        # Setup dynamic model dropdown updates
        embed_provider.change(
            fn=update_embedding_models_dropdown,
            inputs=[embed_provider],
            outputs=[embed_model]
        )
        
        analysis_provider.change(
            fn=update_models_dropdown,
            inputs=[analysis_provider], 
            outputs=[analysis_model]
        )
        
        chat_provider.change(
            fn=update_models_dropdown,
            inputs=[chat_provider],
            outputs=[chat_model]
        )
        
        # Refresh button handlers
        embed_refresh_btn.click(
            fn=refresh_embedding_models_dropdown,
            inputs=[embed_provider],
            outputs=[embed_model]
        )
        
        analysis_refresh_btn.click(
            fn=refresh_models_dropdown,
            inputs=[analysis_provider],
            outputs=[analysis_model]
        )
        
        chat_refresh_btn.click(
            fn=refresh_models_dropdown,
            inputs=[chat_provider],
            outputs=[chat_model]
        )

        def on_send(msg, k, chat_prov, chat_mod, use_mmr, lam, fk, senders, dfrom, dto):
            # Initialize chat interface with analysis context if needed
            initialize_chat_interface()
            
            # Update chat processor configuration based on selected provider/model
            update_chat_processor_config(chat_prov, chat_mod)
            
            # normalize sender list
            senders_list = [s.strip() for s in (senders or "").split(',') if s.strip()]
            chat_hist, err = chat(
                msg, int(k), chat_mod, use_mmr, float(lam), int(fk), senders_list or None, dfrom or None, dto or None
            )
            if err:
                return chat_hist, err, gr.update(value="")
            return chat_hist, "", gr.update(value="")

        send_btn.click(
            fn=on_send,
            inputs=[user_input, topk, chat_provider, chat_model, mmr, lambda_box, fetchk, sender_filter, date_from, date_to],
            outputs=[chatbot, status, user_input]
        )
        user_input.submit(
            fn=on_send,
            inputs=[user_input, topk, chat_provider, chat_model, mmr, lambda_box, fetchk, sender_filter, date_from, date_to],
            outputs=[chatbot, status, user_input]
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    logger.info("Lanzando app en http://%s:%s (share=%s)", host, port, share)
    ui.launch(server_name=host, server_port=port, show_api=False, share=share)
