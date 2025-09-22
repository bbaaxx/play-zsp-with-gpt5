from __future__ import annotations

import os
import logging
import gradio as gr
from typing import List, Tuple, Optional
from dotenv import load_dotenv

from rag.core import (
    parse_whatsapp_txt,
    RAGPipeline,
    build_user_prompt,
    SYSTEM_PROMPT,
)
from rag import (
    ChatDataFrame,
    ChatAnalyzer,
    AnalysisResult,
)

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

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


load_dotenv()

# Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


class AppState:
    def __init__(self) -> None:
        self.pipeline: Optional[RAGPipeline] = None
        self.chat_history: List[Tuple[str, str]] = []
        self.loaded_text: Optional[str] = None
        self.chat_dataframe: Optional[ChatDataFrame] = None
        self.last_analysis: Optional[AnalysisResult] = None


STATE = AppState()


def ensure_pipeline() -> RAGPipeline:
    if STATE.pipeline is None:
        # Read vector backend configuration from environment
        vector_backend = os.environ.get("VECTOR_BACKEND", "faiss").lower()
        vector_store_kwargs = {}
        
        if vector_backend == "qdrant":
            vector_store_kwargs.update({
                "url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
                "api_key": os.environ.get("QDRANT_API_KEY"),
                "collection_name": os.environ.get("QDRANT_COLLECTION_NAME", "whatsapp_rag"),
            })
            # Remove None values
            vector_store_kwargs = {k: v for k, v in vector_store_kwargs.items() if v is not None}
        
        STATE.pipeline = RAGPipeline(
            vector_backend=vector_backend,
            **vector_store_kwargs
        )
    return STATE.pipeline


def _extract_path(file_obj) -> Optional[str]:
    if file_obj is None:
        return None
    # gr.File with type="filepath" returns a string path
    if isinstance(file_obj, str):
        return file_obj
    # Some gradio versions return a dict with 'name'
    if isinstance(file_obj, dict) and "name" in file_obj:
        return str(file_obj["name"])  # type: ignore
    # File-like object with .name
    if hasattr(file_obj, "name"):
        try:
            return str(getattr(file_obj, "name"))
        except Exception:
            return None
    return None


def index_file(file_obj) -> str:
    path = _extract_path(file_obj)
    if not path:
        return "Sube un archivo TXT de WhatsApp primero."
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return f"No se pudo leer el archivo: {e}"
    logger.info("Indexando archivo: %s (tamaño=%d bytes)", path, len(content))
    STATE.loaded_text = content
    messages = parse_whatsapp_txt(content)
    
    # Crear ChatDataFrame para análisis
    STATE.chat_dataframe = ChatDataFrame(messages)
    
    pipe = ensure_pipeline()
    pipe.index_messages(messages)
    n_msgs = len(messages)
    n_chunks = len(pipe.chunks)
    size = pipe.vector_store.size() if pipe.vector_store else 0
    # Determine backend type for display
    if pipe.vector_store:
        backend = type(pipe.vector_store).__name__.replace("VectorStore", "").replace("InMemory", "")
    else:
        backend = "none"
    logger.info(
        "Indexado completado — mensajes=%d, chunks=%d, indice=%d, backend=%s",
        n_msgs,
        n_chunks,
        size,
        backend,
    )
    if n_msgs == 0:
        preview = "\n".join(content.splitlines()[:3])
        return (
            "Indexado OK — mensajes: 0, chunks: 0, tamaño índice: 0\n"
            "No se detectaron mensajes. Verifica que el archivo sea un export estándar de WhatsApp (TXT).\n"
            f"Primeras líneas leídas:\n{preview}"
        )
    return f"Indexado OK — mensajes: {n_msgs}, chunks: {n_chunks}, tamaño índice: {size} (backend: {backend})"


def clear_state():
    STATE.pipeline = None
    STATE.chat_history = []
    STATE.loaded_text = None
    STATE.chat_dataframe = None
    STATE.last_analysis = None
    return [], "Estado limpiado."


def check_llm_status():
    """Check the status of all LLM and embedding providers."""
    pipe = ensure_pipeline()
    
    # LLM providers
    llm_providers = pipe.llm_manager.list_providers()
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
    embedder = pipe.embedder
    
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


def _legacy_llm_client() -> Optional[OpenAI]:
    """Legacy LLM client for backwards compatibility."""
    if OpenAI is None:
        return None
    token = os.environ.get("GITHUB_TOKEN")
    base_url = os.environ.get("GH_MODELS_BASE_URL", "https://models.github.ai/inference")
    if not token:
        return None
    # Avoid passing unsupported kwargs through environment-proxies bug in httpx
    # Build minimal client with only api_key and base_url
    logger.info("Creando cliente LLM legacy (base_url=%s)", base_url)
    return OpenAI(api_key=token, base_url=base_url)  # type: ignore[arg-type]


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
    if not user_msg.strip():
        return STATE.chat_history, "Escribe una pregunta."
    pipe = ensure_pipeline()
    logger.info(
        "Consulta recibida — top_k=%s, MMR=%s, λ=%.2f, fetch_k=%s, senders=%s, dfrom=%s, dto=%s",
        top_k, use_mmr, lambda_, fetch_k, senders, date_from_iso, date_to_iso,
    )
    retrieved = pipe.retrieve(
        user_msg,
        top_k=top_k,
        use_mmr=use_mmr,
        fetch_k=fetch_k,
        lambda_=lambda_,
        senders=senders,
        date_from_iso=date_from_iso,
        date_to_iso=date_to_iso,
    )
    context = pipe.format_context(retrieved)
    logger.info("Recuperados %d fragmentos", len(retrieved))

    # Try using the new LLM manager first
    try:
        answer = pipe.generate_answer(context, user_msg)
        logger.info("Respuesta LLM generada (%d chars)", len(answer))
    except Exception as e:
        logger.warning("Fallo en proveedores LLM nuevos: %s", e)
        
        # Fallback to legacy client for backwards compatibility
        client = _legacy_llm_client()
        if client is None:
            answer = (
                f"No hay proveedores LLM disponibles. Error: {e}\n"
                "Contexto recuperado:\n" + context
            )
        else:
            try:
                user_prompt = build_user_prompt(context, user_msg)
                resp = client.chat.completions.create(
                    model=model_name or os.environ.get("CHAT_MODEL", "openai/gpt-4o"),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=800,
                )
                answer = resp.choices[0].message.content or "(sin contenido)"
                logger.info("Respuesta LLM legacy generada (%d chars)", len(answer))
            except Exception as legacy_e:
                answer = f"Error en todos los proveedores LLM: {legacy_e}"
                logger.exception("Fallo en cliente LLM legacy")

    STATE.chat_history.append((user_msg, answer))
    return STATE.chat_history, ""


def analyze_chat() -> str:
    """Realiza un análisis inteligente del chat usando smolagents."""
    if STATE.chat_dataframe is None or STATE.chat_dataframe.is_empty:
        return "No hay datos de chat cargados para analizar. Primero indexa un archivo."
    
    # Check if GitHub token is configured
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        return ("❌ **Error de configuración**\n\n"
                "Para usar el análisis inteligente, necesitas configurar tu token de GitHub:\n\n"
                "1. Crea un archivo `.env` en la raíz del proyecto\n"
                "2. Agrega la línea: `GITHUB_TOKEN=tu_token_aquí`\n"
                "3. Obtén tu token en: https://github.com/settings/tokens\n\n"
                "El token necesita acceso a GitHub Models para funcionar.")
    
    try:
        # Obtener configuración del modelo LLM
        model_name = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
        analyzer = ChatAnalyzer(llm_model_name=model_name)
        
        # Realizar análisis completo
        result = analyzer.full_analysis(STATE.chat_dataframe)
        STATE.last_analysis = result
        
        # Formatear resultados para mostrar
        output = ["=== ANÁLISIS INTELIGENTE DE CONVERSACIÓN ===\n"]
        
        # Tendencias identificadas
        if result.trend_summaries:
            output.append("🔍 **TENDENCIAS Y PATRONES:**")
            for i, trend in enumerate(result.trend_summaries, 1):
                output.append(f"\n{i}. **{trend.trend_type.upper()}** (confianza: {trend.confidence_score:.1%})")
                output.append(f"   {trend.description}")
                if trend.time_period:
                    output.append(f"   Período: {trend.time_period}")
                if trend.participants:
                    output.append(f"   Participantes: {', '.join(trend.participants)}")
        else:
            output.append("🔍 **TENDENCIAS Y PATRONES:** No se detectaron patrones significativos.")
        
        # Anomalías detectadas
        output.append("\n⚠️ **COMPORTAMIENTOS INUSUALES:**")
        if result.anomalies:
            for i, anomaly in enumerate(result.anomalies, 1):
                severity_icon = "🟥" if anomaly.severity == "high" else "🟨" if anomaly.severity == "medium" else "🟩"
                output.append(f"\n{i}. {severity_icon} **{anomaly.anomaly_type.upper()}**")
                output.append(f"   {anomaly.description}")
                if anomaly.participant:
                    output.append(f"   Participante: {anomaly.participant}")
        else:
            output.append("No se detectaron comportamientos inusuales.")
        
        # Mensajes memorables
        output.append("\n💬 **MENSAJES MEMORABLES:**")
        if result.quotable_messages:
            for i, quote in enumerate(result.quotable_messages, 1):
                type_icon = "😂" if quote.quote_type == "funny" else "💭" if quote.quote_type == "insightful" else "❤️" if quote.quote_type == "emotional" else "⭐"
                output.append(f"\n{i}. {type_icon} **{quote.sender}** ({quote.timestamp.strftime('%Y-%m-%d %H:%M')})")
                output.append(f"   \"{quote.message}\"")
                output.append(f"   Relevancia: {quote.relevance_score:.1%} | Tipo: {quote.quote_type}")
                if quote.context:
                    output.append(f"   Contexto: {quote.context}")
        else:
            output.append("No se encontraron mensajes especialmente memorables.")
        
        # Metadatos
        output.append("\n📊 **METADATOS DEL ANÁLISIS:**")
        output.append(f"- Mensajes analizados: {result.analysis_metadata.get('total_messages_analyzed', 0)}")
        output.append(f"- Modelo usado: {result.analysis_metadata.get('model_used', 'N/A')}")
        output.append(f"- Análisis realizado: {result.analysis_metadata.get('analysis_timestamp', 'N/A')}")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.exception("Error durante el análisis inteligente")
        return f"Error durante el análisis: {e}\n\nVerifica que tengas configurado correctamente el acceso a modelos LLM (variables de entorno GITHUB_TOKEN, etc.)"


def get_analysis_summary() -> str:
    """Obtiene un resumen rápido del último análisis."""
    if STATE.last_analysis is None:
        return "No hay análisis previo disponible."
    
    result = STATE.last_analysis
    summary = [
        "📊 **Último Análisis:**",
        f"• {len(result.trend_summaries)} tendencias identificadas",
        f"• {len(result.anomalies)} anomalías detectadas", 
        f"• {len(result.quotable_messages)} mensajes memorables",
        f"• Realizado: {result.analysis_metadata.get('analysis_timestamp', 'N/A')[:19]}"
    ]
    
    return "\n".join(summary)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="WhatsApp RAG (ES)") as demo:
        gr.Markdown("""
        ### WhatsApp RAG (ES)
        Carga un TXT exportado de WhatsApp, indexa y consulta en español. Solo se envían al LLM los fragmentos recuperados.
        """)

        with gr.Row():
            file_input = gr.File(label="Archivo TXT de WhatsApp", file_count="single", type="filepath")
        with gr.Row():
            topk = gr.Slider(1, 10, value=5, step=1, label="Top-k")
            model = gr.Textbox(value=os.environ.get("CHAT_MODEL", "openai/gpt-4o"), label="Modelo LLM (legacy)")
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
            summary_btn = gr.Button("📊 Resumen Análisis", size="sm")
        status = gr.Textbox(label="Estado", interactive=False)

        # Crear pestañas para chat y análisis
        with gr.Tabs():
            with gr.Tab("💬 Chat RAG"):
                chatbot = gr.Chatbot(height=400)
                user_input = gr.Textbox(label="Tu pregunta (español)")
                send_btn = gr.Button("Enviar")
            
            with gr.Tab("📈 Análisis Inteligente"):
                analysis_output = gr.Textbox(
                    label="Resultados del Análisis", 
                    lines=20, 
                    max_lines=30,
                    interactive=False,
                    placeholder="Haz clic en 'Analizar Conversación' para ver insights inteligentes sobre el chat..."
                )

        def do_index(file):
            return index_file(file)

        index_btn.click(fn=do_index, inputs=[file_input], outputs=[status])
        reindex_btn.click(fn=do_index, inputs=[file_input], outputs=[status])
        clear_btn.click(fn=clear_state, inputs=[], outputs=[chatbot, status])
        status_btn.click(fn=check_llm_status, inputs=[], outputs=[status])
        analyze_btn.click(fn=analyze_chat, inputs=[], outputs=[analysis_output])
        summary_btn.click(fn=get_analysis_summary, inputs=[], outputs=[status])

        def on_send(msg, k, m, use_mmr, lam, fk, senders, dfrom, dto):
            # normalize sender list
            senders_list = [s.strip() for s in (senders or "").split(',') if s.strip()]
            chat_hist, err = chat(
                msg, int(k), m, use_mmr, float(lam), int(fk), senders_list or None, dfrom or None, dto or None
            )
            if err:
                return chat_hist, err, gr.update(value="")
            return chat_hist, "", gr.update(value="")

        send_btn.click(
            fn=on_send,
            inputs=[user_input, topk, model, mmr, lambda_box, fetchk, sender_filter, date_from, date_to],
            outputs=[chatbot, status, user_input]
        )
        user_input.submit(
            fn=on_send,
            inputs=[user_input, topk, model, mmr, lambda_box, fetchk, sender_filter, date_from, date_to],
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


