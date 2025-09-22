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
logger = logging.getLogger("app")


class AppState:
    def __init__(self) -> None:
        self.file_manager = FileManager()
        self.chat_processor = ChatProcessor()
        self.analytics_engine = AnalyticsEngine()
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

        def do_index(file):
            return index_file(file)

        index_btn.click(fn=do_index, inputs=[file_input], outputs=[status])
        reindex_btn.click(fn=do_index, inputs=[file_input], outputs=[status])
        clear_btn.click(fn=clear_state, inputs=[], outputs=[chatbot, status])
        status_btn.click(fn=check_llm_status, inputs=[], outputs=[status])
        analyze_btn.click(
            fn=analyze_chat,
            inputs=[],
            outputs=[analysis_summary, analysis_output],
            show_progress=True
        )
        adaptive_btn.click(
            fn=analyze_chat_adaptive,
            inputs=[],
            outputs=[adaptive_summary, adaptive_analysis_output],
            show_progress=True
        )
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
