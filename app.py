from __future__ import annotations

import os
import logging
import gradio as gr
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv

from rag.core import (
    parse_whatsapp_txt,
    RAGPipeline,
    build_user_prompt,
    SYSTEM_PROMPT,
)

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

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


STATE = AppState()


def ensure_pipeline() -> RAGPipeline:
    if STATE.pipeline is None:
        STATE.pipeline = RAGPipeline()
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
    pipe = ensure_pipeline()
    pipe.index_messages(messages)
    n_msgs = len(messages)
    n_chunks = len(pipe.chunks)
    size = pipe.vector_store.size() if pipe.vector_store else 0
    backend = (
        "FAISS" if (pipe.vector_store and getattr(pipe.vector_store, "index", None) is not None) else "numpy"
    )
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
    except:
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
        status = gr.Textbox(label="Estado", interactive=False)

        chatbot = gr.Chatbot(height=400)
        user_input = gr.Textbox(label="Tu pregunta (español)")
        send_btn = gr.Button("Enviar")

        def do_index(file):
            return index_file(file)

        index_btn.click(fn=do_index, inputs=[file_input], outputs=[status])
        reindex_btn.click(fn=do_index, inputs=[file_input], outputs=[status])
        clear_btn.click(fn=clear_state, inputs=[], outputs=[chatbot, status])
        status_btn.click(fn=check_llm_status, inputs=[], outputs=[status])

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


