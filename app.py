from __future__ import annotations

import os
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
    STATE.loaded_text = content
    messages = parse_whatsapp_txt(content)
    pipe = ensure_pipeline()
    pipe.index_messages(messages)
    n_msgs = len(messages)
    n_chunks = len(pipe.chunks)
    size = pipe.vector_store.size() if pipe.vector_store else 0
    if n_msgs == 0:
        preview = "\n".join(content.splitlines()[:3])
        return (
            "Indexado OK — mensajes: 0, chunks: 0, tamaño índice: 0\n"
            "No se detectaron mensajes. Verifica que el archivo sea un export estándar de WhatsApp (TXT).\n"
            f"Primeras líneas leídas:\n{preview}"
        )
    return f"Indexado OK — mensajes: {n_msgs}, chunks: {n_chunks}, tamaño índice: {size}"


def clear_state():
    STATE.pipeline = None
    STATE.chat_history = []
    STATE.loaded_text = None
    return [], "Estado limpiado."


def _llm_client() -> Optional[OpenAI]:
    if OpenAI is None:
        return None
    token = os.environ.get("GITHUB_TOKEN")
    base_url = os.environ.get("GH_MODELS_BASE_URL", "https://models.github.ai/inference")
    if not token:
        return None
    # Avoid passing unsupported kwargs through environment-proxies bug in httpx
    # Build minimal client with only api_key and base_url
    return OpenAI(api_key=token, base_url=base_url)  # type: ignore[arg-type]


def chat(user_msg: str, top_k: int, model_name: str) -> Tuple[List[Tuple[str, str]], str]:
    if not user_msg.strip():
        return STATE.chat_history, "Escribe una pregunta."
    pipe = ensure_pipeline()
    retrieved = pipe.retrieve(user_msg, top_k=top_k)
    context = pipe.format_context(retrieved)
    user_prompt = build_user_prompt(context, user_msg)

    client = _llm_client()
    if client is None:
        answer = (
            "No hay cliente LLM configurado (GITHUB_TOKEN ausente). "
            "Sin embargo, el contexto recuperado es:\n" + context
        )
    else:
        try:
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
        except Exception as e:  # pragma: no cover
            answer = f"Error al llamar al modelo: {e}"

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
            model = gr.Textbox(value=os.environ.get("CHAT_MODEL", "openai/gpt-4o"), label="Modelo LLM")
        with gr.Row():
            index_btn = gr.Button("Indexar")
            reindex_btn = gr.Button("Reindexar")
            clear_btn = gr.Button("Limpiar chat")
        status = gr.Textbox(label="Estado", interactive=False)

        chatbot = gr.Chatbot(height=400)
        user_input = gr.Textbox(label="Tu pregunta (español)")
        send_btn = gr.Button("Enviar")

        def do_index(file):
            return index_file(file)

        index_btn.click(fn=do_index, inputs=[file_input], outputs=[status])
        reindex_btn.click(fn=do_index, inputs=[file_input], outputs=[status])
        clear_btn.click(fn=clear_state, inputs=[], outputs=[chatbot, status])

        def on_send(msg, k, m):
            chat_hist, err = chat(msg, int(k), m)
            if err:
                return chat_hist, err
            return chat_hist, ""

        send_btn.click(fn=on_send, inputs=[user_input, topk, model], outputs=[chatbot, status])
        user_input.submit(fn=on_send, inputs=[user_input, topk, model], outputs=[chatbot, status])

    return demo


if __name__ == "__main__":
    ui = build_ui()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    ui.launch(server_name=host, server_port=port, show_api=False, share=share)


