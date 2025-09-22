"""UI components for the Gradio interface."""

import gradio as gr
from typing import List, Optional


def create_file_upload_component() -> gr.File:
    """Create the file upload component for WhatsApp chat files."""
    return gr.File(
        label="Archivo de chat de WhatsApp (TXT)",
        file_types=[".txt"],
        type="filepath",
    )


def create_file_status_component() -> gr.Textbox:
    """Create the status textbox for file processing feedback."""
    return gr.Textbox(
        label="Estado",
        interactive=False,
        max_lines=5,
        value="Sube un archivo TXT de WhatsApp para empezar.",
    )


def create_chat_interface() -> tuple[gr.Chatbot, gr.Textbox]:
    """Create the chat interface components."""
    chatbot = gr.Chatbot(
        label="Conversación",
        height=400,
        bubble_full_width=False,
    )
    
    chat_input = gr.Textbox(
        label="Tu pregunta",
        placeholder="Pregunta sobre la conversación de WhatsApp...",
        lines=2,
    )
    
    return chatbot, chat_input


def create_retrieval_controls() -> tuple[gr.Slider, gr.Dropdown, gr.Checkbox, gr.Slider, gr.Slider]:
    """Create controls for retrieval parameters."""
    top_k = gr.Slider(
        label="Top-K (mensajes a recuperar)",
        minimum=1,
        maximum=20,
        value=5,
        step=1,
    )
    
    model_dropdown = gr.Dropdown(
        label="Modelo LLM",
        choices=[
            "gpt-4o",
            "gpt-4o-mini", 
            "meta-llama/Llama-3.2-3B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
        ],
        value="gpt-4o-mini",
    )
    
    use_mmr = gr.Checkbox(
        label="Usar MMR (diversidad)",
        value=True,
    )
    
    lambda_param = gr.Slider(
        label="λ (balance relevancia-diversidad)",
        minimum=0.0,
        maximum=1.0,
        value=0.5,
        step=0.1,
        visible=True,
    )
    
    fetch_k = gr.Slider(
        label="Fetch-K (candidatos para MMR)",
        minimum=5,
        maximum=50,
        value=25,
        step=5,
        visible=True,
    )
    
    return top_k, model_dropdown, use_mmr, lambda_param, fetch_k


def create_filter_controls() -> tuple[gr.CheckboxGroup, gr.Textbox, gr.Textbox]:
    """Create filtering controls for date and sender filters."""
    senders_filter = gr.CheckboxGroup(
        label="Filtrar por remitentes",
        choices=[],
        value=[],
    )
    
    date_from = gr.Textbox(
        label="Fecha desde (YYYY-MM-DD)",
        placeholder="2024-01-01",
    )
    
    date_to = gr.Textbox(
        label="Fecha hasta (YYYY-MM-DD)", 
        placeholder="2024-12-31",
    )
    
    return senders_filter, date_from, date_to


def create_analysis_panel() -> tuple[gr.Textbox, gr.Button, gr.Button]:
    """Create the analysis panel components."""
    analysis_output = gr.Textbox(
        label="Análisis Inteligente",
        lines=15,
        max_lines=30,
        interactive=False,
        placeholder="Los resultados del análisis aparecerán aquí...",
    )
    
    analyze_btn = gr.Button(
        "🧠 Análisis Inteligente",
        variant="secondary",
    )
    
    analyze_adaptive_btn = gr.Button(
        "🎯 Análisis Adaptativo",
        variant="secondary", 
    )
    
    return analysis_output, analyze_btn, analyze_adaptive_btn


def create_action_buttons() -> tuple[gr.Button, gr.Button, gr.Button]:
    """Create the main action buttons."""
    clear_btn = gr.Button(
        "🗑️ Limpiar",
        variant="secondary",
    )
    
    status_btn = gr.Button(
        "📊 Estado LLM",
        variant="secondary",
    )
    
    submit_btn = gr.Button(
        "Enviar",
        variant="primary",
    )
    
    return clear_btn, status_btn, submit_btn