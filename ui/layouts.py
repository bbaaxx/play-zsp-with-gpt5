"""Layout management for the Gradio interface."""

import gradio as gr
from .components import (
    create_file_upload_component,
    create_file_status_component,
    create_chat_interface,
    create_retrieval_controls,
    create_filter_controls,
    create_analysis_panel,
    create_action_buttons,
)


def create_main_interface() -> tuple:
    """Create the main Gradio interface layout."""
    
    with gr.Blocks(
        title="WhatsApp RAG (ES)",
        theme=gr.themes.Soft(),
    ) as demo:
        
        gr.Markdown(
            """
            # 📱 WhatsApp RAG (ES) - Análisis Inteligente de Conversaciones
            
            Sube un archivo TXT exportado de WhatsApp y haz preguntas sobre la conversación.
            Incluye análisis inteligente con detección de patrones, anomalías y mensajes memorables.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # File upload section
                file_input = create_file_upload_component()
                file_status = create_file_status_component()
                
                # Chat interface
                chatbot, chat_input = create_chat_interface()
                
                # Action buttons
                clear_btn, status_btn, submit_btn = create_action_buttons()
                
            with gr.Column(scale=1):
                with gr.Accordion("⚙️ Configuración de recuperación", open=False):
                    top_k, model_dropdown, use_mmr, lambda_param, fetch_k = create_retrieval_controls()
                
                with gr.Accordion("🔍 Filtros avanzados", open=False):
                    senders_filter, date_from, date_to = create_filter_controls()
                
                # Analysis panel
                analysis_output, analyze_btn, analyze_adaptive_btn = create_analysis_panel()
        
        # Visibility controls for MMR parameters
        use_mmr.change(
            fn=lambda mmr: (gr.update(visible=mmr), gr.update(visible=mmr)),
            inputs=[use_mmr],
            outputs=[lambda_param, fetch_k],
        )
    
    return (
        demo,
        file_input,
        file_status,
        chatbot,
        chat_input,
        top_k,
        model_dropdown,
        use_mmr,
        lambda_param,
        fetch_k,
        senders_filter,
        date_from,
        date_to,
        analysis_output,
        analyze_btn,
        analyze_adaptive_btn,
        clear_btn,
        status_btn,
        submit_btn,
    )