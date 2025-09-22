"""Event handlers for the Gradio interface."""

import os
import logging
import gradio as gr
from typing import List, Tuple, Optional

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
    AdaptiveAnalyzer,
    AdaptiveAnalysisResult,
)

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


logger = logging.getLogger("app")


class AppState:
    def __init__(self) -> None:
        self.pipeline: Optional[RAGPipeline] = None
        self.chat_history: List[Tuple[str, str]] = []
        self.loaded_text: Optional[str] = None
        self.chat_dataframe: Optional[ChatDataFrame] = None
        self.last_analysis: Optional[AnalysisResult] = None
        self.last_adaptive_analysis: Optional[AdaptiveAnalysisResult] = None


STATE = AppState()


def ensure_pipeline() -> RAGPipeline:
    """Ensure the RAG pipeline is initialized."""
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
    """Extract file path from Gradio file object."""
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
    """Index a WhatsApp chat file."""
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
    """Clear application state."""
    STATE.pipeline = None
    STATE.chat_history = []
    STATE.loaded_text = None
    STATE.chat_dataframe = None
    STATE.last_analysis = None
    STATE.last_adaptive_analysis = None
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
    base_url = os.environ.get("GH_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
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
    """Handle chat interaction."""
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


def analyze_chat(progress=gr.Progress()) -> str:
    """Realiza un análisis inteligente del chat usando smolagents."""
    progress(0, desc="Iniciando análisis...")
    
    if STATE.chat_dataframe is None or STATE.chat_dataframe.is_empty:
        return "No hay datos de chat cargados para analizar. Primero indexa un archivo."
    
    progress(0.1, desc="Verificando configuración...")
    
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
        progress(0.2, desc="Configurando analizador...")
        
        # Obtener configuración del modelo LLM
        model_name = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
        analyzer = ChatAnalyzer(llm_model_name=model_name)
        
        progress(0.3, desc="Ejecutando análisis inteligente... (esto puede tomar algunos minutos)")
        
        # Realizar análisis completo
        result = analyzer.full_analysis(STATE.chat_dataframe)
        STATE.last_analysis = result
        
        progress(0.8, desc="Procesando resultados...")
        
        progress(0.9, desc="Formateando resultados...")
        
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
        
        progress(1.0, desc="✅ Análisis completado")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.exception("Error durante el análisis inteligente")
        return f"Error durante el análisis: {e}\n\nVerifica que tengas configurado correctamente el acceso a modelos LLM (variables de entorno GITHUB_TOKEN, etc.)"


def analyze_chat_adaptive(progress=gr.Progress()) -> str:
    """Realiza análisis adaptativo de dos etapas."""
    progress(0, desc="Iniciando análisis adaptativo...")
    
    if STATE.chat_dataframe is None or STATE.chat_dataframe.is_empty:
        return "No hay datos de chat cargados para analizar. Primero indexa un archivo."
    
    progress(0.1, desc="Verificando configuración...")
    
    # Check if GitHub token is configured
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        return ("❌ **Error de configuración**\n\n"
                "Para usar el análisis adaptativo, necesitas configurar tu token de GitHub:\n\n"
                "1. Crea un archivo `.env` en la raíz del proyecto\n"
                "2. Agrega la línea: `GITHUB_TOKEN=tu_token_aquí`\n"
                "3. Obtén tu token en: https://github.com/settings/tokens\n\n"
                "El token necesita acceso a GitHub Models para funcionar.")
    
    try:
        progress(0.2, desc="Configurando analizador adaptativo...")
        
        # Crear analizador adaptativo
        adaptive_analyzer = AdaptiveAnalyzer()
        
        progress(0.3, desc="Etapa 1: Ejecutando análisis básico...")
        progress(0.5, desc="Etapa 2: Detectando contextos de conversación...")
        progress(0.7, desc="Etapa 3: Creando agentes especializados...")
        progress(0.85, desc="Etapa 4: Ejecutando análisis especializados...")
        
        # Realizar análisis adaptativo completo
        result = adaptive_analyzer.analyze(STATE.chat_dataframe)
        STATE.last_adaptive_analysis = result
        
        progress(0.95, desc="Formateando resultados...")
        
        # Formatear resultados
        output = ["=== ANÁLISIS ADAPTATIVO DE DOS ETAPAS ===\n"]
        
        # Contextos detectados
        output.append("🎯 **CONTEXTOS DETECTADOS:**")
        if result.detected_contexts:
            for i, context in enumerate(result.detected_contexts, 1):
                confidence_text = "🟢 Alta" if context.confidence > 0.7 else "🟡 Media" if context.confidence > 0.4 else "🔴 Baja"
                output.append(f"\n{i}. **{context.category.replace('_', ' ').title()}** - {confidence_text} ({context.confidence:.1%})")
                if context.evidence:
                    output.append(f"   📋 Evidencia: {'; '.join(context.evidence[:2])}")
        else:
            output.append("No se detectaron contextos específicos.")
        
        # Análisis especializados
        output.append("\n🔬 **ANÁLISIS ESPECIALIZADOS:**")
        if result.specialized_analyses:
            for context_type, analyses in result.specialized_analyses.items():
                context_name = context_type.replace('_', ' ').title()
                output.append(f"\n📊 **{context_name}:**")
                
                if isinstance(analyses, dict) and "error" not in analyses:
                    for focus_area, analysis in analyses.items():
                        if analysis and isinstance(analysis, str):
                            area_name = focus_area.replace('_', ' ').title()
                            # Tomar primera línea significativa del análisis
                            lines = [line.strip() for line in analysis.split('\n') if line.strip()]
                            preview = lines[0] if lines else analysis[:100]
                            output.append(f"   • **{area_name}**: {preview[:200]}...")
                elif "error" in analyses:
                    output.append(f"   ⚠️ Error en análisis: {analyses['error']}")
        else:
            output.append("No se generaron análisis especializados.")
        
        # Insights adaptativos
        output.append("\n💡 **INSIGHTS ADAPTATIVOS:**")
        if result.adaptive_insights:
            for insight in result.adaptive_insights:
                output.append(f"\n{insight}")
        else:
            output.append("No se generaron insights adaptativos.")
        
        # Análisis básico (resumen)
        basic = result.basic_analysis
        output.append("\n📈 **RESUMEN ANÁLISIS BÁSICO:**")
        output.append(f"• {len(basic.trend_summaries)} tendencias identificadas")
        output.append(f"• {len(basic.anomalies)} anomalías detectadas")
        output.append(f"• {len(basic.quotable_messages)} mensajes memorables")
        
        # Metadatos
        output.append("\n📊 **METADATOS:**")
        output.append(f"- Contextos detectados: {result.analysis_metadata.get('total_contexts_detected', 0)}")
        output.append(f"- Agentes especializados: {result.analysis_metadata.get('specialized_agents_used', 0)}")
        output.append(f"- Versión análisis: {result.analysis_metadata.get('analysis_version', 'N/A')}")
        output.append(f"- Realizado: {result.analysis_metadata.get('analysis_timestamp', 'N/A')}")
        
        progress(1.0, desc="✅ Análisis adaptativo completado")
        
        return "\n".join(output)
        
    except Exception as e:
        logger.exception("Error durante el análisis adaptativo")
        return f"Error durante el análisis adaptativo: {e}\n\nVerifica que tengas configurado correctamente el acceso a modelos LLM."


def _get_updated_senders(file_obj) -> List[str]:
    """Get list of unique senders from the uploaded file."""
    if STATE.chat_dataframe is None:
        return []
    return STATE.chat_dataframe.get_unique_senders()


def update_senders_filter(file_obj) -> gr.CheckboxGroup:
    """Update sender filter options when file is uploaded."""
    senders = _get_updated_senders(file_obj)
    return gr.CheckboxGroup(choices=senders, value=[])