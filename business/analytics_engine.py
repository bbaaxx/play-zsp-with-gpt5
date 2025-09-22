"""
Analysis and agent-based trend detection functionality.
"""

from __future__ import annotations

import os
import logging
from typing import Optional
from dataclasses import dataclass

from rag import ChatDataFrame, ChatAnalyzer, AnalysisResult, AdaptiveAnalyzer, AdaptiveAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for analysis operations."""
    github_token: Optional[str] = None
    model_name: Optional[str] = None
    
    def __post_init__(self):
        if self.github_token is None:
            self.github_token = os.environ.get("GITHUB_TOKEN")
        if self.model_name is None:
            self.model_name = os.environ.get("CHAT_MODEL", "gpt-4o-mini")


class AnalyticsEngine:
    """Handles analysis and agent-based trend detection functionality."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize analytics engine with configuration."""
        self.config = config or AnalysisConfig()
        self.last_analysis: Optional[AnalysisResult] = None
        self.last_adaptive_analysis: Optional[AdaptiveAnalysisResult] = None
    
    def _validate_configuration(self) -> Optional[str]:
        """Validate that required configuration is present."""
        if not self.config.github_token:
            return (
                "❌ **Error de configuración**\n\n"
                "Para usar el análisis inteligente, necesitas configurar tu token de GitHub:\n\n"
                "1. Crea un archivo `.env` en la raíz del proyecto\n"
                "2. Agrega la línea: `GITHUB_TOKEN=tu_token_aquí`\n"
                "3. Obtén tu token en: https://github.com/settings/tokens\n\n"
                "El token necesita acceso a GitHub Models para funcionar."
            )
        return None
    
    def _format_basic_analysis_results(self, result: AnalysisResult) -> str:
        """Format basic analysis results for display."""
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
    
    def _format_adaptive_analysis_results(self, result: AdaptiveAnalysisResult) -> str:
        """Format adaptive analysis results for display."""
        output = ["=== ANÁLISIS ADAPTATIVO DE DOS ETAPAS ===\n"]
        
        # Contextos detectados
        output.append("🎯 **CONTEXTOS DETECTADOS:**")
        if result.detected_contexts:
            for i, context in enumerate(result.detected_contexts, 1):
                confidence_text = "🟢 Alta" if context.confidence > 0.7 else "🟡 Media" if context.confidence > 0.4 else "🔴 Baja"
                output.append(f"\n{i}. **{context.category.replace('_', ' ').title()}** - {confidence_text} ({context.confidence:.1%})")
                if context.evidence:
                    output.append(f"   📋 Evidencia:")
                    for evidence_item in context.evidence:
                        output.append(f"     • {evidence_item}")
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
                            # Show complete analysis without truncation
                            output.append(f"   • **{area_name}**:")
                            # Format the analysis with proper indentation
                            analysis_lines = analysis.strip().split('\n')
                            for line in analysis_lines:
                                if line.strip():
                                    output.append(f"     {line}")
                            output.append("")  # Add blank line for readability
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
        output.append(f"- Realizado: {result.analysis_metadata.get('analysis_timestamp', 'N/A')[:19]}")
        
        return "\n".join(output)
    
    def analyze_chat_basic(self, chat_dataframe: ChatDataFrame, progress_callback=None) -> str:
        """Perform basic intelligent chat analysis using smolagents."""
        if progress_callback is not None:
            progress_callback(0, desc="Iniciando análisis...")
        
        if chat_dataframe is None or chat_dataframe.is_empty:
            return "No hay datos de chat cargados para analizar. Primero indexa un archivo."
        
        if progress_callback is not None:
            progress_callback(0.1, desc="Verificando configuración...")
        
        # Check configuration
        config_error = self._validate_configuration()
        if config_error:
            return config_error
        
        try:
            if progress_callback is not None:
                progress_callback(0.2, desc="Configurando analizador...")
            
            # Create analyzer
            analyzer = ChatAnalyzer(llm_model_name=self.config.model_name)
            
            if progress_callback is not None:
                progress_callback(0.3, desc="Ejecutando análisis inteligente... (esto puede tomar algunos minutos)")
            
            # Perform full analysis
            result = analyzer.full_analysis(chat_dataframe)
            self.last_analysis = result
            
            if progress_callback is not None:
                progress_callback(0.8, desc="Procesando resultados...")
                progress_callback(0.9, desc="Formateando resultados...")
            
            formatted_results = self._format_basic_analysis_results(result)
            
            if progress_callback is not None:
                progress_callback(1.0, desc="✅ Análisis completado")
            
            return formatted_results
            
        except Exception as e:
            logger.exception("Error durante el análisis inteligente")
            return f"Error durante el análisis: {e}\n\nVerifica que tengas configurado correctamente el acceso a modelos LLM (variables de entorno GITHUB_TOKEN, etc.)"
    
    def analyze_chat_adaptive(self, chat_dataframe: ChatDataFrame, progress_callback=None) -> str:
        """Perform adaptive two-stage analysis."""
        if progress_callback is not None:
            progress_callback(0, desc="Iniciando análisis adaptativo...")
        
        if chat_dataframe is None or chat_dataframe.is_empty:
            return "No hay datos de chat cargados para analizar. Primero indexa un archivo."
        
        if progress_callback is not None:
            progress_callback(0.1, desc="Verificando configuración...")
        
        # Check configuration
        config_error = self._validate_configuration()
        if config_error:
            return config_error.replace("análisis inteligente", "análisis adaptativo")
        
        try:
            if progress_callback is not None:
                progress_callback(0.2, desc="Configurando analizador adaptativo...")
            
            # Create adaptive analyzer
            adaptive_analyzer = AdaptiveAnalyzer()
            
            if progress_callback is not None:
                progress_callback(0.3, desc="Etapa 1: Ejecutando análisis básico...")
                progress_callback(0.5, desc="Etapa 2: Detectando contextos de conversación...")
                progress_callback(0.7, desc="Etapa 3: Creando agentes especializados...")
                progress_callback(0.85, desc="Etapa 4: Ejecutando análisis especializados...")
            
            # Perform full adaptive analysis
            result = adaptive_analyzer.analyze(chat_dataframe)
            self.last_adaptive_analysis = result
            
            if progress_callback is not None:
                progress_callback(0.95, desc="Formateando resultados...")
            
            formatted_results = self._format_adaptive_analysis_results(result)
            
            if progress_callback is not None:
                progress_callback(1.0, desc="✅ Análisis adaptativo completado")
            
            return formatted_results
            
        except Exception as e:
            logger.exception("Error durante el análisis adaptativo")
            return f"Error durante el análisis adaptativo: {e}\n\nVerifica que tengas configurado correctamente el acceso a modelos LLM."
    
    def get_analysis_summary(self) -> str:
        """Get a quick summary of the last analyses."""
        if self.last_analysis is None and self.last_adaptive_analysis is None:
            return "No hay análisis previo disponible."
        
        summary = ["📊 **Últimos Análisis:**"]
        
        # Basic analysis summary
        if self.last_analysis is not None:
            result = self.last_analysis
            summary.extend([
                "\n🔍 **Análisis Básico:**",
                f"• {len(result.trend_summaries)} tendencias identificadas",
                f"• {len(result.anomalies)} anomalías detectadas", 
                f"• {len(result.quotable_messages)} mensajes memorables",
                f"• Realizado: {result.analysis_metadata.get('analysis_timestamp', 'N/A')[:19]}"
            ])
        
        # Adaptive analysis summary
        if self.last_adaptive_analysis is not None:
            result = self.last_adaptive_analysis
            summary.extend([
                "\n🎯 **Análisis Adaptativo:**",
                f"• {len(result.detected_contexts)} contextos detectados",
                f"• {result.analysis_metadata.get('specialized_agents_used', 0)} agentes especializados",
                f"• {len(result.adaptive_insights)} insights adaptativos",
                f"• Realizado: {result.analysis_metadata.get('analysis_timestamp', 'N/A')[:19]}"
            ])
        
        return "\n".join(summary)
    
    def clear_analysis_state(self):
        """Clear analysis state."""
        self.last_analysis = None
        self.last_adaptive_analysis = None
        logger.info("Estado de análisis limpiado")