from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re

from .analysis import ChatDataFrame
from .smart_analysis import ChatAnalyzer, AnalysisResult
from .llm_providers import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class ContextCategory:
    """Categoría de contexto detectada en el chat."""
    category: str
    confidence: float
    evidence: List[str]
    characteristics: Dict[str, Any]


@dataclass
class SpecializedAgent:
    """Agente especializado para análisis específico del contexto."""
    name: str
    context_category: str
    system_prompt: str
    analysis_focus: List[str]
    specialized_prompts: Dict[str, str]


@dataclass
class AdaptiveAnalysisResult:
    """Resultado del análisis adaptativo de dos etapas."""
    basic_analysis: AnalysisResult
    detected_contexts: List[ContextCategory]
    specialized_analyses: Dict[str, Dict[str, Any]]
    adaptive_insights: List[str]
    analysis_metadata: Dict[str, Any]


class ContextDetector:
    """Detector de contexto para identificar el tipo de conversación."""
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        self.llm_manager = llm_manager or LLMManager()
        
        # Patrones predefinidos para detección de contexto
        self.context_patterns = {
            "friends_casual": {
                "keywords": ["jaja", "lol", "xd", "amigo", "bro", "man", "vamos", "fiesta", "salir"],
                "patterns": [r"ja+", r"jeje", r"meme", r"gif"],
                "time_patterns": ["evening", "weekend"],
            },
            "romantic_couple": {
                "keywords": ["amor", "mi amor", "cariño", "bebé", "corazón", "te amo", "extraño", "beso"],
                "patterns": [r"❤️", r"😘", r"💕", r"te.*amo", r"mi.*amor"],
                "time_patterns": ["late_night", "early_morning"],
            },
            "family": {
                "keywords": ["papá", "mamá", "hijo", "hija", "hermano", "hermana", "familia", "casa", "cena"],
                "patterns": [r"papá|papa", r"mamá|mama", r"mi.*hijo", r"mi.*hija"],
                "time_patterns": ["morning", "evening"],
            },
            "work_professional": {
                "keywords": ["trabajo", "reunión", "proyecto", "cliente", "jefe", "oficina", "deadline", "reporte"],
                "patterns": [r"reunión", r"proyecto", r"informe", r"excel", r"email"],
                "time_patterns": ["business_hours"],
            },
            "gaming": {
                "keywords": ["juego", "game", "partida", "nivel", "win", "gg", "noob", "lag", "fps"],
                "patterns": [r"gg", r"ez", r"1v1", r"rank", r"play"],
                "time_patterns": ["evening", "night"],
            },
            "support_emotional": {
                "keywords": ["triste", "mal", "problema", "ayuda", "apoyo", "entiendo", "fuerza", "ánimo"],
                "patterns": [r"está.*mal", r"me.*siento", r"problema", r"ayuda"],
                "time_patterns": ["any"],
            },
            "planning_organizing": {
                "keywords": ["plan", "cuando", "donde", "hora", "lugar", "quedamos", "vamos", "organizamos"],
                "patterns": [r"qué.*hora", r"dónde", r"cuándo", r"plan"],
                "time_patterns": ["any"],
            },
        }
    
    def detect_contexts(self, data_frame: ChatDataFrame, basic_analysis: AnalysisResult) -> List[ContextCategory]:
        """
        Detecta los contextos principales de la conversación basándose en análisis básico.
        
        Args:
            data_frame: DataFrame con mensajes del chat
            basic_analysis: Resultado del análisis básico
            
        Returns:
            Lista de contextos detectados
        """
        if data_frame.is_empty:
            return []
        
        # Paso 1: Análisis basado en patrones
        pattern_contexts = self._detect_pattern_contexts(data_frame)
        
        # Paso 2: Análisis con LLM para mayor precisión
        llm_contexts = self._detect_llm_contexts(data_frame, basic_analysis)
        
        # Combinar y rankear contextos
        all_contexts = pattern_contexts + llm_contexts
        merged_contexts = self._merge_and_rank_contexts(all_contexts)
        
        return merged_contexts
    
    def _detect_pattern_contexts(self, data_frame: ChatDataFrame) -> List[ContextCategory]:
        """Detecta contextos usando patrones predefinidos."""
        contexts = []
        df = data_frame.df
        all_messages = " ".join(df['message'].astype(str).str.lower())
        
        for context_type, config in self.context_patterns.items():
            score = 0
            evidence = []
            total_checks = 0
            
            # Verificar palabras clave
            keyword_matches = 0
            for keyword in config["keywords"]:
                count = all_messages.count(keyword.lower())
                if count > 0:
                    keyword_matches += count
                    evidence.append(f"Palabra clave '{keyword}': {count} veces")
            
            if len(config["keywords"]) > 0:
                keyword_score = min(keyword_matches / len(config["keywords"]) / 10, 0.5)
                score += keyword_score
                total_checks += 1
            
            # Verificar patrones regex
            pattern_matches = 0
            for pattern in config["patterns"]:
                matches = len(re.findall(pattern, all_messages, re.IGNORECASE))
                if matches > 0:
                    pattern_matches += matches
                    evidence.append(f"Patrón '{pattern}': {matches} coincidencias")
            
            if len(config["patterns"]) > 0:
                pattern_score = min(pattern_matches / len(config["patterns"]) / 10, 0.3)
                score += pattern_score
                total_checks += 1
            
            # Verificar patrones temporales
            time_score = self._check_time_patterns(df, config["time_patterns"])
            if time_score > 0:
                score += time_score * 0.2
                evidence.append(f"Patrón temporal coincide")
                total_checks += 1
            
            # Calcular confianza final
            if total_checks > 0:
                confidence = score / max(total_checks * 0.5, 1.0)
                confidence = min(confidence, 1.0)
                
                if confidence > 0.1:  # Umbral mínimo
                    contexts.append(ContextCategory(
                        category=context_type,
                        confidence=confidence,
                        evidence=evidence,
                        characteristics={"pattern_based": True}
                    ))
        
        return contexts
    
    def _detect_llm_contexts(self, data_frame: ChatDataFrame, basic_analysis: AnalysisResult) -> List[ContextCategory]:
        """Detecta contextos usando LLM para análisis más sofisticado."""
        try:
            # Preparar contexto del análisis básico
            trends_summary = []
            for trend in basic_analysis.trend_summaries:
                trends_summary.append(f"{trend.trend_type}: {trend.description}")
            
            # Muestra de mensajes representativos
            df = data_frame.df
            sample_messages = []
            if not df.empty:
                # Tomar una muestra estratificada por tiempo
                sample_size = min(50, len(df))
                sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
                
                for _, row in sample_df.iterrows():
                    sample_messages.append({
                        "sender": row['author'],
                        "message": row['message'][:200],  # Limitar longitud
                        "time_hour": row['timestamp'].hour
                    })
            
            context_info = {
                "chat_stats": {
                    "total_messages": len(df),
                    "unique_participants": df['author'].nunique(),
                    "date_range_days": (df['timestamp'].max() - df['timestamp'].min()).days if not df.empty else 0,
                },
                "detected_trends": trends_summary,
                "sample_messages": sample_messages
            }
            
            prompt = f"""
            Analiza esta conversación de WhatsApp y determina los tipos de relación/contexto más probables.
            
            Información de la conversación:
            {json.dumps(context_info, indent=2, default=str)}
            
            Evalúa la probabilidad de que esta conversación pertenezca a cada una de estas categorías:
            
            1. **friends_casual**: Amigos conversando de forma casual/divertida
            2. **romantic_couple**: Pareja romántica/matrimonio
            3. **family**: Familia (padres-hijos, hermanos, etc.)
            4. **work_professional**: Contexto laboral/profesional
            5. **gaming**: Discusiones sobre videojuegos
            6. **support_emotional**: Apoyo emocional/conversaciones profundas
            7. **planning_organizing**: Planificación de eventos/actividades
            
            Para cada categoría que consideres relevante (confianza > 0.2), responde en formato JSON:
            [
              {{
                "category": "nombre_categoria",
                "confidence": 0.85,
                "evidence": ["razón 1", "razón 2", "razón 3"],
                "characteristics": {{"key": "value"}}
              }}
            ]
            """
            
            response = self.llm_manager.generate_response([
                {"role": "system", "content": "Eres un experto analizando contextos de conversación. Responde solo en JSON válido."},
                {"role": "user", "content": prompt}
            ], temperature=0.1, max_tokens=1500)
            
            # Parsear respuesta JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                contexts_data = json.loads(json_match.group())
                
                contexts = []
                for ctx_data in contexts_data:
                    if isinstance(ctx_data, dict) and ctx_data.get("confidence", 0) > 0.2:
                        contexts.append(ContextCategory(
                            category=ctx_data.get("category", "unknown"),
                            confidence=float(ctx_data.get("confidence", 0.0)),
                            evidence=ctx_data.get("evidence", []),
                            characteristics=ctx_data.get("characteristics", {"llm_based": True})
                        ))
                
                return contexts
                
        except Exception as e:
            logger.warning(f"Error en detección LLM de contexto: {e}")
        
        return []
    
    def _check_time_patterns(self, df, time_patterns: List[str]) -> float:
        """Verifica patrones temporales en los mensajes."""
        if df.empty or not time_patterns or "any" in time_patterns:
            return 0.1
        
        scores = []
        
        for pattern in time_patterns:
            if pattern == "business_hours":
                business_msgs = df[(df['timestamp'].dt.hour >= 9) & (df['timestamp'].dt.hour <= 17)]
                score = len(business_msgs) / len(df)
            elif pattern == "evening":
                evening_msgs = df[(df['timestamp'].dt.hour >= 18) & (df['timestamp'].dt.hour <= 23)]
                score = len(evening_msgs) / len(df)
            elif pattern == "late_night":
                night_msgs = df[(df['timestamp'].dt.hour >= 22) | (df['timestamp'].dt.hour <= 2)]
                score = len(night_msgs) / len(df)
            elif pattern == "early_morning":
                morning_msgs = df[(df['timestamp'].dt.hour >= 6) & (df['timestamp'].dt.hour <= 9)]
                score = len(morning_msgs) / len(df)
            elif pattern == "weekend":
                weekend_msgs = df[df['timestamp'].dt.weekday >= 5]  # Sábado=5, Domingo=6
                score = len(weekend_msgs) / len(df)
            else:
                score = 0
            
            scores.append(score)
        
        return max(scores) if scores else 0
    
    def _merge_and_rank_contexts(self, contexts: List[ContextCategory]) -> List[ContextCategory]:
        """Combina contextos duplicados y los ordena por confianza."""
        # Agrupar por categoría
        context_groups = {}
        for ctx in contexts:
            if ctx.category in context_groups:
                # Promediar confianza y combinar evidencia
                existing = context_groups[ctx.category]
                combined_confidence = (existing.confidence + ctx.confidence) / 2
                combined_evidence = existing.evidence + ctx.evidence
                combined_characteristics = {**existing.characteristics, **ctx.characteristics}
                
                context_groups[ctx.category] = ContextCategory(
                    category=ctx.category,
                    confidence=combined_confidence,
                    evidence=combined_evidence,
                    characteristics=combined_characteristics
                )
            else:
                context_groups[ctx.category] = ctx
        
        # Ordenar por confianza
        merged_contexts = list(context_groups.values())
        merged_contexts.sort(key=lambda x: x.confidence, reverse=True)
        
        # Return all contexts without limiting
        return merged_contexts


class SpecializedAgentFactory:
    """Fábrica de agentes especializados según el contexto detectado."""
    
    @staticmethod
    def create_agents(detected_contexts: List[ContextCategory]) -> List[SpecializedAgent]:
        """Crea agentes especializados basados en contextos detectados."""
        agents = []
        
        for context in detected_contexts:
            if context.confidence < 0.3:  # Solo crear agentes para contextos con alta confianza
                continue
                
            agent = SpecializedAgentFactory._create_agent_for_context(context)
            if agent:
                agents.append(agent)
        
        return agents
    
    @staticmethod
    def _create_agent_for_context(context: ContextCategory) -> Optional[SpecializedAgent]:
        """Crea un agente especializado para un contexto específico."""
        
        agent_configs = {
            "friends_casual": {
                "name": "Analizador de Amistad Casual",
                "system_prompt": """Eres un experto en analizar dinámicas de amistad casual y conversaciones informales.
                Te especializas en detectar patrones de humor, referencias compartidas, planes sociales,
                bromas internas y la evolución de amistades a través de mensajes de texto.""",
                "analysis_focus": ["humor_patterns", "shared_references", "social_planning", "friendship_dynamics"],
                "specialized_prompts": {
                    "humor_patterns": "Identifica y analiza los patrones de humor en esta conversación entre amigos",
                    "shared_references": "Detecta referencias compartidas, bromas internas y memes",
                    "social_planning": "Analiza cómo los amigos planifican actividades y eventos sociales",
                    "friendship_dynamics": "Evalúa la dinámica y la evolución de la amistad"
                }
            },
            "romantic_couple": {
                "name": "Analizador de Relación Romántica", 
                "system_prompt": """Eres un experto en analizar dinámicas de pareja y comunicación romántica.
                Te especializas en detectar expresiones de afecto, patrones de comunicación emocional,
                resolución de conflictos y la evolución de relaciones románticas.""",
                "analysis_focus": ["affection_patterns", "emotional_communication", "conflict_resolution", "relationship_evolution"],
                "specialized_prompts": {
                    "affection_patterns": "Analiza cómo la pareja expresa afecto y cariño",
                    "emotional_communication": "Evalúa la comunicación emocional entre la pareja",
                    "conflict_resolution": "Identifica patrones en la resolución de conflictos",
                    "relationship_evolution": "Analiza la evolución y etapas de la relación"
                }
            },
            "family": {
                "name": "Analizador Familiar",
                "system_prompt": """Eres un experto en dinámicas familiares y comunicación intergeneracional.
                Te especializas en detectar roles familiares, patrones de apoyo, tradiciones
                y la evolución de relaciones familiares.""",
                "analysis_focus": ["family_roles", "support_patterns", "generational_communication", "family_traditions"],
                "specialized_prompts": {
                    "family_roles": "Identifica los roles y dinámicas familiares",
                    "support_patterns": "Analiza cómo la familia se apoya mutuamente",
                    "generational_communication": "Evalúa la comunicación entre generaciones",
                    "family_traditions": "Detecta tradiciones y rutinas familiares"
                }
            },
            "work_professional": {
                "name": "Analizador Profesional",
                "system_prompt": """Eres un experto en comunicación profesional y dinámicas laborales.
                Te especializas en detectar jerarquías, colaboración, gestión de proyectos
                y cultura organizacional en comunicaciones laborales.""",
                "analysis_focus": ["hierarchy_dynamics", "collaboration_patterns", "project_management", "work_culture"],
                "specialized_prompts": {
                    "hierarchy_dynamics": "Analiza las dinámicas jerárquicas y de poder",
                    "collaboration_patterns": "Evalúa los patrones de colaboración y trabajo en equipo",
                    "project_management": "Identifica aspectos de gestión de proyectos y deadlines",
                    "work_culture": "Analiza la cultura organizacional reflejada en la comunicación"
                }
            },
            "gaming": {
                "name": "Analizador Gaming",
                "system_prompt": """Eres un experto en cultura gaming y comunicación entre jugadores.
                Te especializas en detectar patrones de juego cooperativo/competitivo, progresión,
                estrategias y dinámicas de comunidades gaming.""",
                "analysis_focus": ["gameplay_patterns", "competitive_dynamics", "team_coordination", "gaming_progression"],
                "specialized_prompts": {
                    "gameplay_patterns": "Analiza patrones de juego y preferencias gaming",
                    "competitive_dynamics": "Evalúa dinámicas competitivas y cooperativas",
                    "team_coordination": "Identifica estrategias de coordinación en equipo",
                    "gaming_progression": "Analiza progresión y logros en juegos"
                }
            },
            "support_emotional": {
                "name": "Analizador de Apoyo Emocional",
                "system_prompt": """Eres un experto en comunicación de apoyo emocional y bienestar psicológico.
                Te especializas en detectar patrones de apoyo, expresiones emocionales,
                estrategias de afrontamiento y evolución del bienestar emocional.""",
                "analysis_focus": ["support_strategies", "emotional_expression", "coping_mechanisms", "wellbeing_evolution"],
                "specialized_prompts": {
                    "support_strategies": "Analiza las estrategias de apoyo emocional utilizadas",
                    "emotional_expression": "Evalúa cómo se expresan y procesan las emociones",
                    "coping_mechanisms": "Identifica mecanismos de afrontamiento y resiliencia",
                    "wellbeing_evolution": "Analiza la evolución del bienestar emocional"
                }
            },
            "planning_organizing": {
                "name": "Analizador de Planificación",
                "system_prompt": """Eres un experto en análisis de planificación y organización de eventos.
                Te especializas en detectar patrones de toma de decisiones, coordinación logística,
                liderazgo organizativo y efectividad en la planificación.""",
                "analysis_focus": ["decision_making", "logistic_coordination", "leadership_patterns", "planning_effectiveness"],
                "specialized_prompts": {
                    "decision_making": "Analiza los patrones de toma de decisiones grupales",
                    "logistic_coordination": "Evalúa la coordinación logística y organizativa",
                    "leadership_patterns": "Identifica patrones de liderazgo en la planificación",
                    "planning_effectiveness": "Analiza la efectividad de los procesos de planificación"
                }
            }
        }
        
        config = agent_configs.get(context.category)
        if not config:
            return None
        
        return SpecializedAgent(
            name=config["name"],
            context_category=context.category,
            system_prompt=config["system_prompt"],
            analysis_focus=config["analysis_focus"],
            specialized_prompts=config["specialized_prompts"]
        )


class AdaptiveAnalyzer:
    """Analizador adaptativo de dos etapas."""
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        self.llm_manager = llm_manager or LLMManager()
        self.context_detector = ContextDetector(self.llm_manager)
        self.basic_analyzer = ChatAnalyzer()
    
    def analyze(self, data_frame: ChatDataFrame) -> AdaptiveAnalysisResult:
        """
        Ejecuta análisis adaptativo de dos etapas.
        
        Args:
            data_frame: DataFrame con mensajes del chat
            
        Returns:
            Resultado completo del análisis adaptativo
        """
        if data_frame.is_empty:
            return AdaptiveAnalysisResult(
                basic_analysis=self.basic_analyzer.full_analysis(data_frame),
                detected_contexts=[],
                specialized_analyses={},
                adaptive_insights=[],
                analysis_metadata={"error": "No data to analyze"}
            )
        
        logger.info("Iniciando análisis adaptativo de dos etapas")
        
        # ETAPA 1: Análisis básico
        logger.info("Etapa 1: Ejecutando análisis básico")
        basic_analysis = self.basic_analyzer.full_analysis(data_frame)
        
        # ETAPA 2: Detección de contexto
        logger.info("Etapa 2: Detectando contextos")
        detected_contexts = self.context_detector.detect_contexts(data_frame, basic_analysis)
        
        # ETAPA 3: Creación de agentes especializados
        logger.info(f"Etapa 3: Creando agentes especializados para {len(detected_contexts)} contextos")
        specialized_agents = SpecializedAgentFactory.create_agents(detected_contexts)
        
        # ETAPA 4: Análisis especializado
        specialized_analyses = {}
        for agent in specialized_agents:
            logger.info(f"Ejecutando análisis especializado: {agent.name}")
            try:
                analysis = self._run_specialized_analysis(agent, data_frame, basic_analysis)
                specialized_analyses[agent.context_category] = analysis
            except Exception as e:
                logger.warning(f"Error en análisis especializado {agent.name}: {e}")
                specialized_analyses[agent.context_category] = {"error": str(e)}
        
        # ETAPA 5: Generar insights adaptativos
        adaptive_insights = self._generate_adaptive_insights(
            basic_analysis, detected_contexts, specialized_analyses
        )
        
        metadata = {
            "analysis_timestamp": basic_analysis.analysis_metadata.get("analysis_timestamp"),
            "total_contexts_detected": len(detected_contexts),
            "specialized_agents_used": len(specialized_agents),
            "analysis_version": "2.0_adaptive"
        }
        
        return AdaptiveAnalysisResult(
            basic_analysis=basic_analysis,
            detected_contexts=detected_contexts,
            specialized_analyses=specialized_analyses,
            adaptive_insights=adaptive_insights,
            analysis_metadata=metadata
        )
    
    def _run_specialized_analysis(
        self,
        agent: SpecializedAgent,
        data_frame: ChatDataFrame,
        basic_analysis: AnalysisResult
    ) -> Dict[str, Any]:
        """Ejecuta análisis con un agente especializado."""
        
        # Preparar contexto específico para el agente
        df = data_frame.df
        sample_messages = []
        
        if not df.empty:
            # Tomar muestra estratégica de mensajes
            sample_size = min(30, len(df))
            sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
            
            for _, row in sample_df.iterrows():
                sample_messages.append({
                    "sender": row['author'],
                    "message": row['message'][:300],
                    "timestamp": str(row['timestamp'])
                })
        
        context_data = {
            "chat_summary": {
                "total_messages": len(df),
                "participants": df['author'].nunique(),
                "main_trends": [trend.description for trend in basic_analysis.trend_summaries]
            },
            "sample_messages": sample_messages
        }
        
        analyses = {}
        
        # Ejecutar cada tipo de análisis especializado
        for focus_area in agent.analysis_focus:
            if focus_area in agent.specialized_prompts:
                try:
                    prompt = f"""
                    {agent.system_prompt}
                    
                    Contexto de la conversación:
                    {json.dumps(context_data, indent=2, default=str)}
                    
                    Tarea específica: {agent.specialized_prompts[focus_area]}
                    
                    Proporciona un análisis detallado y específico para esta área.
                    Incluye insights únicos que no estarían disponibles en un análisis general.
                    Responde en español con bullet points y conclusiones claras.
                    """
                    
                    response = self.llm_manager.generate_response([
                        {"role": "system", "content": agent.system_prompt},
                        {"role": "user", "content": prompt}
                    ], temperature=0.3, max_tokens=800)
                    
                    analyses[focus_area] = response
                    
                except Exception as e:
                    logger.warning(f"Error en análisis {focus_area} para {agent.name}: {e}")
                    analyses[focus_area] = f"Error: {e}"
        
        return analyses
    
    def _generate_adaptive_insights(
        self,
        basic_analysis: AnalysisResult,
        contexts: List[ContextCategory],
        specialized_analyses: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Genera insights adaptativos combinando todos los análisis."""
        
        insights = []
        
        # Insight sobre contextos detectados
        if contexts:
            main_context = contexts[0]
            confidence_text = "alta" if main_context.confidence > 0.7 else "media" if main_context.confidence > 0.4 else "baja"
            insights.append(
                f"🎯 **Contexto Principal Detectado**: {main_context.category.replace('_', ' ').title()} "
                f"(confianza {confidence_text}: {main_context.confidence:.1%})"
            )
            
            # Evidencia del contexto principal
            if main_context.evidence:
                insights.append(f"📋 **Evidencia**:")
                for evidence_item in main_context.evidence:
                    insights.append(f"  • {evidence_item}")
        
        # Insights de análisis especializados
        for context_type, analyses in specialized_analyses.items():
            if isinstance(analyses, dict) and "error" not in analyses:
                context_name = context_type.replace('_', ' ').title()
                insights.append(f"🔍 **Análisis Especializado - {context_name}**:")
                
                for focus_area, analysis in analyses.items():
                    if analysis and isinstance(analysis, str) and len(analysis) > 10:
                        area_name = focus_area.replace('_', ' ').title()
                        insights.append(f"  • **{area_name}**:")
                        # Show complete analysis without truncation
                        analysis_lines = analysis.strip().split('\n')
                        for line in analysis_lines:
                            if line.strip():
                                insights.append(f"    {line.strip()}")
                        insights.append("")  # Add blank line
        
        # Insight combinado basado en análisis múltiples
        if len(contexts) > 1:
            context_names = [ctx.category.replace('_', ' ').title() for ctx in contexts]
            insights.append(
                f"🔄 **Conversación Multifacética**: Esta conversación combina elementos de "
                f"{', '.join(context_names)}, sugiriendo una relación compleja y multidimensional."
            )
        
        # Insight temporal si hay suficiente variedad en análises
        if len(specialized_analyses) >= 2:
            insights.append(
                f"📊 **Análisis Profundo Completado**: Se ejecutaron {len(specialized_analyses)} análisis "
                f"especializados, proporcionando una visión 360° de la conversación."
            )
        
        return insights  # Return all insights without limiting