from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

import pandas as pd
import os
from dotenv import load_dotenv

# Load .env file with override=True to take precedence over existing env vars
load_dotenv(override=True)

# Try to import OpenAI client directly for LLM calls
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .llm_providers import LLMManager

from .analysis import ChatDataFrame


@dataclass
class TrendSummary:
    """Resumen de una tendencia identificada."""
    trend_type: str
    description: str
    confidence_score: float
    supporting_data: Dict[str, Any]
    time_period: Optional[str] = None
    participants: Optional[List[str]] = None


@dataclass
class AnomalyDetection:
    """Detección de comportamiento anómalo."""
    anomaly_type: str
    description: str
    severity: str  # "low", "medium", "high"
    timestamp: Optional[datetime] = None
    participant: Optional[str] = None
    metrics: Dict[str, Any] = None


@dataclass
class QuotableMessage:
    """Mensaje memorable o citable."""
    message: str
    sender: str
    timestamp: datetime
    quote_type: str  # "funny", "insightful", "memorable", "emotional"
    relevance_score: float
    context: Optional[str] = None


@dataclass
class AnalysisResult:
    """Resultado completo del análisis de chat."""
    trend_summaries: List[TrendSummary]
    anomalies: List[AnomalyDetection]
    quotable_messages: List[QuotableMessage]
    analysis_metadata: Dict[str, Any]


class ChatAnalyzer:
    """Analizador de conversaciones WhatsApp usando LiteLLM."""
    
    def __init__(self, llm_model_name: Optional[str] = None):
        """
        Inicializa el analizador con un modelo LLM.

        Args:
            llm_model_name: Nombre del modelo LLM a usar (por defecto usa OpenAI GPT-4)
        """
        self.model_name = llm_model_name or "gpt-4o-mini"
        self.llm_manager = LLMManager()
        
    def _call_llm(self, prompt: str) -> str:
        """Llama al modelo LLM con el prompt dado."""
        messages = [
            {
                "role": "system",
                "content": """Eres un experto analista de RELACIONES HUMANAS y comunicación interpersonal.
                Tu enfoque PRINCIPAL es analizar dinámicas relacionales, vínculos emocionales, conexiones afectivas
                y patrones de comunicación humana que revelan la calidad de las relaciones entre personas.
                PRIORIZA SIEMPRE aspectos de amistad, amor, familia, apoyo emocional y conexión humana
                por encima de temas de trabajo, tareas o contenido transaccional. Busca la dimensión emocional
                y relacional en cada interacción."""
            },
            {"role": "user", "content": prompt}
        ]

        try:
            return self.llm_manager.generate_response(messages, temperature=0.2, max_tokens=2000)
        except Exception as e:
            raise RuntimeError(f"Error al llamar al modelo LLM: {e}")
    
    def analyze_conversation_patterns(self, data_frame: ChatDataFrame) -> List[TrendSummary]:
        """
        Analiza patrones de conversación usando smolagents.
        
        Args:
            data_frame: DataFrame con los mensajes del chat
            
        Returns:
            Lista de tendencias identificadas
        """
        if data_frame.is_empty:
            return []
        
        # Preparar datos para análisis
        stats = data_frame.get_message_stats()
        daily_activity = data_frame.get_daily_activity()
        hourly_activity = data_frame.get_hourly_activity()
        
        # Preparar contexto para el agente
        def serialize_dataframe_dict(df_dict):
            """Convierte claves datetime a strings para serialización JSON."""
            if not df_dict:
                return {}
            
            serialized = {}
            for key, value in df_dict.items():
                # Convertir claves datetime a string
                if hasattr(key, 'strftime'):
                    str_key = key.strftime('%Y-%m-%d') if hasattr(key, 'date') else str(key)
                else:
                    str_key = str(key)
                
                # Convertir valores que puedan ser problemáticos
                if isinstance(value, dict):
                    value = serialize_dataframe_dict(value)
                
                serialized[str_key] = value
            
            return serialized
        
        # Preparar datos de actividad diaria con conversión segura
        daily_activity_dict = {}
        if not daily_activity.empty:
            daily_sample = daily_activity.head(10).to_dict()
            daily_activity_dict = serialize_dataframe_dict(daily_sample)
        
        # Preparar patrones horarios con conversión segura
        hourly_patterns_dict = {}
        if not hourly_activity.empty:
            hourly_dict = hourly_activity.sum(axis=1).to_dict()
            hourly_patterns_dict = serialize_dataframe_dict(hourly_dict)
        
        context = {
            "total_messages": stats.get("total_messages", 0),
            "unique_authors": stats.get("unique_authors", 0),
            "date_range": {
                "start": str(stats.get("date_range", {}).get("start", "")),
                "end": str(stats.get("date_range", {}).get("end", ""))
            },
            "top_authors": dict(list(stats.get("authors", {}).items())[:5]),
            "avg_message_length": stats.get("avg_message_length", 0),
            "most_active_hour": stats.get("most_active_hour"),
            "daily_activity_sample": daily_activity_dict,
            "hourly_patterns": hourly_patterns_dict
        }
        
        # Prompt para análisis de patrones
        prompt = f"""
        Analiza los siguientes datos de una conversación de WhatsApp ENFOCÁNDOTE EN RELACIONES HUMANAS Y DINÁMICAS INTERPERSONALES:
        
        {json.dumps(context, indent=2, default=str)}
        
        PRIORIZA identificar patrones que revelen:
        - Vínculos afectivos y emocionales entre participantes
        - Patrones de cuidado mutuo y apoyo emocional
        - Dinámicas de intimidad, confianza y cercanía
        - Evolución de relaciones (amistad, romance, familia)
        - Expresiones de cariño, preocupación o afecto
        - Momentos de vulnerabilidad compartida
        - Rituales de comunicación únicos entre las personas
        
        MINIMIZA o evita patrones relacionados con:
        - Gestión de tareas o proyectos
        - Coordinación puramente logística
        - Contenido transaccional sin dimensión emocional
        
        Identifica hasta 5 tendencias RELACIONALES más relevantes. Para cada patrón, proporciona:
        1. Tipo de tendencia (ej: "vínculo_afectivo_creciente", "apoyo_emocional_constante", "intimidad_comunicativa")
        2. Descripción clara en español enfocada en la dimensión humana
        3. Puntuación de confianza (0.0-1.0)
        4. Datos que lo respaldan
        
        Responde en formato JSON con una lista de objetos con las llaves:
        - trend_type: string
        - description: string  
        - confidence_score: number
        - supporting_data: object
        - time_period: string (opcional)
        - participants: array de strings (opcional)
        """
        
        try:
            response = self._call_llm(prompt)
            
            # Parsear respuesta JSON
            if isinstance(response, str):
                # Extraer JSON de la respuesta si está embebido en texto
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    response = json_match.group()
                    
                trends_data = json.loads(response)
            else:
                trends_data = response
            
            # Convertir a objetos TrendSummary
            trends = []
            for trend_data in trends_data:
                if isinstance(trend_data, dict):
                    trend = TrendSummary(
                        trend_type=trend_data.get("trend_type", "unknown"),
                        description=trend_data.get("description", ""),
                        confidence_score=float(trend_data.get("confidence_score", 0.5)),
                        supporting_data=trend_data.get("supporting_data", {}),
                        time_period=trend_data.get("time_period"),
                        participants=trend_data.get("participants")
                    )
                    trends.append(trend)
                    
            return trends
            
        except Exception:
            # Fallback a análisis básico sin LLM
            return self._basic_pattern_analysis(data_frame)
    
    def detect_unusual_behavior(self, data_frame: ChatDataFrame) -> List[AnomalyDetection]:
        """
        Detecta comportamientos inusuales en la conversación.
        
        Args:
            data_frame: DataFrame con los mensajes del chat
            
        Returns:
            Lista de anomalías detectadas
        """
        if data_frame.is_empty:
            return []
        
        anomalies = []
        df = data_frame.df
        
        # Detectar picos de actividad inusuales
        daily_counts = df.groupby(df['timestamp'].dt.date).size()
        if len(daily_counts) > 1:
            mean_daily = daily_counts.mean()
            std_daily = daily_counts.std()
            
            for date, count in daily_counts.items():
                if count > mean_daily + 2 * std_daily:  # 2 desviaciones estándar
                    anomalies.append(AnomalyDetection(
                        anomaly_type="pico_actividad",
                        description=f"Actividad inusualmente alta el {date}: {count} mensajes (promedio: {mean_daily:.1f})",
                        severity="medium",
                        timestamp=datetime.combine(date, datetime.min.time()),
                        metrics={"count": count, "average": mean_daily, "std_dev": std_daily}
                    ))
        
        # Detectar mensajes excepcionalmente largos
        msg_lengths = df['message'].str.len()
        mean_length = msg_lengths.mean()
        
        long_messages = df[msg_lengths > mean_length + 3 * msg_lengths.std()]
        for _, msg in long_messages.head(3).iterrows():  # Limitar a 3 mensajes más largos
            anomalies.append(AnomalyDetection(
                anomaly_type="mensaje_largo",
                description=f"Mensaje excepcionalmente largo de {msg['author']}: {len(msg['message'])} caracteres",
                severity="low",
                timestamp=msg['timestamp'].to_pydatetime(),
                participant=msg['author'],
                metrics={"length": len(msg['message']), "average": mean_length}
            ))
        
        # Mensajes muy tarde o muy temprano (madrugada)
        late_night_msgs = df[(df['timestamp'].dt.hour >= 2) & (df['timestamp'].dt.hour <= 5)]
        if not late_night_msgs.empty:
            late_counts = late_night_msgs.groupby('author', observed=True).size()
            for author, count in late_counts.head(3).items():
                if count >= 5:  # Al menos 5 mensajes en horario de madrugada
                    anomalies.append(AnomalyDetection(
                        anomaly_type="horario_inusual",
                        description=f"{author} envió {count} mensajes en horario de madrugada (2-5 AM)",
                        severity="low",
                        participant=author,
                        metrics={"count": count, "time_range": "2-5 AM"}
                    ))
        
        return anomalies
    
    def find_quotable_messages(self, data_frame: ChatDataFrame, limit: int = 10) -> List[QuotableMessage]:
        """
        Encuentra mensajes memorables o citables usando smolagents.
        
        Args:
            data_frame: DataFrame con los mensajes del chat
            limit: Número máximo de mensajes a retornar
            
        Returns:
            Lista de mensajes citables
        """
        if data_frame.is_empty:
            return []
        
        df = data_frame.df
        
        # Seleccionar una muestra representativa de mensajes para análisis
        # Priorizar mensajes más largos y únicos
        sample_df = df[df['message'].str.len() > 20].copy()  # Mensajes con al menos 20 caracteres
        
        if sample_df.empty:
            return []
        
        # Limitar a una muestra manejable para el LLM
        if len(sample_df) > 100:
            sample_df = sample_df.sample(100)
        
        # Preparar mensajes para análisis
        messages_for_analysis = []
        for _, row in sample_df.iterrows():
            messages_for_analysis.append({
                "sender": row['author'],
                "message": row['message'],
                "timestamp": str(row['timestamp'])
            })
        
        prompt = f"""
        Analiza estos mensajes de WhatsApp PRIORIZANDO aquellos que revelan CONEXIÓN HUMANA Y DIMENSIÓN RELACIONAL.
        
        Busca mensajes que sean:
        - Expresiones auténticas de afecto, cariño o amor
        - Momentos de vulnerabilidad emocional compartida
        - Apoyo genuino en momentos difíciles
        - Declaraciones de aprecio o gratitud personal
        - Expresiones de preocupación sincera por el otro
        - Momentos íntimos o de complicidad especial
        - Manifestaciones de confianza profunda
        - Comunicación que fortalece vínculos relacionales
        
        DEPRIORITIZA mensajes que sean:
        - Puramente informativos o transaccionales
        - Relacionados solo con tareas o logística
        - Sin carga emocional o relacional significativa
        
        Mensajes a analizar:
        {json.dumps(messages_for_analysis, indent=2, ensure_ascii=False)}
        
        Para cada mensaje seleccionado, proporciona:
        - message: texto del mensaje
        - sender: autor del mensaje  
        - timestamp: timestamp del mensaje
        - quote_type: tipo ("affectionate", "supportive", "intimate", "caring", "emotionally_significant")
        - relevance_score: puntuación 0.0-1.0 (prioriza alto score para contenido relacional)
        - context: explicación de por qué es significativo para la relación humana
        
        Responde en formato JSON con una lista de objetos.
        """
        
        try:
            response = self._call_llm(prompt)
            
            # Parsear respuesta JSON
            if isinstance(response, str):
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    response = json_match.group()
                quotes_data = json.loads(response)
            else:
                quotes_data = response
                
            # Convertir a objetos QuotableMessage
            quotes = []
            for quote_data in quotes_data:
                if isinstance(quote_data, dict):
                    timestamp_str = quote_data.get("timestamp", "")
                    try:
                        timestamp = pd.to_datetime(timestamp_str).to_pydatetime()
                    except Exception:
                        timestamp = datetime.now()
                    
                    quote = QuotableMessage(
                        message=quote_data.get("message", ""),
                        sender=quote_data.get("sender", ""),
                        timestamp=timestamp,
                        quote_type=quote_data.get("quote_type", "memorable"),
                        relevance_score=float(quote_data.get("relevance_score", 0.5)),
                        context=quote_data.get("context")
                    )
                    quotes.append(quote)
                    
            return quotes[:limit]
            
        except Exception:
            # Fallback a selección básica sin LLM
            return self._basic_quote_selection(data_frame, limit)
    
    def full_analysis(self, data_frame: ChatDataFrame) -> AnalysisResult:
        """
        Realiza un análisis completo de la conversación.
        
        Args:
            data_frame: DataFrame con los mensajes del chat
            
        Returns:
            Resultado completo del análisis
        """
        trends = self.analyze_conversation_patterns(data_frame)
        anomalies = self.detect_unusual_behavior(data_frame)
        quotes = self.find_quotable_messages(data_frame)
        
        # Metadatos del análisis
        metadata = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_messages_analyzed": len(data_frame),
            "analysis_version": "1.0",
            "model_used": self.model_name
        }
        
        return AnalysisResult(
            trend_summaries=trends,
            anomalies=anomalies,
            quotable_messages=quotes,
            analysis_metadata=metadata
        )
    
    def _basic_pattern_analysis(self, data_frame: ChatDataFrame) -> List[TrendSummary]:
        """Análisis básico de patrones sin LLM como fallback."""
        trends = []
        stats = data_frame.get_message_stats()
        
        # Tendencia de participación
        authors = stats.get("authors", {})
        if len(authors) > 1:
            total_msgs = sum(authors.values())
            max_participation = max(authors.values()) / total_msgs
            
            if max_participation > 0.7:
                dominant_author = max(authors, key=authors.get)
                trends.append(TrendSummary(
                    trend_type="participacion_desigual",
                    description=f"{dominant_author} domina la conversación con {max_participation:.1%} de los mensajes",
                    confidence_score=0.8,
                    supporting_data={"participation_rates": authors}
                ))
        
        return trends
    
    def _basic_quote_selection(self, data_frame: ChatDataFrame, limit: int) -> List[QuotableMessage]:
        """Selección básica de mensajes citables sin LLM como fallback."""
        df = data_frame.df

        logging.info(f"DataFrame columns: {list(df.columns)}")
        logging.info(f"Limit: {limit}")
        logging.info(f"Message lengths sample: {df['message'].str.len().head().tolist()}")

        # Seleccionar mensajes más largos como proxy para contenido interesante
        df['message_length'] = df['message'].str.len()
        long_messages = df.nlargest(limit, 'message_length')
        
        quotes = []
        for _, row in long_messages.iterrows():
            quote = QuotableMessage(
                message=row['message'],
                sender=row['author'],
                timestamp=row['timestamp'].to_pydatetime(),
                quote_type="memorable",
                relevance_score=min(len(row['message']) / 200, 1.0),  # Score based on length
                context="Mensaje seleccionado por longitud"
            )
            quotes.append(quote)
            
        return quotes