from __future__ import annotations

import re
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import pandas as pd

from .core import parse_whatsapp_txt, ChatMessage


class ChatDataFrame:
    """Análisis de datos de WhatsApp usando pandas DataFrame."""
    
    def __init__(self, messages: Optional[List[ChatMessage]] = None):
        """
        Inicializa el analizador con mensajes opcionales.
        
        Args:
            messages: Lista de ChatMessage objects
        """
        self._df: Optional[pd.DataFrame] = None
        if messages:
            self.load_from_messages(messages)
    
    def load_from_file(self, file_path: Union[str, Path], chat_id: Optional[str] = None) -> None:
        """
        Carga mensajes desde un archivo de texto de WhatsApp.
        
        Args:
            file_path: Ruta al archivo de exportación de WhatsApp
            chat_id: ID opcional del chat
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        messages = parse_whatsapp_txt(content, chat_id)
        self.load_from_messages(messages)
    
    def load_from_messages(self, messages: List[ChatMessage]) -> None:
        """
        Carga mensajes desde una lista de ChatMessage objects.
        
        Args:
            messages: Lista de ChatMessage objects
        """
        if not messages:
            self._df = pd.DataFrame(columns=['timestamp', 'author', 'message', 'chat_id', 'line_no'])
            return
        
        data = []
        for msg in messages:
            data.append({
                'timestamp': msg.timestamp,
                'author': msg.sender,
                'message': msg.text,
                'chat_id': msg.chat_id,
                'line_no': msg.line_no
            })
        
        self._df = pd.DataFrame(data)
        
        # Optimizar tipos de datos
        self._df['timestamp'] = pd.to_datetime(self._df['timestamp'])
        self._df['author'] = self._df['author'].astype('category')
        self._df['chat_id'] = self._df['chat_id'].astype('category')
        self._df['line_no'] = self._df['line_no'].astype('int32')
        
        # Ordenar por timestamp
        self._df = self._df.sort_values('timestamp').reset_index(drop=True)
    
    @property
    def df(self) -> pd.DataFrame:
        """Acceso de solo lectura al DataFrame."""
        if self._df is None:
            return pd.DataFrame()
        return self._df.copy()
    
    @property
    def is_empty(self) -> bool:
        """Verifica si el DataFrame está vacío."""
        return self._df is None or self._df.empty
    
    def filter_by_date_range(
        self, 
        start_date: Optional[Union[str, datetime, date]] = None,
        end_date: Optional[Union[str, datetime, date]] = None
    ) -> ChatDataFrame:
        """
        Filtra mensajes por rango de fechas.
        
        Args:
            start_date: Fecha de inicio (inclusive)
            end_date: Fecha de fin (inclusive)
            
        Returns:
            Nueva instancia de ChatDataFrame con los datos filtrados
        """
        if self.is_empty:
            return ChatDataFrame()
        
        df_filtered = self._df.copy()
        
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            elif isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_date = datetime.combine(start_date, datetime.min.time())
            df_filtered = df_filtered[df_filtered['timestamp'] >= start_date]
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                # Si solo es una fecha string, incluir todo el día
                if len(end_date.strftime('%Y-%m-%d')) == len(str(end_date).split()[0]):
                    end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            elif isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_date = datetime.combine(end_date, datetime.max.time())
            df_filtered = df_filtered[df_filtered['timestamp'] <= end_date]
        
        # Convertir de vuelta a ChatMessage objects para mantener consistencia
        messages = self._df_to_messages(df_filtered)
        return ChatDataFrame(messages)
    
    def filter_by_author(self, authors: Union[str, List[str]], exact_match: bool = True) -> ChatDataFrame:
        """
        Filtra mensajes por autor(es).
        
        Args:
            authors: Nombre(s) del autor o lista de autores
            exact_match: Si True, coincidencia exacta. Si False, coincidencia parcial
            
        Returns:
            Nueva instancia de ChatDataFrame con los datos filtrados
        """
        if self.is_empty:
            return ChatDataFrame()
        
        if isinstance(authors, str):
            authors = [authors]
        
        df_filtered = self._df.copy()
        
        if exact_match:
            df_filtered = df_filtered[df_filtered['author'].isin(authors)]
        else:
            # Coincidencia parcial (case insensitive)
            pattern = '|'.join(re.escape(author) for author in authors)
            mask = df_filtered['author'].str.contains(pattern, case=False, na=False)
            df_filtered = df_filtered[mask]
        
        messages = self._df_to_messages(df_filtered)
        return ChatDataFrame(messages)
    
    def filter_by_content(
        self, 
        pattern: str, 
        case_sensitive: bool = False,
        regex: bool = False
    ) -> ChatDataFrame:
        """
        Filtra mensajes por contenido usando patrones de texto.
        
        Args:
            pattern: Patrón a buscar en el contenido del mensaje
            case_sensitive: Si la búsqueda distingue mayúsculas/minúsculas
            regex: Si el patrón es una expresión regular
            
        Returns:
            Nueva instancia de ChatDataFrame con los datos filtrados
        """
        if self.is_empty:
            return ChatDataFrame()
        
        df_filtered = self._df.copy()
        
        try:
            mask = df_filtered['message'].str.contains(
                pattern, 
                case=case_sensitive, 
                regex=regex, 
                na=False
            )
            df_filtered = df_filtered[mask]
        except re.error as e:
            raise ValueError(f"Patrón regex inválido: {e}")
        
        messages = self._df_to_messages(df_filtered)
        return ChatDataFrame(messages)
    
    def get_message_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas básicas del chat.
        
        Returns:
            Diccionario con estadísticas del chat
        """
        if self.is_empty:
            return {}
        
        stats = {
            'total_messages': len(self._df),
            'unique_authors': self._df['author'].nunique(),
            'authors': self._df['author'].value_counts().to_dict(),
            'date_range': {
                'start': self._df['timestamp'].min(),
                'end': self._df['timestamp'].max()
            },
            'messages_per_day': self._df.groupby(self._df['timestamp'].dt.date).size().mean(),
            'avg_message_length': self._df['message'].str.len().mean(),
            'most_active_hour': self._df['timestamp'].dt.hour.mode().iloc[0] if not self._df.empty else None
        }
        
        return stats
    
    def get_daily_activity(self) -> pd.DataFrame:
        """
        Obtiene actividad diaria del chat.
        
        Returns:
            DataFrame con conteo de mensajes por día y autor
        """
        if self.is_empty:
            return pd.DataFrame()
        
        daily = self._df.groupby([
            self._df['timestamp'].dt.date,
            'author'
        ], observed=True).size().unstack(fill_value=0)
        
        daily.index.name = 'date'
        return daily
    
    def get_hourly_activity(self) -> pd.DataFrame:
        """
        Obtiene actividad por hora del día.
        
        Returns:
            DataFrame con conteo de mensajes por hora y autor
        """
        if self.is_empty:
            return pd.DataFrame()
        
        hourly = self._df.groupby([
            self._df['timestamp'].dt.hour,
            'author'
        ], observed=True).size().unstack(fill_value=0)
        
        hourly.index.name = 'hour'
        return hourly
    
    def search_keywords(self, keywords: Union[str, List[str]], context_window: int = 2) -> pd.DataFrame:
        """
        Busca palabras clave y devuelve mensajes con contexto.
        
        Args:
            keywords: Palabra clave o lista de palabras clave
            context_window: Número de mensajes antes y después a incluir como contexto
            
        Returns:
            DataFrame con mensajes que contienen las palabras clave y su contexto
        """
        if self.is_empty:
            return pd.DataFrame()
        
        if isinstance(keywords, str):
            keywords = [keywords]
        
        # Crear patrón regex para todas las palabras clave
        pattern = '|'.join(re.escape(keyword) for keyword in keywords)
        
        # Encontrar mensajes que contienen las palabras clave
        matches = self._df[self._df['message'].str.contains(pattern, case=False, na=False)]
        
        if matches.empty:
            return matches
        
        # Obtener índices con contexto
        context_indices = set()
        for idx in matches.index:
            start_idx = max(0, idx - context_window)
            end_idx = min(len(self._df), idx + context_window + 1)
            context_indices.update(range(start_idx, end_idx))
        
        context_df = self._df.iloc[sorted(context_indices)].copy()
        context_df['is_match'] = context_df.index.isin(matches.index)
        
        return context_df
    
    def _df_to_messages(self, df: pd.DataFrame) -> List[ChatMessage]:
        """Convierte un DataFrame de vuelta a lista de ChatMessage objects."""
        messages = []
        for _, row in df.iterrows():
            msg = ChatMessage(
                chat_id=row['chat_id'],
                timestamp=row['timestamp'].to_pydatetime(),
                sender=row['author'],
                text=row['message'],
                line_no=row['line_no']
            )
            messages.append(msg)
        return messages
    
    def to_messages(self) -> List[ChatMessage]:
        """
        Convierte el DataFrame actual a lista de ChatMessage objects.
        
        Returns:
            Lista de ChatMessage objects
        """
        if self.is_empty:
            return []
        return self._df_to_messages(self._df)
    
    def export_to_csv(self, file_path: Union[str, Path]) -> None:
        """
        Exporta el DataFrame a un archivo CSV.
        
        Args:
            file_path: Ruta del archivo CSV de salida
        """
        if self.is_empty:
            raise ValueError("No hay datos para exportar")
        
        self._df.to_csv(file_path, index=False, encoding='utf-8')
    
    def __len__(self) -> int:
        """Retorna el número de mensajes."""
        return len(self._df) if not self.is_empty else 0
    
    def __repr__(self) -> str:
        if self.is_empty:
            return "ChatDataFrame(vacío)"
        return f"ChatDataFrame({len(self._df)} mensajes, {self._df['author'].nunique()} autores)"