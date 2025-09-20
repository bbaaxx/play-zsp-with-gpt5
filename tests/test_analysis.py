"""Tests para el módulo de análisis de datos de WhatsApp."""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from rag.analysis import ChatDataFrame
from rag.core import ChatMessage


# Datos de prueba
SAMPLE_MESSAGES = [
    ChatMessage("test_chat", datetime(2023, 10, 12, 21, 15), "Juan", "¿Salimos mañana?", 1),
    ChatMessage("test_chat", datetime(2023, 10, 12, 21, 16), "María", "Sí, ¿a qué hora te viene bien?", 2),
    ChatMessage("test_chat", datetime(2023, 10, 12, 21, 17), "Juan", "A las 20:30 en el centro.", 3),
    ChatMessage("test_chat", datetime(2023, 10, 12, 21, 20), "Pedro", "Me apunto. ¿Reservo mesa?", 4),
    ChatMessage("test_chat", datetime(2023, 10, 13, 9, 30), "María", "Buenos días a todos", 5),
]

SAMPLE_CHAT_TEXT = """[12/10/2023, 21:15] Juan: ¿Salimos mañana?
[12/10/2023, 21:16] María: Sí, ¿a qué hora te viene bien?
[12/10/2023, 21:17] Juan: A las 20:30 en el centro.
[12/10/2023, 21:20] Pedro: Me apunto. ¿Reservo mesa?
[13/10/2023, 09:30] María: Buenos días a todos"""


class TestChatDataFrame:
    """Tests para la clase ChatDataFrame."""
    
    def test_empty_initialization(self):
        """Test inicialización vacía."""
        analyzer = ChatDataFrame()
        assert analyzer.is_empty
        assert len(analyzer) == 0
        assert analyzer.df.empty
    
    def test_load_from_messages(self):
        """Test carga desde lista de mensajes."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        
        assert not analyzer.is_empty
        assert len(analyzer) == 5
        assert analyzer.df['author'].nunique() == 3
        assert analyzer.df['timestamp'].min() == datetime(2023, 10, 12, 21, 15)
        assert analyzer.df['timestamp'].max() == datetime(2023, 10, 13, 9, 30)
    
    def test_load_from_file(self):
        """Test carga desde archivo."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(SAMPLE_CHAT_TEXT)
            temp_path = f.name
        
        try:
            analyzer = ChatDataFrame()
            analyzer.load_from_file(temp_path)
            
            assert not analyzer.is_empty
            assert len(analyzer) == 5
            assert "Juan" in analyzer.df['author'].values
            assert "María" in analyzer.df['author'].values
            assert "Pedro" in analyzer.df['author'].values
        finally:
            Path(temp_path).unlink()
    
    def test_data_types(self):
        """Test que los tipos de datos están optimizados."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        df = analyzer.df
        
        assert df['timestamp'].dtype == 'datetime64[ns]'
        assert df['author'].dtype.name == 'category'
        assert df['chat_id'].dtype.name == 'category'
        assert df['line_no'].dtype == 'int32'
    
    def test_filter_by_author_exact(self):
        """Test filtrado exacto por autor."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        juan_msgs = analyzer.filter_by_author("Juan")
        
        assert len(juan_msgs) == 2
        for msg in juan_msgs.to_messages():
            assert msg.sender == "Juan"
    
    def test_filter_by_author_partial(self):
        """Test filtrado parcial por autor."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        msgs_with_a = analyzer.filter_by_author("a", exact_match=False)
        
        # Juan (2) + María (2) = 4 mensajes con "a" en el nombre
        assert len(msgs_with_a) == 4  
        authors = {msg.sender for msg in msgs_with_a.to_messages()}
        assert "Juan" in authors
        assert "María" in authors
        assert "Pedro" not in authors
    
    def test_filter_by_content(self):
        """Test filtrado por contenido."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        
        # Búsqueda simple
        hora_msgs = analyzer.filter_by_content("hora")
        assert len(hora_msgs) == 1
        
        # Búsqueda regex
        regex_msgs = analyzer.filter_by_content(r'\d+:\d+', regex=True)
        assert len(regex_msgs) == 1
        
        # Case insensitive
        buenos_msgs = analyzer.filter_by_content("BUENOS", case_sensitive=False)
        assert len(buenos_msgs) == 1
    
    def test_filter_by_date_range(self):
        """Test filtrado por rango de fechas."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        
        # Solo 12 de octubre
        day_12_msgs = analyzer.filter_by_date_range("2023-10-12", "2023-10-12")
        assert len(day_12_msgs) == 4
        
        # Solo 13 de octubre
        day_13_msgs = analyzer.filter_by_date_range("2023-10-13", "2023-10-13")
        assert len(day_13_msgs) == 1
        
        # Rango completo
        all_msgs = analyzer.filter_by_date_range("2023-10-12", "2023-10-13")
        assert len(all_msgs) == 5
    
    def test_get_message_stats(self):
        """Test estadísticas básicas."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        stats = analyzer.get_message_stats()
        
        assert stats['total_messages'] == 5
        assert stats['unique_authors'] == 3
        assert stats['authors']['Juan'] == 2
        assert stats['authors']['María'] == 2
        assert stats['authors']['Pedro'] == 1
        assert stats['most_active_hour'] == 21
    
    def test_search_keywords(self):
        """Test búsqueda de palabras clave con contexto."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        results = analyzer.search_keywords("mesa", context_window=1)
        
        assert not results.empty
        assert results['is_match'].sum() == 1  # Solo un mensaje contiene "mesa"
        assert len(results) >= 2  # Al menos el mensaje + contexto
    
    def test_daily_activity(self):
        """Test análisis de actividad diaria."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        daily = analyzer.get_daily_activity()
        
        assert not daily.empty
        assert len(daily) == 2  # 2 días diferentes
        assert daily.loc[pd.to_datetime('2023-10-12').date(), 'Juan'] == 2
        assert daily.loc[pd.to_datetime('2023-10-13').date(), 'María'] == 1
    
    def test_hourly_activity(self):
        """Test análisis de actividad por hora."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        hourly = analyzer.get_hourly_activity()
        
        assert not hourly.empty
        assert 21 in hourly.index  # Hora 21
        assert 9 in hourly.index   # Hora 9
        assert hourly.loc[21, 'Juan'] == 2
    
    def test_export_csv(self):
        """Test exportación a CSV."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            analyzer.export_to_csv(csv_path)
            
            # Verificar que se creó el archivo
            assert Path(csv_path).exists()
            
            # Verificar contenido
            df_imported = pd.read_csv(csv_path)
            assert len(df_imported) == 5
            assert 'timestamp' in df_imported.columns
            assert 'author' in df_imported.columns
            assert 'message' in df_imported.columns
        finally:
            Path(csv_path).unlink(missing_ok=True)
    
    def test_to_messages_conversion(self):
        """Test conversión de vuelta a ChatMessage objects."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        messages = analyzer.to_messages()
        
        assert len(messages) == 5
        assert all(isinstance(msg, ChatMessage) for msg in messages)
        assert messages[0].sender == "Juan"
        assert messages[0].text == "¿Salimos mañana?"
    
    def test_combined_filters(self):
        """Test filtros combinados."""
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        
        # Filtrar por autor y luego por contenido
        juan_msgs = analyzer.filter_by_author("Juan")
        juan_with_centro = juan_msgs.filter_by_content("centro")
        
        assert len(juan_with_centro) == 1
        msg = juan_with_centro.to_messages()[0]
        assert msg.sender == "Juan"
        assert "centro" in msg.text
    
    def test_empty_operations(self):
        """Test operaciones en DataFrame vacío."""
        analyzer = ChatDataFrame()
        
        # Todas las operaciones deben retornar resultados vacíos sin error
        assert analyzer.filter_by_author("Juan").is_empty
        assert analyzer.filter_by_content("test").is_empty
        assert analyzer.filter_by_date_range("2023-01-01", "2023-01-02").is_empty
        assert analyzer.get_message_stats() == {}
        assert analyzer.get_daily_activity().empty
        assert analyzer.get_hourly_activity().empty
        assert analyzer.search_keywords("test").empty
        assert analyzer.to_messages() == []
    
    def test_repr(self):
        """Test representación string."""
        empty_analyzer = ChatDataFrame()
        assert "vacío" in repr(empty_analyzer)
        
        analyzer = ChatDataFrame(SAMPLE_MESSAGES)
        repr_str = repr(analyzer)
        assert "5 mensajes" in repr_str
        assert "3 autores" in repr_str