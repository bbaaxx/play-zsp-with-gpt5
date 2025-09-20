"""Tests básicos para el módulo core de parsing de WhatsApp."""

from datetime import datetime


from rag.core import parse_whatsapp_txt


class TestWhatsAppParser:
    """Tests para el parser de mensajes de WhatsApp."""
    
    def test_basic_parsing(self):
        """Test parsing básico de mensajes."""
        content = """[12/10/2023, 21:15] Juan: ¿Salimos mañana?
[12/10/2023, 21:16] María: Sí, ¿a qué hora te viene bien?"""
        
        messages = parse_whatsapp_txt(content)
        
        assert len(messages) == 2
        assert messages[0].sender == "Juan"
        assert messages[0].text == "¿Salimos mañana?"
        assert messages[0].timestamp == datetime(2023, 10, 12, 21, 15)
        
        assert messages[1].sender == "María"
        assert messages[1].text == "Sí, ¿a qué hora te viene bien?"
    
    def test_different_formats(self):
        """Test parsing de diferentes formatos de fecha/hora."""
        content = """[12/10/2023, 21:15] Juan: Mensaje 1
12/10/23, 21:16 - María: Mensaje 2"""
        
        messages = parse_whatsapp_txt(content)
        
        assert len(messages) == 2
        assert messages[0].sender == "Juan"
        assert messages[1].sender == "María"
        assert messages[0].timestamp.year == 2023
        assert messages[1].timestamp.year == 2023
    
    def test_system_messages_filtered(self):
        """Test que los mensajes del sistema se filtran."""
        content = """[12/10/2023, 21:15] Juan: Mensaje real
Messages and calls are end-to-end encrypted
[12/10/2023, 21:16] María: Otro mensaje real
<image omitted>"""
        
        messages = parse_whatsapp_txt(content)
        
        assert len(messages) == 2
        assert messages[0].text == "Mensaje real"
        assert messages[1].text == "Otro mensaje real"
    
    def test_empty_content(self):
        """Test con contenido vacío."""
        messages = parse_whatsapp_txt("")
        assert len(messages) == 0
    
    def test_chat_id_generation(self):
        """Test generación de chat_id."""
        content = "[12/10/2023, 21:15] Juan: Test"
        
        messages1 = parse_whatsapp_txt(content)
        messages2 = parse_whatsapp_txt(content)
        
        # Mismo contenido = mismo chat_id
        assert messages1[0].chat_id == messages2[0].chat_id
        
        # Contenido diferente = chat_id diferente
        messages3 = parse_whatsapp_txt("[12/10/2023, 21:15] Pedro: Test")
        assert messages1[0].chat_id != messages3[0].chat_id
    
    def test_line_numbers(self):
        """Test que los números de línea se asignan correctamente."""
        content = """[12/10/2023, 21:15] Juan: Línea 1
[12/10/2023, 21:16] María: Línea 2
[12/10/2023, 21:17] Pedro: Línea 3"""
        
        messages = parse_whatsapp_txt(content)
        
        assert messages[0].line_no == 1
        assert messages[1].line_no == 2
        assert messages[2].line_no == 3