"""
File upload and document handling operations.
"""

from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass

from rag.core import parse_whatsapp_txt
from rag import ChatDataFrame

logger = logging.getLogger(__name__)


@dataclass
class FileProcessingResult:
    """Result of file processing operation."""
    success: bool
    message: str
    content: Optional[str] = None
    chat_dataframe: Optional[ChatDataFrame] = None
    n_messages: int = 0


class FileManager:
    """Handles file upload and document processing operations."""
    
    def extract_path(self, file_obj) -> Optional[str]:
        """Extract file path from various Gradio file object types."""
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
    
    def read_file_content(self, file_obj) -> FileProcessingResult:
        """Read and validate WhatsApp export file content."""
        path = self.extract_path(file_obj)
        if not path:
            return FileProcessingResult(
                success=False,
                message="Sube un archivo TXT de WhatsApp primero."
            )
        
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            return FileProcessingResult(
                success=False,
                message=f"No se pudo leer el archivo: {e}"
            )
        
        logger.info("Archivo leído: %s (tamaño=%d bytes)", path, len(content))
        return FileProcessingResult(
            success=True,
            message="Archivo leído exitosamente",
            content=content
        )
    
    def process_whatsapp_file(self, file_obj) -> FileProcessingResult:
        """Process WhatsApp export file and create ChatDataFrame."""
        result = self.read_file_content(file_obj)
        if not result.success:
            return result
        
        content = result.content
        messages = parse_whatsapp_txt(content)
        
        # Create ChatDataFrame for analysis
        chat_dataframe = ChatDataFrame(messages)
        
        if len(messages) == 0:
            preview = "\n".join(content.splitlines()[:3])
            return FileProcessingResult(
                success=False,
                message=(
                    "No se detectaron mensajes. Verifica que el archivo sea un export estándar de WhatsApp (TXT).\n"
                    f"Primeras líneas leídas:\n{preview}"
                ),
                content=content,
                chat_dataframe=chat_dataframe,
                n_messages=0
            )
        
        return FileProcessingResult(
            success=True,
            message=f"Archivo procesado exitosamente - {len(messages)} mensajes detectados",
            content=content,
            chat_dataframe=chat_dataframe,
            n_messages=len(messages)
        )