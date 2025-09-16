#!/usr/bin/env python3
"""
Ejemplo básico de uso del sistema WhatsApp RAG (ES)

Este script demuestra cómo:
1. Cargar un archivo TXT de WhatsApp exportado
2. Inicializar el pipeline RAG
3. Indexar los mensajes
4. Ejecutar consultas contra la base de conocimientos

Requiere:
- Un archivo TXT de WhatsApp exportado
- Configuración opcional de GITHUB_TOKEN para LLM
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar rag
sys.path.append(str(Path(__file__).parent.parent))

from rag.core import parse_whatsapp_txt, RAGPipeline


def main():
    # 1. Cargar archivo de chat de WhatsApp
    data_file = Path(__file__).parent.parent / "data" / "sample_whatsapp.txt"
    
    if not data_file.exists():
        print(f"Error: Archivo de ejemplo no encontrado en {data_file}")
        print("Por favor asegúrate de que existe un archivo de ejemplo en la carpeta data/")
        return
    
    print("=== WhatsApp RAG - Ejemplo Básico ===")
    print(f"Cargando archivo: {data_file}")
    
    # Leer contenido del archivo
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        return
    
    print(f"Archivo cargado: {len(content)} caracteres")
    
    # 2. Parsear mensajes de WhatsApp
    print("\n--- Parseando mensajes ---")
    messages = parse_whatsapp_txt(content)
    print(f"Mensajes parseados: {len(messages)}")
    
    if messages:
        print(f"Primer mensaje: {messages[0].sender} - {messages[0].timestamp} - {messages[0].text[:50]}...")
        print(f"Último mensaje: {messages[-1].sender} - {messages[-1].timestamp} - {messages[-1].text[:50]}...")
    else:
        print("No se encontraron mensajes válidos en el archivo")
        return
    
    # 3. Inicializar pipeline RAG
    print("\n--- Inicializando Pipeline RAG ---")
    pipeline = RAGPipeline()
    
    # 4. Indexar mensajes
    print("Indexando mensajes...")
    pipeline.index_messages(messages)
    
    n_chunks = len(pipeline.chunks)
    vector_size = pipeline.vector_store.size() if pipeline.vector_store else 0
    backend = "FAISS" if (pipeline.vector_store and hasattr(pipeline.vector_store, 'index')) else "numpy"
    
    print("Indexación completada:")
    print(f"  - Chunks creados: {n_chunks}")
    print(f"  - Vectores en índice: {vector_size}")
    print(f"  - Backend de vectores: {backend}")
    
    if n_chunks > 0:
        print(f"  - Ejemplo de chunk: {pipeline.chunks[0].participants} ({pipeline.chunks[0].start_ts})")
    
    # 5. Ejecutar consultas de ejemplo
    print("\n--- Ejecutando Consultas de Ejemplo ---")
    
    queries = [
        "¿A qué hora quedaron?",
        "¿Dónde van a cenar?",
        "¿Quién reservó la mesa?",
        "¿Cuántas personas van?"
    ]
    
    for query in queries:
        print(f"\nPregunta: {query}")
        
        # Recuperar documentos relevantes
        retrieved = pipeline.retrieve(query, top_k=3)
        
        if retrieved:
            print(f"Encontrados {len(retrieved)} fragmentos relevantes:")
            
            # Formatear contexto recuperado
            context = pipeline.format_context(retrieved)
            print("\nContexto recuperado:")
            print("-" * 40)
            print(context)
            print("-" * 40)
            
            # Mostrar información adicional de los fragmentos
            for i, meta in enumerate(retrieved, 1):
                participants = ", ".join(meta.get("participants", []))
                start_ts = meta.get("start_ts", "")
                print(f"  Fragmento {i}: {participants} ({start_ts})")
        else:
            print("No se encontraron fragmentos relevantes")
    
    # Información sobre configuración del LLM (opcional)
    print("\n--- Información Adicional ---")
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        print("✓ GITHUB_TOKEN configurado - LLM disponible")
        print("Para usar el LLM completo, ejecuta app.py o mira llm_example.py")
    else:
        print("ℹ GITHUB_TOKEN no configurado - solo recuperación de documentos")
        print("Para habilitar LLM: export GITHUB_TOKEN=tu_token")
    
    print("\n✓ Ejemplo completado exitosamente")


if __name__ == "__main__":
    main()