#!/usr/bin/env python3
"""
Ejemplo de uso con datos personalizados - WhatsApp RAG (ES)

Este script demuestra cómo usar el sistema RAG con tus propios archivos:
1. Carga de archivo TXT personalizado
2. Configuración avanzada de parámetros
3. Filtros por remitente y fecha
4. Análisis detallado de resultados

Uso:
    python custom_data_example.py /ruta/a/tu/whatsapp_export.txt
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar rag
sys.path.append(str(Path(__file__).parent.parent))

from rag.core import parse_whatsapp_txt, RAGPipeline


def analyze_chat_statistics(messages):
    """Analiza estadísticas básicas del chat"""
    if not messages:
        return
    
    # Estadísticas por remitente
    sender_stats = {}
    for msg in messages:
        if msg.sender not in sender_stats:
            sender_stats[msg.sender] = {"count": 0, "first": msg.timestamp, "last": msg.timestamp}
        
        sender_stats[msg.sender]["count"] += 1
        sender_stats[msg.sender]["last"] = max(sender_stats[msg.sender]["last"], msg.timestamp)
        sender_stats[msg.sender]["first"] = min(sender_stats[msg.sender]["first"], msg.timestamp)
    
    print("\n📊 Estadísticas del Chat:")
    print(f"  Total de mensajes: {len(messages)}")
    print(f"  Período: {messages[0].timestamp.date()} - {messages[-1].timestamp.date()}")
    print(f"  Participantes: {len(sender_stats)}")
    
    print("\n👥 Por remitente:")
    for sender, stats in sorted(sender_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        print(f"  - {sender}: {stats['count']} mensajes ({stats['first'].date()} - {stats['last'].date()})")


def demonstrate_filtering(pipeline, query):
    """Demuestra diferentes opciones de filtrado"""
    print(f"\n🔍 Consulta: '{query}'")
    
    # Consulta básica
    basic_results = pipeline.retrieve(query, top_k=3)
    print(f"  Resultados sin filtros: {len(basic_results)} fragmentos")
    
    # Obtener lista de remitentes únicos
    all_senders = set()
    for chunk in pipeline.chunks:
        all_senders.update(chunk.participants)
    
    # Filtrar por remitente (si hay más de uno)
    if len(all_senders) > 1:
        first_sender = list(all_senders)[0]
        filtered_results = pipeline.retrieve(
            query, 
            top_k=3, 
            senders=[first_sender]
        )
        print(f"  Filtrado por '{first_sender}': {len(filtered_results)} fragmentos")
    
    # Demostrar diferentes algoritmos de recuperación
    mmr_results = pipeline.retrieve(query, top_k=3, use_mmr=True, lambda_=0.7)
    similarity_results = pipeline.retrieve(query, top_k=3, use_mmr=False)
    
    print(f"  Con MMR (λ=0.7): {len(mmr_results)} fragmentos")
    print(f"  Solo similitud: {len(similarity_results)} fragmentos")
    
    return basic_results


def main():
    print("=== WhatsApp RAG - Ejemplo con Datos Personalizados ===")
    
    # Verificar argumentos
    if len(sys.argv) < 2:
        print("Uso: python custom_data_example.py <archivo_whatsapp.txt>")
        print("\nEjemplo con archivo de muestra:")
        data_file = Path(__file__).parent.parent / "data" / "sample_whatsapp.txt"
        
        if not data_file.exists():
            print(f"Error: Archivo de muestra no encontrado en {data_file}")
            return
        
        file_path = data_file
        print(f"Usando archivo de muestra: {file_path}")
    else:
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"Error: Archivo no encontrado: {file_path}")
            return
    
    # Cargar archivo
    print(f"\nCargando archivo: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        print("Intentando con encoding latin-1...")
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e2:
            print(f"Error con latin-1: {e2}")
            return
    
    print(f"Archivo cargado: {len(content):,} caracteres")
    
    # Parsear mensajes
    messages = parse_whatsapp_txt(content)
    
    if not messages:
        print("❌ No se encontraron mensajes válidos en el archivo")
        print("\nFormatos soportados:")
        print("  [12/10/2023, 21:15] Juan: Mensaje")
        print("  12/10/23, 21:22 - María: Mensaje")
        print("  [26/05/25, 3:18:25 p.m.] Pedro: Mensaje")
        return
    
    # Mostrar estadísticas
    analyze_chat_statistics(messages)
    
    # Configurar pipeline
    print("\n⚙️ Configurando Pipeline RAG...")
    pipeline = RAGPipeline()
    
    # Configuración personalizada de chunking
    print("Indexando mensajes...")
    pipeline.index_messages(messages)
    
    n_chunks = len(pipeline.chunks)
    vector_size = pipeline.vector_store.size() if pipeline.vector_store else 0
    
    print(f"✓ Indexación completada: {n_chunks} chunks, {vector_size} vectores")
    
    if n_chunks == 0:
        print("❌ No se pudieron crear chunks. Verifica el formato del archivo.")
        return
    
    # Mostrar ejemplos de chunks
    print("\n📦 Ejemplos de chunks creados:")
    for i, chunk in enumerate(pipeline.chunks[:3]):
        participants = ", ".join(chunk.participants)
        duration = chunk.end_ts - chunk.start_ts
        print(f"  {i+1}. {participants} | {chunk.start_ts.strftime('%Y-%m-%d %H:%M')} | {duration} | líneas {chunk.line_span[0]}-{chunk.line_span[1]}")
        # Mostrar primeras líneas del chunk
        preview = "\n".join(chunk.text_window.split("\n")[:2])
        print(f"     {preview}...")
    
    # Demostrar diferentes tipos de consultas
    example_queries = [
        "¿De qué hablaron?",
        "¿Cuándo fue la última vez que mencionaron trabajo?",
        "¿Qué planes tienen para el fin de semana?"
    ]
    
    print("\n🎯 Ejecutando consultas de ejemplo:")
    
    for query in example_queries:
        results = demonstrate_filtering(pipeline, query)
        
        if results:
            # Mostrar el mejor resultado
            best_result = results[0]
            context_preview = pipeline.format_context([best_result])
            print("  Mejor resultado:")
            print(f"    Participantes: {', '.join(best_result.get('participants', []))}")
            print(f"    Período: {best_result.get('start_ts', '')} - {best_result.get('end_ts', '')}")
            preview_lines = context_preview.split("\n")[:2]
            print(f"    Vista previa: {' | '.join(preview_lines)}")
    
    # Configuraciones avanzadas
    print("\n⚙️ Configuraciones Avanzadas Disponibles:")
    print("  - top_k: Número de fragmentos a recuperar (1-20)")
    print("  - use_mmr: Usar Maximum Marginal Relevance para diversidad")
    print("  - lambda_: Balance relevancia/diversidad en MMR (0.0-1.0)")
    print("  - fetch_k: Candidatos iniciales para MMR (mayor = más diverse)")
    print("  - senders: Filtrar por remitentes específicos")
    print("  - date_from_iso/date_to_iso: Filtrar por rango de fechas")
    
    # Ejemplo de configuración avanzada
    print("\n🔧 Ejemplo de configuración avanzada:")
    advanced_config = {
        "top_k": 7,
        "use_mmr": True,
        "lambda_": 0.3,  # Más diversidad
        "fetch_k": 50
    }
    
    test_query = "¿Qué mensaje es más importante?"
    print(f"Consulta: '{test_query}'")
    print(f"Configuración: {advanced_config}")
    
    advanced_results = pipeline.retrieve(test_query, **advanced_config)
    print(f"Resultados: {len(advanced_results)} fragmentos con mayor diversidad")
    
    print(f"\n✓ Ejemplo completado. Tu archivo tiene {len(messages)} mensajes listos para consultar!")
    
    # Sugerencias para siguiente paso
    if os.environ.get("GITHUB_TOKEN"):
        print("\n💡 Siguiente paso: Ejecuta llm_example.py para respuestas con IA")
    else:
        print("\n💡 Para respuestas con IA: configura GITHUB_TOKEN y ejecuta llm_example.py")


if __name__ == "__main__":
    main()