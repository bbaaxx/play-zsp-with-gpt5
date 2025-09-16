#!/usr/bin/env python3
"""
Ejemplo de uso completo con LLM - WhatsApp RAG (ES)

Este script demuestra cómo usar el sistema RAG completo incluyendo:
1. Carga e indexación de mensajes
2. Recuperación de contexto relevante
3. Generación de respuestas usando LLM (GitHub Models/OpenAI)

Requiere:
- GITHUB_TOKEN configurado en el entorno
- Un archivo TXT de WhatsApp exportado
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar rag
sys.path.append(str(Path(__file__).parent.parent))

from rag.core import parse_whatsapp_txt, RAGPipeline, build_user_prompt, SYSTEM_PROMPT

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai no está instalado")
    print("Instalar con: pip install openai")
    sys.exit(1)


def create_llm_client():
    """Crea cliente LLM usando variables de entorno"""
    token = os.environ.get("GITHUB_TOKEN")
    base_url = os.environ.get("GH_MODELS_BASE_URL", "https://models.github.ai/inference")
    
    if not token:
        print("Error: GITHUB_TOKEN no configurado")
        print("Configurar con: export GITHUB_TOKEN=tu_token")
        return None
    
    print(f"Conectando a LLM (base_url: {base_url})")
    return OpenAI(api_key=token, base_url=base_url)


def generate_response(client, context: str, question: str, model: str = "openai/gpt-4o") -> str:
    """Genera respuesta usando LLM con el contexto recuperado"""
    user_prompt = build_user_prompt(context, question)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content or "(sin contenido)"
    except Exception as e:
        return f"Error al generar respuesta: {e}"


def main():
    print("=== WhatsApp RAG con LLM - Ejemplo Completo ===")
    
    # Verificar configuración de LLM
    client = create_llm_client()
    if not client:
        return
    
    # Cargar archivo de datos
    data_file = Path(__file__).parent.parent / "data" / "sample_whatsapp.txt"
    
    if not data_file.exists():
        print(f"Error: Archivo de ejemplo no encontrado en {data_file}")
        return
    
    print(f"Cargando archivo: {data_file}")
    
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        return
    
    # Parsear e indexar mensajes
    print("\n--- Procesando Mensajes ---")
    messages = parse_whatsapp_txt(content)
    print(f"Mensajes parseados: {len(messages)}")
    
    if not messages:
        print("No se encontraron mensajes válidos")
        return
    
    # Inicializar y configurar pipeline
    pipeline = RAGPipeline()
    pipeline.index_messages(messages)
    
    print(f"Chunks indexados: {len(pipeline.chunks)}")
    print(f"Vectores en índice: {pipeline.vector_store.size()}")
    
    # Configuración de recuperación
    retrieval_config = {
        "top_k": 5,
        "use_mmr": True,
        "lambda_": 0.5,
        "fetch_k": 25
    }
    
    print(f"\nConfiguración de recuperación: {retrieval_config}")
    
    # Preguntas de ejemplo
    questions = [
        "¿A qué hora quedaron para salir?",
        "¿Dónde van a encontrarse?",
        "¿Quién va a reservar la mesa y para cuántas personas?",
        "¿En qué restaurante van a cenar?",
        "Resume la conversación sobre la cena"
    ]
    
    print("\n--- Ejecutando Consultas con LLM ---")
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Pregunta {i}: {question}")
        print("="*60)
        
        # Recuperar contexto relevante
        retrieved = pipeline.retrieve(question, **retrieval_config)
        
        if not retrieved:
            print("❌ No se encontró contexto relevante")
            continue
        
        # Formatear contexto
        context = pipeline.format_context(retrieved)
        
        print(f"\n📝 Contexto recuperado ({len(retrieved)} fragmentos):")
        print("-" * 40)
        print(context)
        print("-" * 40)
        
        # Generar respuesta con LLM
        print("\n🤖 Respuesta del LLM:")
        print("-" * 40)
        
        model_name = os.environ.get("CHAT_MODEL", "openai/gpt-4o")
        response = generate_response(client, context, question, model_name)
        
        print(response)
        print("-" * 40)
        
        # Mostrar información de los fragmentos recuperados
        print("\n📊 Detalles de fragmentos recuperados:")
        for j, meta in enumerate(retrieved, 1):
            participants = ", ".join(meta.get("participants", []))
            start_ts = meta.get("start_ts", "")
            line_span = meta.get("line_span", [])
            print(f"  {j}. {participants} | {start_ts} | líneas {line_span[0]}-{line_span[1]}")
    
    print(f"\n{'='*60}")
    print("✓ Ejemplo completado exitosamente")
    print("\nConfiguraciones disponibles:")
    print("- CHAT_MODEL: Modelo LLM a usar (actual: {})".format(os.environ.get("CHAT_MODEL", "openai/gpt-4o")))
    print("- GH_MODELS_BASE_URL: URL base para GitHub Models")
    print("- Modificar retrieval_config en el código para ajustar la recuperación")


if __name__ == "__main__":
    main()