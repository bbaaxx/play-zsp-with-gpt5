#!/usr/bin/env python3
"""
Demo del módulo de análisis de datos de WhatsApp.
Muestra cómo usar ChatDataFrame para análisis estructurado de mensajes.
"""

from pathlib import Path

from rag import ChatDataFrame


def main():
    """Demuestra el uso del módulo de análisis de datos."""
    
    # Cargar datos desde archivo de ejemplo
    data_path = Path(__file__).parent.parent / "data" / "sample_whatsapp.txt"
    
    print("=== Demo: Análisis de Datos de WhatsApp ===\n")
    
    # Inicializar el analizador
    analyzer = ChatDataFrame()
    analyzer.load_from_file(data_path)
    
    print(f"📊 {analyzer}")
    print()
    
    # Mostrar estadísticas básicas
    stats = analyzer.get_message_stats()
    print("📈 Estadísticas del Chat:")
    print(f"  • Total mensajes: {stats['total_messages']}")
    print(f"  • Autores únicos: {stats['unique_authors']}")
    print(f"  • Rango de fechas: {stats['date_range']['start'].date()} - {stats['date_range']['end'].date()}")
    print(f"  • Mensajes por día: {stats['messages_per_day']:.1f}")
    print(f"  • Longitud promedio: {stats['avg_message_length']:.1f} caracteres")
    print(f"  • Hora más activa: {stats['most_active_hour']}:00")
    print()
    
    print("👥 Mensajes por autor:")
    for author, count in stats['authors'].items():
        print(f"  • {author}: {count} mensajes")
    print()
    
    # Filtrar por autor
    print("🔍 Filtrar mensajes de Juan:")
    juan_messages = analyzer.filter_by_author("Juan")
    print(f"  Encontrados: {len(juan_messages)} mensajes")
    for msg in juan_messages.to_messages():
        print(f"  [{msg.timestamp.strftime('%H:%M')}] {msg.sender}: {msg.text}")
    print()
    
    # Filtrar por contenido
    print("🔎 Buscar mensajes con 'hora':")
    time_messages = analyzer.filter_by_content("hora", case_sensitive=False)
    print(f"  Encontrados: {len(time_messages)} mensajes")
    for msg in time_messages.to_messages():
        print(f"  [{msg.timestamp.strftime('%H:%M')}] {msg.sender}: {msg.text}")
    print()
    
    # Filtrar por fecha
    print("📅 Filtrar por fecha específica (12/10/2023):")
    date_messages = analyzer.filter_by_date_range(
        start_date="2023-10-12",
        end_date="2023-10-12"
    )
    print(f"  Encontrados: {len(date_messages)} mensajes")
    print()
    
    # Buscar palabras clave con contexto
    print("🔍 Buscar 'mesa' con contexto:")
    keyword_results = analyzer.search_keywords("mesa", context_window=1)
    if not keyword_results.empty:
        for _, row in keyword_results.iterrows():
            marker = ">>> " if row['is_match'] else "    "
            print(f"  {marker}[{row['timestamp'].strftime('%H:%M')}] {row['author']}: {row['message']}")
    print()
    
    # Mostrar DataFrame completo
    print("📋 DataFrame completo:")
    df = analyzer.df
    df_display = df.copy()
    df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    print(df_display.to_string(index=False))
    print()
    
    # Información de tipos de datos
    print("🏷️  Tipos de datos optimizados:")
    print(df.dtypes)


if __name__ == "__main__":
    main()