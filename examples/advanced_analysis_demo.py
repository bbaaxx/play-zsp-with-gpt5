#!/usr/bin/env python3
"""
Demo avanzado del módulo de análisis de datos de WhatsApp.
Muestra análisis temporal y exportación de datos.
"""

from pathlib import Path
import tempfile

from rag import ChatDataFrame


def main():
    """Demuestra análisis avanzado y exportación de datos."""
    
    # Cargar datos desde archivo de ejemplo
    data_path = Path(__file__).parent.parent / "data" / "sample_whatsapp.txt"
    
    print("=== Demo Avanzado: Análisis Temporal y Exportación ===\n")
    
    # Inicializar el analizador
    analyzer = ChatDataFrame()
    analyzer.load_from_file(data_path)
    
    print(f"📊 {analyzer}")
    print()
    
    # Análisis de actividad diaria
    print("📅 Actividad diaria por autor:")
    daily_activity = analyzer.get_daily_activity()
    if not daily_activity.empty:
        print(daily_activity.to_string())
    print()
    
    # Análisis de actividad por hora
    print("⏰ Actividad por hora del día:")
    hourly_activity = analyzer.get_hourly_activity()
    if not hourly_activity.empty:
        print(hourly_activity.to_string())
    print()
    
    # Filtrado combinado: autor y contenido
    print("🔍 Filtrado combinado - Pedro + palabras con números:")
    pedro_messages = analyzer.filter_by_author("Pedro")
    pedro_with_numbers = pedro_messages.filter_by_content(r'\d+', regex=True)
    print(f"  Encontrados: {len(pedro_with_numbers)} mensajes")
    for msg in pedro_with_numbers.to_messages():
        print(f"  [{msg.timestamp.strftime('%H:%M')}] {msg.sender}: {msg.text}")
    print()
    
    # Búsqueda de múltiples palabras clave
    print("🔎 Buscar múltiples palabras: ['mesa', 'hora', 'centro']")
    multi_search = analyzer.search_keywords(['mesa', 'hora', 'centro'], context_window=1)
    if not multi_search.empty:
        print("  Resultados con contexto:")
        for _, row in multi_search.iterrows():
            marker = ">>> " if row['is_match'] else "    "
            print(f"  {marker}[{row['timestamp'].strftime('%H:%M')}] {row['author']}: {row['message']}")
    print()
    
    # Filtrado por coincidencia parcial de autor
    print("👥 Filtrado por coincidencia parcial de autor (nombres que contengan 'a'):")
    partial_author = analyzer.filter_by_author(['a'], exact_match=False)
    print(f"  Encontrados: {len(partial_author)} mensajes")
    for msg in partial_author.to_messages():
        print(f"  [{msg.timestamp.strftime('%H:%M')}] {msg.sender}: {msg.text}")
    print()
    
    # Análisis de longitud de mensajes
    print("📏 Análisis de longitud de mensajes:")
    df = analyzer.df
    for author in df['author'].cat.categories:
        author_messages = df[df['author'] == author]['message']
        lengths = author_messages.str.len()
        print(f"  {author}:")
        print(f"    Promedio: {lengths.mean():.1f} caracteres")
        print(f"    Min/Max: {lengths.min():.0f}/{lengths.max():.0f} caracteres")
    print()
    
    # Exportación a CSV
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_file:
        csv_path = tmp_file.name
    
    try:
        analyzer.export_to_csv(csv_path)
        print(f"✅ Datos exportados a: {csv_path}")
        
        # Leer de vuelta para verificar
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"   Archivo CSV tiene {len(lines)} líneas (incluyendo header)")
        print("   Primeras 3 líneas:")
        for i, line in enumerate(lines[:3], 1):
            print(f"     {i}: {line.strip()}")
    
    finally:
        # Limpiar archivo temporal
        Path(csv_path).unlink(missing_ok=True)
    
    print()
    
    # Demostrar carga desde mensajes existentes
    print("🔄 Cargar desde mensajes existentes:")
    messages = analyzer.to_messages()
    new_analyzer = ChatDataFrame(messages)
    print(f"  {new_analyzer}")
    print(f"  Los datos son idénticos: {len(analyzer.df) == len(new_analyzer.df)}")


if __name__ == "__main__":
    main()