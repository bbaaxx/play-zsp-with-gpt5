#!/usr/bin/env python3
"""
Script to filter and sort GitHub Models by context size (max_input_tokens)
and display them in a formatted table.
"""

import json
import sys
from tabulate import tabulate

def load_models(json_file):
    """Load models from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{json_file}': {e}")
        sys.exit(1)

def format_tokens(tokens):
    """Format token numbers for better readability."""
    if tokens is None:
        return "N/A"
    if tokens >= 1_000_000:
        return f"{tokens/1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens/1_000:.0f}K"
    else:
        return str(tokens)

def filter_and_sort_models(models):
    """Filter out embedding models and sort by max_input_tokens (descending)."""
    # Filter out embedding models (they have different output modalities)
    text_models = [
        model for model in models 
        if 'embeddings' not in model.get('supported_output_modalities', [])
    ]
    
    # Sort by max_input_tokens in descending order
    # Handle None values by treating them as 0
    return sorted(
        text_models, 
        key=lambda x: x.get('limits', {}).get('max_input_tokens', 0), 
        reverse=True
    )

def create_table_data(models):
    """Create table data from sorted models."""
    table_data = []
    for i, model in enumerate(models, 1):
        limits = model.get('limits', {})
        max_input = limits.get('max_input_tokens')
        max_output = limits.get('max_output_tokens')
        
        # Format the row
        row = [
            i,  # Rank
            model.get('name', 'N/A'),
            model.get('publisher', 'N/A'),
            f"{max_input:,}" if max_input else "N/A",  # Exact token count
            format_tokens(max_input),  # Human readable
            format_tokens(max_output),
            ', '.join(model.get('tags', [])[:3])  # Limit to first 3 tags
        ]
        table_data.append(row)
    
    return table_data

def main():
    """Main function."""
    json_file = 'availableModelsGithub.json'
    
    print("🔍 Loading GitHub Models data...")
    models = load_models(json_file)
    
    print(f"📊 Found {len(models)} total models")
    
    print("🔧 Filtering and sorting by context size...")
    sorted_models = filter_and_sort_models(models)
    
    print(f"📋 Found {len(sorted_models)} text generation models")
    
    # Create table
    headers = [
        "Rank",
        "Model Name",
        "Publisher",
        "Exact Tokens",
        "Max Input",
        "Max Output",
        "Tags"
    ]
    
    table_data = create_table_data(sorted_models)
    
    print("\n" + "="*100)
    print("🏆 GITHUB MODELS SORTED BY CONTEXT SIZE (MAX INPUT TOKENS)")
    print("="*100)
    
    # Print the table
    print(tabulate(
        table_data,
        headers=headers,
        tablefmt="grid",
        maxcolwidths=[None, 35, 15, 15, None, None, 30]
    ))
    
    print(f"\n📈 Context size ranges:")
    print(f"   • Largest: {format_tokens(sorted_models[0]['limits']['max_input_tokens'])}")
    print(f"   • Smallest: {format_tokens(sorted_models[-1]['limits']['max_input_tokens'])}")
    
    # Show top 5 models with their exact token counts
    print(f"\n🥇 TOP 5 MODELS BY CONTEXT SIZE:")
    for i, model in enumerate(sorted_models[:5], 1):
        tokens = model['limits']['max_input_tokens']
        print(f"   {i}. {model['name']}: {tokens:,} tokens")

if __name__ == "__main__":
    main()