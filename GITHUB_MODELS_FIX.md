# GitHub Models API Fix

## Issues Fixed

1. **Incorrect API Endpoint**: The app was using `https://models.github.ai/inference` which is incorrect for GitHub Models
2. **Wrong Model Names**: Models were prefixed with provider names (e.g., `openai/gpt-4o`) when they should be just the model name (e.g., `gpt-4o`)
3. **Outdated Model List**: The fallback model list didn't include current available models

## Changes Made

### 1. Updated API Endpoints

**Old (incorrect)**: 
```python
base_url = "https://models.github.ai/inference"
```

**New (correct)**:
```python
base_url = "https://models.inference.ai.azure.com"
```

### 2. Fixed Model Fetching Logic

- Updated `_fetch_github_models()` to use the correct GitHub API endpoint: `https://api.github.com/models`
- Fixed response parsing to handle GitHub Models API format properly
- Improved filtering to distinguish between text generation and embedding models

### 3. Updated Model Names

**Old (incorrect)**:
- `openai/gpt-4o`
- `meta-llama/Llama-3.2-3B-Instruct`

**New (correct)**:
- `gpt-4o`
- `Llama-3.2-3B-Instruct`

### 4. Enhanced Fallback Models List

Added comprehensive fallback list including:
- OpenAI models: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, etc.
- Meta Llama models: `Llama-3.2-3B-Instruct`, `Meta-Llama-3.1-8B-Instruct`, etc.
- Microsoft Phi models: `Phi-3.5-mini-instruct`, `Phi-3-medium-128k-instruct`, etc.
- Mistral models: `Mistral-7B-Instruct-v0.3`, `Mistral-large`, etc.
- Cohere models: `Cohere-command-r`, `Cohere-command-r-plus`
- AI21 models: `jamba-1.5-large`, `jamba-1.5-mini`

### 5. Fixed Headers and Error Handling

- Removed unnecessary headers that were causing 404 errors
- Added better HTTP error handling with status code logging
- Improved timeout handling

## Files Modified

1. `app.py` - Main model fetching functions
2. `rag/llm_providers.py` - LLM provider implementation
3. `rag/embeddings.py` - Embedding provider endpoint
4. `business/chat_processor.py` - Chat processor legacy client
5. `ui/handlers.py` - UI handler endpoint
6. `examples/llm_example.py` - Example endpoint
7. `examples/advanced/api_integration.py` - Advanced example endpoint
8. `.env.example` - Environment variables template

## Testing

The fix ensures that:
1. The dropdown now shows the correct, current list of available models
2. Llama models (like `Llama-3.2-3B-Instruct`) are properly included
3. API calls use the correct endpoint and model names
4. The 404 error when using Llama models is resolved

## Environment Variables

Update your `.env` file with the correct values:
```bash
GH_MODELS_BASE_URL=https://models.inference.ai.azure.com
CHAT_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
```