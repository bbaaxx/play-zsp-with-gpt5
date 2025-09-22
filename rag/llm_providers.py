import os
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import httpx
from dotenv import load_dotenv

# Load .env file with override=True to take precedence over existing env vars
load_dotenv(override=True)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs
    ) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured properly."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the provider name for logging."""
        pass


class GitHubModelsProvider(LLMProvider):
    """GitHub Models provider using OpenAI-compatible API with rate limiting."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get("CHAT_MODEL", "gpt-4o")
        self.token = os.environ.get("GITHUB_TOKEN")
        self.base_url = os.environ.get("GH_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
        
        # Rate limiting state
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # 60 seconds
        self.max_requests_per_window = 8  # Conservative limit (GitHub allows 10, we use 8)
        self.min_request_interval = 6.0  # Minimum 6 seconds between requests
        
        logger.info("GitHub Models provider initialized with model: '%s', base_url: '%s'", self.model_name, self.base_url)

    def _enforce_rate_limit(self):
        """Enforce rate limiting to prevent 429 errors."""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_request_time > self.rate_limit_window:
            self.request_count = 0
        
        # Check if we're approaching the limit
        if self.request_count >= self.max_requests_per_window:
            # Wait for the window to reset
            wait_time = self.rate_limit_window - (current_time - self.last_request_time) + 1
            if wait_time > 0:
                logger.info(f"Rate limit approached ({self.request_count}/{self.max_requests_per_window}), waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self.request_count = 0
        
        # Enforce minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Enforcing minimum interval, waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        # Update tracking
        self.last_request_time = time.time()
        self.request_count += 1

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs
    ) -> str:
        if not self.is_available():
            raise RuntimeError("GitHub Models provider no está disponible")

        # Enforce rate limiting before making request
        self._enforce_rate_limit()

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # Strip vendor prefix from model name for inference API
        # GitHub Models catalog returns "openai/gpt-4o" but inference expects "gpt-4o"
        inference_model_name = self.model_name
        if "/" in inference_model_name:
            inference_model_name = inference_model_name.split("/", 1)[1]

        logger.info("Making API call with model: '%s' (original: '%s')", inference_model_name, self.model_name)

        # Some models (GPT-5 series) use max_completion_tokens instead of max_tokens
        payload = {
            "model": inference_model_name,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Try max_completion_tokens for newer models, fallback to max_tokens
        if "gpt-5" in inference_model_name or "o1" in inference_model_name or "o3" in inference_model_name or "o4" in inference_model_name:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens

        logger.debug("API payload: %s", payload)

        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

                    if "choices" not in data or not data["choices"]:
                        raise RuntimeError("Respuesta inválida de GitHub Models")

                    content = data["choices"][0].get("message", {}).get("content", "")
                    return content or "(sin contenido)"

            except httpx.TimeoutException:
                logger.error("Timeout conectando a GitHub Models")
                raise RuntimeError("Timeout conectando a GitHub Models")
            except httpx.HTTPStatusError as e:
                error_text = e.response.text
                
                # Handle rate limiting with retry
                if e.response.status_code == 429:
                    if attempt < max_retries:
                        # Extract wait time from error message if available
                        wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                        
                        # Try to parse wait time from error message
                        import re
                        wait_match = re.search(r'wait (\d+) seconds', error_text)
                        if wait_match:
                            wait_time = int(wait_match.group(1)) + 1
                        
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}), waiting {wait_time}s before retry")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error("Max retries exceeded for rate limiting")
                        raise RuntimeError("Rate limit exceeded, max retries reached")
                
                logger.error("HTTP error %s en GitHub Models: %s", e.response.status_code, error_text)
                
                # Handle max_tokens vs max_completion_tokens parameter error
                if e.response.status_code == 400 and "max_tokens" in error_text and "max_completion_tokens" in error_text:
                    logger.info("Retrying with max_completion_tokens parameter...")
                    # Retry with max_completion_tokens
                    payload_retry = payload.copy()
                    if "max_tokens" in payload_retry:
                        payload_retry["max_completion_tokens"] = payload_retry.pop("max_tokens")
                    elif "max_completion_tokens" in payload_retry:
                        payload_retry["max_tokens"] = payload_retry.pop("max_completion_tokens")
                    
                    try:
                        with httpx.Client(timeout=60.0) as retry_client:
                            retry_resp = retry_client.post(url, json=payload_retry, headers=headers)
                            retry_resp.raise_for_status()
                            retry_data = retry_resp.json()
                            
                            if "choices" not in retry_data or not retry_data["choices"]:
                                raise RuntimeError("Respuesta inválida de GitHub Models en retry")
                            
                            content = retry_data["choices"][0].get("message", {}).get("content", "")
                            logger.info("Retry successful with different token parameter")
                            return content or "(sin contenido)"
                    except Exception as retry_e:
                        logger.error("Retry también falló: %s", retry_e)
                
                raise RuntimeError(f"HTTP {e.response.status_code} error from GitHub Models")
            except Exception as e:
                if attempt < max_retries:
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Error en GitHub Models: %s", e)
                    raise
        
        # This should not be reached due to the raise statements above
        raise RuntimeError("Unexpected error in generate_response")

    def is_available(self) -> bool:
        return bool(self.token)

    def get_name(self) -> str:
        return f"GitHub Models ({self.model_name})"


class LMStudioProvider(LLMProvider):
    """LM Studio local provider using OpenAI-compatible API."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, chat_model: Optional[str] = None):
        self.host = host or os.environ.get("LMSTUDIO_HOST", "localhost")
        self.port = port or int(os.environ.get("LMSTUDIO_PORT", "1234"))
        self.chat_model = chat_model or os.environ.get("LMSTUDIO_CHAT_MODEL", "local-chat-model")
        self.base_url = f"http://{self.host}:{self.port}/v1"
        self.timeout = float(os.environ.get("LMSTUDIO_TIMEOUT", "360.0"))

        # Test connection timeout for availability check
        self.test_timeout = 5.0

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs
    ) -> str:
        if not self.is_available():
            raise RuntimeError("LM Studio no está disponible")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                
                if "choices" not in data or not data["choices"]:
                    raise RuntimeError("Respuesta inválida de LM Studio")
                
                content = data["choices"][0].get("message", {}).get("content", "")
                reasoning = data["choices"][0].get("message", {}).get("reasoning_content", "")

                # For thinking models, use reasoning_content if content is empty
                final_content = content or reasoning
                return final_content or "(sin contenido)"
                
        except httpx.TimeoutException:
            logger.error("Timeout conectando a LM Studio en %s:%s", self.host, self.port)
            raise RuntimeError(f"Timeout conectando a LM Studio ({self.timeout}s)")
        except httpx.ConnectError:
            logger.error("No se puede conectar a LM Studio en %s:%s", self.host, self.port)
            raise RuntimeError("No se puede conectar a LM Studio")
        except Exception as e:
            logger.error("Error en LM Studio: %s", e)
            raise

    def is_available(self) -> bool:
        """Check if LM Studio server is running and responsive."""
        try:
            url = f"{self.base_url}/models"
            with httpx.Client(timeout=self.test_timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.debug("LM Studio no disponible: %s", e)
            return False

    def get_name(self) -> str:
        return f"LM Studio ({self.host}:{self.port}, {self.chat_model})"


class LLMManager:
    """Manages multiple LLM providers with fallback logic."""

    def __init__(self):
        self.providers: List[LLMProvider] = []
        self._setup_providers()

    def _setup_providers(self):
        """Initialize providers based on configuration."""
        # Add GitHub Models provider first (cloud models)
        github_provider = GitHubModelsProvider()
        if github_provider.is_available():
            self.providers.append(github_provider)
            logger.info("GitHub Models provider agregado: %s", github_provider.get_name())

        # Add LM Studio provider as fallback (local models)
        if os.environ.get("LMSTUDIO_ENABLED", "0") == "1":
            lm_studio = LMStudioProvider()
            self.providers.append(lm_studio)
            logger.info("LM Studio provider agregado: %s", lm_studio.get_name())

        if not self.providers:
            logger.warning("No hay proveedores LLM disponibles")

    def get_available_provider(self) -> Optional[LLMProvider]:
        """Get the first available provider."""
        for provider in self.providers:
            if provider.is_available():
                logger.debug("Usando proveedor: %s", provider.get_name())
                return provider
        return None

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs
    ) -> str:
        """Generate response using the first available provider with intelligent fallback."""
        errors = []
        
        for provider in self.providers:
            if not provider.is_available():
                continue
                
            try:
                logger.info("Intentando con %s", provider.get_name())
                return provider.generate_response(messages, temperature, max_tokens, **kwargs)
            except Exception as e:
                error_msg = f"{provider.get_name()}: {str(e)}"
                errors.append(error_msg)
                
                # Check if this is a rate limiting error that should trigger immediate fallback
                if "Rate limit" in str(e) or "429" in str(e) or "rate limit" in str(e).lower():
                    logger.warning("Rate limiting detected in %s, trying next provider: %s", provider.get_name(), e)
                else:
                    logger.warning("Fallo en %s: %s", provider.get_name(), e)
                continue

        # If we get here, all providers failed
        if errors:
            raise RuntimeError(f"Todos los proveedores LLM fallaron: {'; '.join(errors)}")
        else:
            raise RuntimeError("No hay proveedores LLM disponibles")

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all providers and their availability status."""
        result = []
        for provider in self.providers:
            result.append({
                "name": provider.get_name(),
                "available": provider.is_available()
            })
        return result