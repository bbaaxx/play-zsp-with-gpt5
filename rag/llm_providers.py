import os
import logging
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
    """GitHub Models provider using OpenAI-compatible API."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.environ.get("CHAT_MODEL", "openai/gpt-4o")
        self.token = os.environ.get("GITHUB_TOKEN")
        self.base_url = os.environ.get("GH_MODELS_BASE_URL", "https://models.github.ai/inference")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 800,
        **kwargs
    ) -> str:
        if not self.is_available():
            raise RuntimeError("GitHub Models provider no está disponible")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

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
        except Exception as e:
            logger.error("Error en GitHub Models: %s", e)
            raise

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
        """Generate response using the first available provider."""
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