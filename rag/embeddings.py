import os
import logging
from typing import List, Optional, Iterable

import numpy as np
import httpx
from dotenv import load_dotenv

# Load .env file with override=True to take precedence over existing env vars
load_dotenv(override=True)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional, only when fallback needed
    SentenceTransformer = None  # type: ignore


logger = logging.getLogger(__name__)


class LMStudioEmbeddingProvider:
    """LM Studio embedding provider using OpenAI-compatible /embeddings API."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, model_name: Optional[str] = None):
        self.host = host or os.environ.get("LMSTUDIO_HOST", "localhost")
        self.port = port or int(os.environ.get("LMSTUDIO_PORT", "1234"))
        self.model_name = model_name or os.environ.get("LMSTUDIO_EMBEDDING_MODEL", "local-embedding-model")
        self.base_url = f"http://{self.host}:{self.port}/v1"
        self.timeout = float(os.environ.get("LMSTUDIO_TIMEOUT", "60.0"))
        self.test_timeout = 5.0

    def is_available(self) -> bool:
        """Check if LM Studio server is running and supports embeddings."""
        try:
            url = f"{self.base_url}/models"
            with httpx.Client(timeout=self.test_timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.debug("LM Studio embeddings no disponible: %s", e)
            return False

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using LM Studio's /embeddings endpoint."""
        if not self.is_available():
            raise RuntimeError("LM Studio no está disponible para embeddings")

        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model_name,
            "input": texts
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                
                if "data" not in data:
                    raise RuntimeError("Respuesta inválida de LM Studio embeddings")
                
                embeddings = np.array([item["embedding"] for item in data["data"]], dtype="float32")
                return embeddings
                
        except httpx.TimeoutException:
            logger.error("Timeout en LM Studio embeddings (%s:%s)", self.host, self.port)
            raise RuntimeError(f"Timeout en LM Studio embeddings ({self.timeout}s)")
        except httpx.ConnectError:
            logger.error("No se puede conectar a LM Studio embeddings (%s:%s)", self.host, self.port)
            raise RuntimeError("No se puede conectar a LM Studio embeddings")
        except Exception as e:
            logger.error("Error en LM Studio embeddings: %s", e)
            raise


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class EmbeddingProvider:
    """Embedding provider with multiple backends: LM Studio, GitHub Models, and local fallback.

    - Priority 1: LM Studio (if enabled and available)
    - Priority 2: GitHub Models (if token available) 
    - Priority 3: Local sentence-transformers fallback
    """

    def __init__(self) -> None:
        self.use_local_fallback: bool = os.environ.get("USE_LOCAL_EMBEDDINGS", "0") == "1"
        self.use_lm_studio: bool = os.environ.get("LMSTUDIO_EMBEDDINGS_ENABLED", "0") == "1"
        
        # Model names for different providers
        self.remote_model_name: str = os.environ.get(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
        self.local_model_name: str = os.environ.get(
            "LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        )

        # Batching controls to avoid 413 Payload Too Large and keep requests manageable
        # - Max items per remote request
        self.batch_size: int = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
        # - Max total characters per remote request payload (rough heuristic)
        self.max_chars_per_request: int = int(os.environ.get("EMBED_MAX_CHARS_PER_REQUEST", "60000"))
        # - Max characters per individual text; longer inputs are truncated
        self.max_chars_per_item: int = int(os.environ.get("EMBED_MAX_CHARS_PER_ITEM", "4000"))

        # Initialize providers
        self._lm_studio_provider: Optional[LMStudioEmbeddingProvider] = None
        self._remote_base_url: Optional[str] = None
        self._remote_token: Optional[str] = None
        self._local_model: Optional[SentenceTransformer] = None

        # Setup LM Studio provider
        if self.use_lm_studio:
            self._lm_studio_provider = LMStudioEmbeddingProvider()

        # Setup GitHub Models provider
        if not self.use_local_fallback:
            base_url = os.environ.get("GH_MODELS_BASE_URL", "https://models.inference.ai.azure.com")
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                self._remote_base_url = base_url.rstrip("/")
                self._remote_token = token
            else:
                logger.warning("GITHUB_TOKEN ausente; embeddings remotos deshabilitados.")

        # Setup local fallback if needed
        if self.use_local_fallback or (self._remote_token is None and not self.use_lm_studio):
            self._ensure_local_model()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts and return L2-normalized float32 vectors.

        Tries providers in order: LM Studio -> GitHub Models -> Local fallback
        """
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        # Truncate overly long items to keep payloads small and consistent across providers
        texts = [t if len(t) <= self.max_chars_per_item else t[: self.max_chars_per_item] for t in texts]

        # Try LM Studio first if enabled
        if self.use_lm_studio and self._lm_studio_provider is not None:
            try:
                logger.info("Usando LM Studio para embeddings")
                vectors = self._lm_studio_provider.embed_texts(texts)
                return _l2_normalize(vectors)
            except Exception as e:
                logger.warning("LM Studio embeddings falló: %s. Probando GitHub Models.", e)

        # Try GitHub Models second if configured
        if not self.use_local_fallback and self._remote_token is not None and self._remote_base_url:
            try:
                logger.info("Usando GitHub Models para embeddings")
                vectors_list: List[np.ndarray] = []
                for batch in self._iter_batches(texts):
                    batch_vectors = self._embed_remote_batch(batch)
                    vectors_list.append(batch_vectors)
                vectors = np.vstack(vectors_list) if vectors_list else np.zeros((0, 0), dtype="float32")
                return _l2_normalize(vectors)
            except Exception as e:  # pragma: no cover - network/feature dependent
                logger.warning("GitHub Models embeddings falló: %s. Usando fallback local.", e)

        # Local fallback
        logger.info("Usando modelo local para embeddings")
        self._ensure_local_model()
        assert self._local_model is not None
        local_vectors = self._local_model.encode(
            texts, normalize_embeddings=False, show_progress_bar=False
        )
        vectors = np.asarray(local_vectors, dtype="float32")
        return _l2_normalize(vectors)

    # -------------------- Helpers --------------------

    def _ensure_local_model(self) -> None:
        if self._local_model is not None:
            return
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers no está instalado y se requiere para el fallback local"
            )
        logger.info("Cargando modelo local de embeddings: %s", self.local_model_name)
        self._local_model = SentenceTransformer(self.local_model_name)

    def _iter_batches(self, texts: List[str]) -> Iterable[List[str]]:
        if not texts:
            return []  # type: ignore[return-value]
        batch: List[str] = []
        chars_in_batch = 0
        for t in texts:
            t_len = len(t)
            # If adding this text exceeds either constraint and we already have items, yield current batch
            if batch and (
                len(batch) >= self.batch_size or (chars_in_batch + t_len) > self.max_chars_per_request
            ):
                yield batch
                batch = []
                chars_in_batch = 0
            batch.append(t)
            chars_in_batch += t_len
            # If batch becomes exactly full by count, flush it
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
                chars_in_batch = 0
        if batch:
            yield batch

    def _embed_remote_batch(self, batch: List[str]) -> np.ndarray:
        url = f"{self._remote_base_url}/embeddings"  # type: ignore[operator]
        headers = {
            "Authorization": f"Bearer {self._remote_token}",  # type: ignore[operator]
            "Content-Type": "application/json",
        }
        
        # Strip vendor prefix from model name for inference API
        # GitHub Models catalog returns "openai/text-embedding-3-small" but inference expects "text-embedding-3-small"
        inference_model_name = self.remote_model_name
        if "/" in inference_model_name:
            inference_model_name = inference_model_name.split("/", 1)[1]
        
        payload = {"model": inference_model_name, "input": batch}
        with httpx.Client(timeout=60.0) as http:
            resp = http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        vectors = np.asarray([d["embedding"] for d in data["data"]], dtype="float32")
        return vectors


