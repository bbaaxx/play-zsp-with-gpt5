import os
import logging
from typing import List, Optional

import numpy as np
import requests
import httpx

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional, only when fallback needed
    SentenceTransformer = None  # type: ignore


logger = logging.getLogger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class EmbeddingProvider:
    """Embedding provider with GitHub Models /embeddings HTTP API and local fallback.

    - Preferred: remote embeddings via OpenAI-compatible `/embeddings` endpoint using `GITHUB_TOKEN`.
    - Fallback: sentence-transformers 'intfloat/multilingual-e5-small'.
    """

    def __init__(self) -> None:
        self.use_local_fallback: bool = os.environ.get("USE_LOCAL_EMBEDDINGS", "0") == "1"
        self.remote_model_name: str = os.environ.get(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
        self.local_model_name: str = os.environ.get(
            "LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        )

        self._remote_base_url: Optional[str] = None
        self._remote_token: Optional[str] = None
        self._local_model: Optional[SentenceTransformer] = None

        if not self.use_local_fallback:
            base_url = os.environ.get("GH_MODELS_BASE_URL", "https://models.github.ai/inference")
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                self._remote_base_url = base_url.rstrip("/")
                self._remote_token = token
            else:
                logger.warning("GITHUB_TOKEN ausente; embeddings remotos deshabilitados.")

        if self.use_local_fallback or self._remote_token is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers no está instalado y se requiere para el fallback local"
                )
            self._local_model = SentenceTransformer(self.local_model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts and return L2-normalized float32 vectors.

        Attempts remote embeddings first (if enabled), otherwise falls back to local model.
        """
        texts = [t if isinstance(t, str) else str(t) for t in texts]

        # Try remote first when configured via HTTP
        if not self.use_local_fallback and self._remote_token is not None and self._remote_base_url:
            try:
                url = f"{self._remote_base_url}/embeddings"
                headers = {
                    "Authorization": f"Bearer {self._remote_token}",
                    "Content-Type": "application/json",
                }
                payload = {"model": self.remote_model_name, "input": texts}
                # Use httpx to avoid proxies kwarg incompatibility and to have better control
                with httpx.Client(timeout=60.0) as http:
                    resp = http.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                vectors = np.asarray([d["embedding"] for d in data["data"]], dtype="float32")
                return _l2_normalize(vectors)
            except Exception as e:  # pragma: no cover - network/feature dependent
                logger.warning("Embeddings remotos fallaron: %s. Usando fallback local.", e)

        # Local fallback
        assert self._local_model is not None
        local_vectors = self._local_model.encode(
            texts, normalize_embeddings=False, show_progress_bar=False
        )
        vectors = np.asarray(local_vectors, dtype="float32")
        return _l2_normalize(vectors)


