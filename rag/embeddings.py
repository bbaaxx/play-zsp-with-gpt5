import os
import logging
from typing import List, Optional, Iterable

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

        # Batching controls to avoid 413 Payload Too Large and keep requests manageable
        # - Max items per remote request
        self.batch_size: int = int(os.environ.get("EMBED_BATCH_SIZE", "64"))
        # - Max total characters per remote request payload (rough heuristic)
        self.max_chars_per_request: int = int(os.environ.get("EMBED_MAX_CHARS_PER_REQUEST", "60000"))
        # - Max characters per individual text; longer inputs are truncated
        self.max_chars_per_item: int = int(os.environ.get("EMBED_MAX_CHARS_PER_ITEM", "4000"))

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
            self._ensure_local_model()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts and return L2-normalized float32 vectors.

        Attempts remote embeddings first (if enabled), otherwise falls back to local model.
        """
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        # Truncate overly long items to keep payloads small and consistent across providers
        texts = [t if len(t) <= self.max_chars_per_item else t[: self.max_chars_per_item] for t in texts]

        # Try remote first when configured via HTTP
        if not self.use_local_fallback and self._remote_token is not None and self._remote_base_url:
            try:
                vectors_list: List[np.ndarray] = []
                for batch in self._iter_batches(texts):
                    batch_vectors = self._embed_remote_batch(batch)
                    vectors_list.append(batch_vectors)
                vectors = np.vstack(vectors_list) if vectors_list else np.zeros((0, 0), dtype="float32")
                return _l2_normalize(vectors)
            except Exception as e:  # pragma: no cover - network/feature dependent
                logger.warning("Embeddings remotos fallaron: %s. Usando fallback local.", e)

        # Local fallback
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
        payload = {"model": self.remote_model_name, "input": batch}
        with httpx.Client(timeout=60.0) as http:
            resp = http.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        vectors = np.asarray([d["embedding"] for d in data["data"]], dtype="float32")
        return vectors


