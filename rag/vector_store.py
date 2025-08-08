from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

try:  # optional, will fallback to numpy-only search if unavailable
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    HAS_FAISS = False


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class InMemoryFAISS:
    """In-memory vector store with FAISS if available, otherwise numpy search.

    - Cosine similarity via L2-normalized vectors.
    - Python-side metadata kept in parallel arrays.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # used when FAISS is not available
        if HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
        else:
            self.index = None

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Dimensión incorrecta: {embeddings.shape[1]} != {self.dim}")
        normalized = l2_normalize(embeddings)
        if HAS_FAISS and self.index is not None:
            self.index.add(normalized)  # type: ignore[union-attr]
        else:
            if self._matrix is None:
                self._matrix = normalized
            else:
                self._matrix = np.vstack([self._matrix, normalized])
        self.ids.extend(ids)
        self.metadatas.extend(metadatas)

    def search(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype("float32")
        q = l2_normalize(query_embeddings)
        if HAS_FAISS and self.index is not None:
            scores, idxs = self.index.search(q, top_k)  # type: ignore[union-attr]
        else:
            if self._matrix is None or self._matrix.size == 0:
                scores = np.empty((q.shape[0], 0), dtype="float32")
                idxs = np.empty((q.shape[0], 0), dtype=int)
            else:
                # cosine via dot product on normalized vectors
                sims = np.dot(q, self._matrix.T)
                k = min(top_k, sims.shape[1])
                # argpartition for top-k indices per row
                idxs = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
                # gather the corresponding scores
                rows = np.arange(sims.shape[0])[:, None]
                scores = sims[rows, idxs]
                # sort top-k per row by score desc
                order = np.argsort(-scores, axis=1)
                idxs = idxs[rows, order]
                scores = scores[rows, order]

        results: List[List[Dict[str, Any]]] = []
        for row in idxs:
            row_meta: List[Dict[str, Any]] = []
            for j in row:
                if j < 0 or j >= len(self.metadatas):
                    row_meta.append({})
                else:
                    row_meta.append(self.metadatas[j])
            results.append(row_meta)
        return scores, results

    def size(self) -> int:
        return len(self.metadatas)


