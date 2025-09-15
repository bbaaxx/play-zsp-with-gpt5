from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Iterable
import json
import os

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
        # Always keep a normalized embeddings matrix to enable MMR and filtering
        self._matrix: Optional[np.ndarray] = None
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
        # Always maintain _matrix for post-processing
        if self._matrix is None:
            self._matrix = normalized
        else:
            self._matrix = np.vstack([self._matrix, normalized])
        if HAS_FAISS and self.index is not None:
            self.index.add(normalized)  # type: ignore[union-attr]
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

    # ---------- Filtering helpers ----------

    def _indices_by_filters(
        self,
        senders: Optional[Iterable[str]] = None,
        date_from_iso: Optional[str] = None,
        date_to_iso: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        if not senders and not date_from_iso and not date_to_iso:
            return None
        mask = np.ones(self.size(), dtype=bool)
        if senders:
            senders_set = set(senders)
            arr = np.array([
                bool(set(m.get("participants", [])) & senders_set) for m in self.metadatas
            ])
            mask &= arr
        if date_from_iso:
            arr = np.array([
                m.get("end_ts", "") >= date_from_iso for m in self.metadatas
            ])
            mask &= arr
        if date_to_iso:
            arr = np.array([
                m.get("start_ts", "") <= date_to_iso for m in self.metadatas
            ])
            mask &= arr
        return np.where(mask)[0]

    # ---------- MMR Search ----------

    def search_mmr(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        fetch_k: int = 25,
        lambda_: float = 0.5,
        senders: Optional[Iterable[str]] = None,
        date_from_iso: Optional[str] = None,
        date_to_iso: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype("float32")
        q = l2_normalize(query_embeddings)
        if self._matrix is None or self._matrix.size == 0:
            return np.empty((q.shape[0], 0), dtype="float32"), []
        # Filter indices if requested
        idx_pool = self._indices_by_filters(senders, date_from_iso, date_to_iso)
        if idx_pool is None:
            idx_pool = np.arange(self._matrix.shape[0])
        pool = self._matrix[idx_pool]
        # Initial similarities to query
        sims_q = np.dot(q, pool.T)[0]  # shape: (pool_size,)
        # Preselect top fetch_k
        k0 = min(fetch_k, sims_q.shape[0])
        pre_idx_rel = np.argpartition(-sims_q, kth=k0 - 1)[:k0]
        selected: List[int] = []
        candidates = pre_idx_rel.tolist()
        # MMR loop
        for _ in range(min(top_k, len(candidates))):
            if not selected:
                # pick the highest similarity first
                next_local = int(max(candidates, key=lambda i: sims_q[i]))
            else:
                # Compute diversity term: max similarity to any already selected
                selected_mat = pool[selected]
                cand_mat = pool[candidates]
                div = np.dot(cand_mat, selected_mat.T)
                max_div = div.max(axis=1) if div.size > 0 else np.zeros(len(candidates))
                rel = sims_q[candidates]
                mmr_scores = lambda_ * rel - (1 - lambda_) * max_div
                next_local = candidates[int(np.argmax(mmr_scores))]
            selected.append(next_local)
            candidates.remove(next_local)
        # Map local pool indices back to global indices
        global_indices = idx_pool[np.array(selected, dtype=int)]
        # Scores are their similarity to query
        scores = np.array([np.dot(q[0], self._matrix[i]) for i in global_indices], dtype="float32")[None, :]
        # Build metas
        metas_row: List[Dict[str, Any]] = [self.metadatas[i] for i in global_indices]
        return scores, [metas_row]

    # ---------- Persistence ----------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        if self._matrix is None:
            raise ValueError("No hay matriz de embeddings para guardar")
        np.save(os.path.join(directory, "matrix.npy"), self._matrix)
        with open(os.path.join(directory, "metadatas.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False)
        with open(os.path.join(directory, "ids.json"), "w", encoding="utf-8") as f:
            json.dump(self.ids, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "InMemoryFAISS":
        matrix = np.load(os.path.join(directory, "matrix.npy"))
        with open(os.path.join(directory, "metadatas.json"), "r", encoding="utf-8") as f:
            metadatas = json.load(f)
        with open(os.path.join(directory, "ids.json"), "r", encoding="utf-8") as f:
            ids = json.load(f)
        store = cls(dim=matrix.shape[1])
        # Directly set internals
        store._matrix = matrix.astype("float32")
        if HAS_FAISS:
            store.index = faiss.IndexFlatIP(store.dim)  # type: ignore[attr-defined]
            store.index.add(store._matrix)  # type: ignore[union-attr]
        store.metadatas = metadatas
        store.ids = ids
        return store


