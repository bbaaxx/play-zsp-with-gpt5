from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # optional, will fallback to numpy-only search if unavailable
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore
    HAS_FAISS = False

try:  # optional Qdrant support
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        PointStruct,
        Range,
        VectorParams,
    )
    HAS_QDRANT = True
except ImportError:
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None  # type: ignore
    Filter = None  # type: ignore
    FieldCondition = None  # type: ignore
    Range = None  # type: ignore
    HAS_QDRANT = False


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    def search(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        """Perform similarity search."""
        pass

    @abstractmethod
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
        """Perform MMR search with optional filtering."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of documents in the store."""
        pass

    @abstractmethod
    def save(self, directory: str) -> None:
        """Save the vector store to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, directory: str) -> "VectorStore":
        """Load the vector store from disk."""
        pass


class InMemoryFAISS(VectorStore):
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

    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Dimensión incorrecta: {embeddings.shape[1]} != {self.dim}"
            )
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
                max_div = (
                    div.max(axis=1)
                    if div.size > 0
                    else np.zeros(len(candidates))
                )
                rel = sims_q[candidates]
                mmr_scores = lambda_ * rel - (1 - lambda_) * max_div
                next_local = candidates[int(np.argmax(mmr_scores))]
            selected.append(next_local)
            candidates.remove(next_local)
        # Map local pool indices back to global indices
        global_indices = idx_pool[np.array(selected, dtype=int)]
        # Scores are their similarity to query
        scores = np.array(
            [np.dot(q[0], self._matrix[i]) for i in global_indices],
            dtype="float32",
        )[None, :]
        # Build metas
        metas_row: List[Dict[str, Any]] = [
            self.metadatas[i] for i in global_indices
        ]
        return scores, [metas_row]

    # ---------- Persistence ----------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        if self._matrix is None:
            raise ValueError("No hay matriz de embeddings para guardar")
        np.save(os.path.join(directory, "matrix.npy"), self._matrix)
        with open(
            os.path.join(directory, "metadatas.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.metadatas, f, ensure_ascii=False)
        with open(
            os.path.join(directory, "ids.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.ids, f, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "InMemoryFAISS":
        matrix = np.load(os.path.join(directory, "matrix.npy"))
        with open(
            os.path.join(directory, "metadatas.json"), "r", encoding="utf-8"
        ) as f:
            metadatas = json.load(f)
        with open(
            os.path.join(directory, "ids.json"), "r", encoding="utf-8"
        ) as f:
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


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store implementation."""
    
    def __init__(
        self,
        dim: int,
        collection_name: str = "whatsapp_rag",
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
    ):
        if not HAS_QDRANT:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install with: pip install qdrant-client"
            )
        
        self.dim = dim
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key)
        
        # Create collection if it doesn't exist
        collections = self.client.get_collections()
        collection_exists = any(
            c.name == collection_name for c in collections.collections
        )
        
        if not collection_exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        
        self._counter = self._get_max_id() + 1
    
    def _get_max_id(self) -> int:
        """Get the maximum numeric ID in the collection."""
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                order_by="id",
                with_payload=False,
                with_vectors=False
            )
            if result[0]:
                # Get the last point and extract its numeric ID
                max_point = max(
                    result[0],
                    key=lambda p: int(p.id) if str(p.id).isdigit() else 0,
                )
                return (
                    int(max_point.id) if str(max_point.id).isdigit() else 0
                )
            return 0
        except Exception:
            return 0
    
    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Dimensión incorrecta: {embeddings.shape[1]} != {self.dim}"
            )
        
        points = []
        for i, (doc_id, embedding, metadata) in enumerate(
            zip(ids, embeddings, metadatas)
        ):
            # Use sequential numeric IDs for Qdrant
            point_id = self._counter + i
            # Store original doc_id in payload
            payload = {"doc_id": doc_id, **metadata}

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name, points=points
        )
        self._counter += len(points)
    
    def search(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype("float32")
        
        results_list = []
        scores_list = []
        
        for query_vector in query_embeddings:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k
            )
            
            row_scores = []
            row_metadatas = []
            
            for point in search_result:
                row_scores.append(point.score)
                row_metadatas.append(dict(point.payload))
            
            scores_list.append(row_scores)
            results_list.append(row_metadatas)
        
        # Convert to numpy array format expected by the interface
        max_results = (
            max(len(row) for row in scores_list) if scores_list else 0
        )
        padded_scores = np.zeros(
            (len(query_embeddings), max_results), dtype=np.float32
        )
        
        for i, row_scores in enumerate(scores_list):
            padded_scores[i, :len(row_scores)] = row_scores
        
        return padded_scores, results_list
    
    def _build_filter(
        self,
        senders: Optional[Iterable[str]] = None,
        date_from_iso: Optional[str] = None,
        date_to_iso: Optional[str] = None,
    ) -> Optional[Filter]:
        """Build Qdrant filter from search criteria."""
        conditions = []
        
        if senders:
            # Filter documents that contain any of the specified senders in participants
            senders_list = list(senders)
            conditions.append(FieldCondition(
                key="participants",
                match={"any": senders_list}
            ))
        
        if date_from_iso:
            conditions.append(
                FieldCondition(key="end_ts", range=Range(gte=date_from_iso))
            )

        if date_to_iso:
            conditions.append(
                FieldCondition(key="start_ts", range=Range(lte=date_to_iso))
            )
        
        if conditions:
            return Filter(must=conditions)
        return None
    
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
        """MMR search with Qdrant. For simplicity, we do a larger initial search and apply MMR post-processing."""
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype("float32")
        
        # Build filter for Qdrant query
        filter_condition = self._build_filter(senders, date_from_iso, date_to_iso)
        
        results_list = []
        scores_list = []
        
        for query_vector in query_embeddings:
            # Fetch more candidates than needed for MMR
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                query_filter=filter_condition,
                limit=max(fetch_k, top_k * 2),
                with_vectors=True,  # Need vectors for MMR calculation
            )
            
            if not search_result:
                results_list.append([])
                scores_list.append([])
                continue
            
            # Extract vectors and metadatas for MMR
            candidates_vectors = np.array(
                [point.vector for point in search_result], dtype=np.float32
            )
            candidates_metadatas = [
                dict(point.payload) for point in search_result
            ]
            query_similarities = np.array(
                [point.score for point in search_result], dtype=np.float32
            )
            
            # Apply MMR selection
            selected_indices = self._mmr_selection(
                query_vector,
                candidates_vectors,
                query_similarities,
                top_k,
                lambda_,
            )
            
            # Build final results
            row_scores = query_similarities[selected_indices].tolist()
            row_metadatas = [
                candidates_metadatas[i] for i in selected_indices
            ]
            
            scores_list.append(row_scores)
            results_list.append(row_metadatas)
        
        # Convert to expected format
        max_results = max(len(row) for row in scores_list) if scores_list else 0
        padded_scores = np.zeros((len(query_embeddings), max_results), dtype=np.float32)
        
        for i, row_scores in enumerate(scores_list):
            padded_scores[i, :len(row_scores)] = row_scores
        
        return padded_scores, results_list
    
    def _mmr_selection(
        self, 
        query_vector: np.ndarray, 
        candidates: np.ndarray, 
        similarities: np.ndarray, 
        k: int, 
        lambda_: float
    ) -> List[int]:
        """Apply MMR selection to candidate vectors."""
        selected = []
        remaining = list(range(len(candidates)))
        
        # Normalize vectors for cosine similarity
        candidates_norm = candidates / (
            np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10
        )
        
        for _ in range(min(k, len(candidates))):
            if not selected:
                # Select the most similar document first
                best_idx = remaining[np.argmax(similarities[remaining])]
            else:
                # Calculate MMR scores for remaining candidates
                mmr_scores = []
                selected_vectors = candidates_norm[selected]
                
                for idx in remaining:
                    # Relevance score (similarity to query)
                    relevance = similarities[idx]
                    
                    # Diversity score (max similarity to selected documents)
                    diversity = np.max(
                        np.dot(candidates_norm[idx], selected_vectors.T)
                    )
                    
                    # MMR score
                    mmr_score = lambda_ * relevance - (1 - lambda_) * diversity
                    mmr_scores.append(mmr_score)
                
                # Select candidate with highest MMR score
                best_local_idx = np.argmax(mmr_scores)
                best_idx = remaining[best_local_idx]
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return selected
    
    def size(self) -> int:
        """Return the number of documents in the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0
    
    def save(self, directory: str) -> None:
        """Save collection metadata (Qdrant handles persistence automatically)."""
        os.makedirs(directory, exist_ok=True)
        
        # Save collection configuration for reconstruction
        config = {
            "collection_name": self.collection_name,
            "dim": self.dim,
            "size": self.size(),
        }
        
        with open(
            os.path.join(directory, "qdrant_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(config, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, directory: str) -> "QdrantVectorStore":
        """Load collection configuration (assumes Qdrant server running with existing data)."""
        config_path = os.path.join(directory, "qdrant_config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No se encontró configuración de Qdrant en {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Reconstruct the store (assumes collection already exists in Qdrant)
        store = cls(
            dim=config["dim"], collection_name=config["collection_name"]
        )
        
        return store


def create_vector_store(
    dim: int, backend: str = "faiss", **kwargs
) -> VectorStore:
    """Factory function to create vector store instances."""
    backend = backend.lower()

    if backend == "faiss":
        return InMemoryFAISS(dim)
    elif backend == "qdrant":
        return QdrantVectorStore(dim, **kwargs)
    else:
        raise ValueError(
            f"Backend no soportado: {backend}. Opciones: 'faiss', 'qdrant'"
        )


