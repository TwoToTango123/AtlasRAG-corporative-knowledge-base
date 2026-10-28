"""In-memory dense vector store with cosine similarity and disk persistence.

Embeddings are L2-normalized on the way in, so cosine similarity reduces to a
single matrix-vector dot product — fast enough for tens of thousands of chunks
without an external vector database, and trivially swappable for FAISS/pgvector
later (the interface is just ``add`` / ``search``).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .models import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._id_to_pos: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors and chunks length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim {vectors.shape[1]} != store dim {self.dim}"
            )
        start = len(self.chunks)
        for i, ch in enumerate(chunks):
            self._id_to_pos[ch.chunk_id] = start + i
        self.chunks.extend(chunks)
        self._matrix = (
            vectors.astype(np.float32)
            if self._matrix.size == 0
            else np.vstack([self._matrix, vectors.astype(np.float32)])
        )

    def get(self, chunk_id: str) -> Chunk | None:
        pos = self._id_to_pos.get(chunk_id)
        return self.chunks[pos] if pos is not None else None

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if len(self.chunks) == 0:
            return []
        q = query_vec.reshape(-1).astype(np.float32)
        n = np.linalg.norm(q)
        if n:
            q = q / n
        sims = self._matrix @ q
        k = min(top_k, len(self.chunks))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.chunks[i].chunk_id, float(sims[i])) for i in idx]

    # --- persistence ------------------------------------------------------
    def save(self, directory: str, embedder: str = "") -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "vectors.npy"), self._matrix)
        with open(os.path.join(directory, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for ch in self.chunks:
                f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "embedder": embedder}, f)

    @staticmethod
    def saved_embedder(directory: str) -> str:
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as f:
            return str(json.load(f).get("embedder", ""))

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        store = cls(dim=int(meta["dim"]))
        matrix = np.load(os.path.join(directory, "vectors.npy"))
        chunks: list[Chunk] = []
        with open(os.path.join(directory, "chunks.jsonl"), encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk.from_dict(json.loads(line)))
        store.chunks = chunks
        store._matrix = matrix.astype(np.float32)
        store._id_to_pos = {c.chunk_id: i for i, c in enumerate(chunks)}
        return store

    @staticmethod
    def exists(directory: str) -> bool:
        return os.path.exists(os.path.join(directory, "meta.json"))
