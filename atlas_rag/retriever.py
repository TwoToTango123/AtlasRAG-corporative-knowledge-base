"""Hybrid retrieval: dense + sparse fused with Reciprocal Rank Fusion, plus
optional HyDE query expansion.

Why hybrid? Dense embeddings capture meaning ("time off" ≈ "vacation") but miss
rare exact tokens (SKUs, error codes, names). BM25 nails exact tokens but is
blind to paraphrase. RRF combines their *rankings* (not their incomparable raw
scores), giving robust recall across query types.
"""

from __future__ import annotations

import numpy as np

from .bm25 import BM25Index
from .config import Settings
from .embeddings import EmbeddingProvider
from .models import ScoredChunk
from .vector_store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        bm25: BM25Index,
        embedder: EmbeddingProvider,
    ):
        self.settings = settings
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder

    def _dense(self, query: str, hyde_text: str | None) -> list[tuple[str, float]]:
        vec: np.ndarray = self.embedder.embed_queries([query])[0]
        if hyde_text:
            # the hypothetical answer is a document, so it gets the document prefix
            vec = vec + self.embedder.embed_documents([hyde_text])[0]
        return self.store.search(vec, self.settings.dense_top_k)

    def retrieve(
        self, query: str, hyde_text: str | None = None
    ) -> list[ScoredChunk]:
        """Return fused candidates for ``query``.

        ``settings.retrieval_mode`` picks the legs: ``hybrid`` (dense + BM25
        fused with RRF), ``dense`` or ``sparse``. If ``hyde_text`` is provided,
        its embedding is added to the query embedding for the dense leg
        (Hypothetical Document Embeddings) — a hypothetical answer often lands
        closer to real answer passages than the bare question does.
        """
        mode = self.settings.retrieval_mode
        if mode not in ("hybrid", "dense", "sparse"):
            raise ValueError(f"unknown retrieval_mode: {mode!r}")

        rankings: dict[str, list[tuple[str, float]]] = {}
        if mode in ("hybrid", "dense"):
            rankings["dense"] = self._dense(query, hyde_text)
        if mode in ("hybrid", "sparse"):
            rankings["sparse"] = self.bm25.search(query, self.settings.sparse_top_k)

        fused = self._rrf(rankings)
        results: list[ScoredChunk] = []
        for chunk_id, score, per_source in fused[: self.settings.fused_top_k]:
            ch = self.store.get(chunk_id)
            if ch is None:
                continue
            results.append(ScoredChunk(chunk=ch, score=score, sources=per_source))
        return results

    def _rrf(
        self, rankings: dict[str, list[tuple[str, float]]]
    ) -> list[tuple[str, float, dict[str, float]]]:
        k = self.settings.rrf_k
        fused: dict[str, float] = {}
        per_source: dict[str, dict[str, float]] = {}
        for source, ranked in rankings.items():
            for rank, (chunk_id, _raw) in enumerate(ranked):
                contribution = 1.0 / (k + rank + 1)
                fused[chunk_id] = fused.get(chunk_id, 0.0) + contribution
                per_source.setdefault(chunk_id, {})[source] = float(rank + 1)
        return sorted(
            ((cid, sc, per_source[cid]) for cid, sc in fused.items()),
            key=lambda x: x[1],
            reverse=True,
        )
