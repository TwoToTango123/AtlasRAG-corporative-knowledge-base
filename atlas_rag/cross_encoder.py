"""Local pairwise reranking; raw scores are not CRAG relevance grades."""

from __future__ import annotations

from .models import ScoredChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        # Match the text visible to the existing LLM reranker.
        passages = [c.chunk.text.replace("\n", " ").strip() for c in candidates]
        passages = [p[:1200] + " …" if len(p) > 1200 else p for p in passages]
        scores = self.model.predict([(query, p) for p in passages])
        ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
        # Keep these scores separate from the 0–10 LLM grades used by the agent.
        return [candidate for candidate, _ in ranked[:k]]
