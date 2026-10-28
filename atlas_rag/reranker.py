"""LLM reranking with built-in corrective-RAG (CRAG) grading.

After hybrid retrieval over-fetches candidates, a single Claude call scores each
passage's relevance to the query on a 0-10 scale with a short rationale. This
plays the role of a cross-encoder reranker *and* the CRAG relevance grader at
once: passages that score low are demoted, and the caller can read the average
top score to decide whether retrieval was good enough or a reformulation is
needed.

One batched call grades all candidates, keeping latency and cost low. The fast
model is used by default since grading is a high-volume, low-stakes judgment.
"""

from __future__ import annotations

from .config import Settings
from .llm import LLM
from .models import ScoredChunk

_SYSTEM = (
    "Ты — точный оценщик релевантности при поиске. По запросу пользователя и "
    "набору фрагментов-кандидатов оцени, насколько каждый фрагмент полезен для "
    "ответа на запрос. Оценивай только релевантность запросу, а не качество "
    "текста. Фрагмент, прямо отвечающий на запрос, получает 9-10; косвенно "
    "связанный — 3-6; не по теме — 0-2. Будь строгим и откалиброванным."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "candidate index"},
                    "relevance": {
                        "type": "number",
                        "description": "0-10 relevance score",
                    },
                    "reason": {"type": "string", "description": "≤15 word rationale"},
                },
                "required": ["id", "relevance", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["grades"],
    "additionalProperties": False,
}


class Reranker:
    def __init__(self, settings: Settings, llm: LLM):
        self.settings = settings
        self.llm = llm
        self.failures = 0

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []

        blocks = []
        for i, sc in enumerate(candidates):
            passage = sc.chunk.text.replace("\n", " ").strip()
            if len(passage) > 1200:
                passage = passage[:1200] + " …"
            blocks.append(f"[{i}] (source: {sc.chunk.source})\n{passage}")
        user = (
            f"Запрос: {query}\n\nФрагменты-кандидаты:\n\n"
            + "\n\n".join(blocks)
            + "\n\nОцени каждого кандидата по его индексу."
        )

        try:
            data = self.llm.structured(
                system=_SYSTEM,
                messages=[{"role": "user", "content": user}],
                schema=_SCHEMA,
                model=self.settings.fast_model,
                max_tokens=2048,
            )
        except Exception:
            # Grading is best-effort; fall back to fusion order on any failure.
            self.failures += 1
            return candidates[: self.settings.rerank_top_k]

        by_id = {int(g["id"]): g for g in data.get("grades", [])}
        for i, sc in enumerate(candidates):
            g = by_id.get(i)
            if g is not None:
                sc.relevance = float(g["relevance"])
                sc.reason = str(g.get("reason", ""))
                # Blend a little of the fusion signal so ties break sensibly.
                sc.score = sc.relevance + 0.05 * sc.score
            else:
                sc.relevance = 0.0

        ranked = sorted(candidates, key=lambda s: (s.relevance or 0.0), reverse=True)
        return ranked[: self.settings.rerank_top_k]
