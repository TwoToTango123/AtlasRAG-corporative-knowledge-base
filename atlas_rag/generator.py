"""Citation-grounded answer generation + a faithfulness self-check.

Generation is constrained to JSON so every answer ships with explicit citations
(chunk id + verbatim quote). A second pass grades whether the answer is fully
supported by the cited evidence — the standard guardrail against hallucinated
RAG answers.
"""

from __future__ import annotations

from .config import Settings
from .llm import LLM
from .models import AnswerResult, Citation, ScoredChunk

_GEN_SYSTEM = """Ты — AtlasRAG, корпоративный ассистент по базе знаний. Отвечай \
на вопрос пользователя, опираясь ТОЛЬКО на пронумерованные фрагменты \
предоставленных данных. Правила:

- Подкрепляй каждое утверждение фрагментами. Если в данных нет ответа, прямо \
скажи об этом и поставь низкую уверенность — никогда не выдумывай факты.
- Указывай конкретные фрагменты, которые использовал. Каждая ссылка должна \
содержать chunk_id фрагмента и короткую дословную цитату (не более 25 слов), \
скопированную ровно из этого фрагмента.
- Отвечай прямо и кратко. Сначала ответ, затем детали.
- Не упоминай слова "фрагменты" или "данные" в тексте ответа; пиши естественно.
- ВСЕГДА отвечай на русском языке, даже если исходные документы на другом языке."""

_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["chunk_id", "quote"],
                "additionalProperties": False,
            },
        },
        "answer_found": {
            "type": "boolean",
            "description": "true if the evidence actually answers the question",
        },
        "confidence": {"type": "number", "description": "0-1 self-assessed"},
    },
    "required": ["answer", "citations", "answer_found", "confidence"],
    "additionalProperties": False,
}

_FAITH_SYSTEM = (
    "Ты — строгий аудитор фактической точности. На основе фрагментов-данных и "
    "ответа оцени долю фактических утверждений в ответе, которые напрямую "
    "подтверждаются данными. Верни оценку от 0 (выдумано) до 1 (полностью "
    "обосновано) и перечисли все неподтверждённые утверждения."
)

_FAITH_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["faithfulness", "unsupported_claims"],
    "additionalProperties": False,
}


def _evidence_block(evidence: list[ScoredChunk]) -> str:
    lines = []
    for sc in evidence:
        text = sc.chunk.text.strip()
        lines.append(
            f"chunk_id: {sc.chunk.chunk_id}\nsource: {sc.chunk.source}\n{text}"
        )
    return "\n\n---\n\n".join(lines)


class Generator:
    def __init__(self, settings: Settings, llm: LLM):
        self.settings = settings
        self.llm = llm

    def generate(
        self, question: str, evidence: list[ScoredChunk]
    ) -> AnswerResult:
        if not evidence:
            return AnswerResult(
                question=question,
                answer="В базе знаний не нашлось ничего, относящегося к "
                "этому вопросу.",
                confidence=0.0,
                warnings=["no_evidence"],
            )

        user = (
            f"Question: {question}\n\nEvidence passages:\n\n"
            + _evidence_block(evidence)
        )
        data = self.llm.structured(
            system=_GEN_SYSTEM,
            messages=[{"role": "user", "content": user}],
            schema=_GEN_SCHEMA,
            model=self.settings.model,
            max_tokens=4096,
            thinking=True,
        )

        valid_ids = {sc.chunk.chunk_id for sc in evidence}
        by_id = {sc.chunk.chunk_id: sc for sc in evidence}
        citations: list[Citation] = []
        warnings: list[str] = []
        for c in data.get("citations", []):
            cid = str(c.get("chunk_id", ""))
            if cid in valid_ids:
                citations.append(
                    Citation(
                        chunk_id=cid,
                        source=by_id[cid].chunk.source,
                        quote=str(c.get("quote", "")).strip(),
                    )
                )
            else:
                warnings.append(f"dropped_invalid_citation:{cid}")

        confidence = float(data.get("confidence", 0.0))
        if not data.get("answer_found", True):
            warnings.append("model_reports_answer_not_found")
            confidence = min(confidence, 0.3)

        return AnswerResult(
            question=question,
            answer=str(data.get("answer", "")).strip(),
            citations=citations,
            confidence=confidence,
            warnings=warnings,
        )

    def faithfulness(
        self, result: AnswerResult, evidence: list[ScoredChunk]
    ) -> tuple[float, list[str]]:
        if not result.answer or not evidence:
            return 0.0, []
        user = (
            "Evidence:\n\n"
            + _evidence_block(evidence)
            + f"\n\nAnswer to audit:\n{result.answer}"
        )
        try:
            data = self.llm.structured(
                system=_FAITH_SYSTEM,
                messages=[{"role": "user", "content": user}],
                schema=_FAITH_SCHEMA,
                model=self.settings.fast_model,
                max_tokens=1024,
            )
            return float(data.get("faithfulness", 0.0)), list(
                data.get("unsupported_claims", [])
            )
        except Exception:
            return 0.0, []
