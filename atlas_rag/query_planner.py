"""Query understanding: HyDE generation and complex-question decomposition.

These run before/around retrieval to widen recall on hard questions:

* **HyDE** (Hypothetical Document Embeddings): draft a short hypothetical answer
  and embed it alongside the question, pulling the dense search toward passages
  that look like real answers.
* **Decomposition**: break a multi-part question into standalone sub-questions
  so each can be retrieved for independently. The agent uses this to plan its
  first round of searches.
"""

from __future__ import annotations

from .config import Settings
from .llm import LLM

_HYDE_SYSTEM = (
    "Ты пишешь краткий правдоподобный гипотетический ответ на вопрос "
    "пользователя так, будто цитируешь внутренний документ компании. 2-4 "
    "предложения. Конкретику можно придумывать свободно — этот текст нужен "
    "только для наведения семантического поиска и никогда не показывается "
    "пользователю. Используй лексику реального регламента/спецификации. "
    "Пиши на русском языке."
)

_DECOMP_SYSTEM = (
    "Ты раскладываешь вопрос пользователя на минимальный набор самостоятельных, "
    "автономных под-вопросов, нужных для ответа по базе знаний. Если вопрос уже "
    "атомарный — верни его без изменений одним элементом. Не более 4. "
    "Под-вопросы пиши на русском языке."
)

_DECOMP_SCHEMA = {
    "type": "object",
    "properties": {
        "is_complex": {"type": "boolean"},
        "sub_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-4 standalone sub-questions",
        },
    },
    "required": ["is_complex", "sub_questions"],
    "additionalProperties": False,
}


class QueryPlanner:
    def __init__(self, settings: Settings, llm: LLM):
        self.settings = settings
        self.llm = llm
        self.hyde_failures = 0

    def hyde(self, question: str) -> str:
        try:
            return self.llm.complete(
                system=_HYDE_SYSTEM,
                messages=[{"role": "user", "content": question}],
                model=self.settings.fast_model,
                max_tokens=300,
            )
        except Exception:
            self.hyde_failures += 1
            return ""

    def decompose(self, question: str) -> list[str]:
        try:
            data = self.llm.structured(
                system=_DECOMP_SYSTEM,
                messages=[{"role": "user", "content": question}],
                schema=_DECOMP_SCHEMA,
                model=self.settings.fast_model,
                max_tokens=600,
            )
            subs = [s.strip() for s in data.get("sub_questions", []) if s.strip()]
            return subs or [question]
        except Exception:
            return [question]
