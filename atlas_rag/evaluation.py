"""A small offline evaluation harness.

Runs a set of question/expectation cases through the pipeline and grades each
answer with an LLM judge on three axes that matter for business RAG:

* **correctness** — does the answer match the expected fact?
* **faithfulness** — is the answer grounded in retrieved evidence (no hallucination)?
* **citation_quality** — are the citations present and on-point?

This is the loop you tune retrieval/chunking against. ``cases`` is a list of
dicts: ``{"question": ..., "expected": ...}`` (``expected`` optional).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .llm import LLM
from .models import AnswerResult

_JUDGE_SYSTEM = (
    "Ты — беспристрастный оценщик ассистента по базе знаний. По вопросу, "
    "необязательному эталонному ответу и ответу ассистента выстави оценки. "
    "correctness: совпадает ли с эталоном / фактически верен ли ответ (0-1)? "
    "citation_quality: присутствуют ли ссылки и подкрепляют ли они ответ (0-1)? "
    "Будь строгим."
)

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "number"},
        "citation_quality": {"type": "number"},
        "verdict": {"type": "string", "description": "≤20 word justification"},
    },
    "required": ["correctness", "citation_quality", "verdict"],
    "additionalProperties": False,
}


@dataclass
class CaseResult:
    question: str
    answer: str
    correctness: float
    citation_quality: float
    faithfulness: float
    verdict: str


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def mean_correctness(self) -> float:
        return _mean(r.correctness for r in self.results)

    @property
    def mean_citation_quality(self) -> float:
        return _mean(r.citation_quality for r in self.results)

    @property
    def mean_faithfulness(self) -> float:
        return _mean(r.faithfulness for r in self.results)

    def summary(self) -> str:
        return (
            f"кейсов={len(self.results)}  "
            f"корректность={self.mean_correctness:.2f}  "
            f"обоснованность={self.mean_faithfulness:.2f}  "
            f"ссылки={self.mean_citation_quality:.2f}"
        )


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


class Evaluator:
    def __init__(self, settings: Settings, llm: LLM):
        self.settings = settings
        self.llm = llm

    def judge(self, question: str, expected: str | None, result: AnswerResult) -> CaseResult:
        user = (
            f"Вопрос: {question}\n"
            f"Эталонный ответ: {expected or '(не задан)'}\n\n"
            f"Ответ ассистента: {result.answer}\n"
            f"Ссылок: {len(result.citations)}"
        )
        try:
            data = self.llm.structured(
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user}],
                schema=_JUDGE_SCHEMA,
                model=self.settings.model,
                max_tokens=512,
            )
        except Exception:
            data = {"correctness": 0.0, "citation_quality": 0.0, "verdict": "ошибка оценщика"}
        return CaseResult(
            question=question,
            answer=result.answer,
            correctness=float(data["correctness"]),
            citation_quality=float(data["citation_quality"]),
            faithfulness=result.faithfulness or 0.0,
            verdict=str(data["verdict"]),
        )

    def run(self, ask_fn, cases: list[dict[str, Any]]) -> EvalReport:
        report = EvalReport()
        for case in cases:
            q = case["question"]
            result: AnswerResult = ask_fn(q)
            report.results.append(self.judge(q, case.get("expected"), result))
        return report
