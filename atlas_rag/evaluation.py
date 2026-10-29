"""Offline evaluation harness: retrieval and generation are scored separately.

**Retrieval** is scored against gold labels, no LLM involved. Each case lists
``relevant_sources`` — pairs ``{"source": file, "quote": verbatim text}`` that
hold the answer. A chunk is *relevant* if it comes from that file and contains
the quote, so labels survive re-chunking (chunk ids would not).

* **recall@k** — share of gold quotes covered by the top-k chunks (for a
  multi-hop question every part has to be found);
* **MRR** — 1 / rank of the first relevant chunk;
* **nDCG@k** — binary-relevance nDCG: relevant chunks ranked higher count more.

**Generation** is graded by an LLM judge on answerable cases (correctness,
citation quality; faithfulness comes from the generator's self-check). On cases
marked ``"answerable": false`` the only right behaviour is an honest refusal,
so the report shows the refusal rate there and the false-refusal rate on
answerable ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Settings
from .llm import LLM
from .models import AnswerResult, Chunk

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


# --- retrieval metrics ----------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, drop Markdown emphasis, collapse whitespace and line breaks."""
    return " ".join(text.replace("*", "").replace("`", "").lower().split())


def label_chunks(chunks: list[Chunk], relevant: list[dict[str, str]]) -> dict[str, set[int]]:
    """Map chunk_id -> indices of gold quotes the chunk contains (relevant chunks only)."""
    quotes = [(r["source"], normalize(r["quote"])) for r in relevant]
    labels: dict[str, set[int]] = {}
    for ch in chunks:
        text = normalize(ch.text)
        hit = {i for i, (src, q) in enumerate(quotes) if ch.source == src and q in text}
        if hit:
            labels[ch.chunk_id] = hit
    return labels


@dataclass
class RetrievalScores:
    recall: float
    mrr: float
    ndcg: float


def retrieval_scores(
    ranked_ids: list[str], labels: dict[str, set[int]], n_quotes: int, k: int
) -> RetrievalScores:
    top = ranked_ids[:k]
    covered: set[int] = set()
    first_hit = 0
    dcg = 0.0
    for rank, cid in enumerate(top, start=1):
        hit = labels.get(cid)
        if not hit:
            continue
        covered |= hit
        first_hit = first_hit or rank
        dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(labels), k) + 1))
    return RetrievalScores(
        recall=len(covered) / n_quotes if n_quotes else 0.0,
        mrr=1.0 / first_hit if first_hit else 0.0,
        ndcg=dcg / ideal if ideal else 0.0,
    )


def score_retrieval(
    ranked: list[Chunk], corpus: list[Chunk], relevant: list[dict[str, str]], k: int
) -> RetrievalScores:
    labels = label_chunks(corpus, relevant)
    return retrieval_scores([c.chunk_id for c in ranked], labels, len(relevant), k)


# --- report ---------------------------------------------------------------
@dataclass
class CaseResult:
    question: str
    answer: str
    answerable: bool
    refused: bool
    correctness: float
    citation_quality: float
    faithfulness: float
    verdict: str
    retrieval: RetrievalScores | None = None


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)
    k: int = 5

    @property
    def answerable(self) -> list[CaseResult]:
        return [r for r in self.results if r.answerable]

    @property
    def unanswerable(self) -> list[CaseResult]:
        return [r for r in self.results if not r.answerable]

    @property
    def mean_correctness(self) -> float:
        return _mean(r.correctness for r in self.answerable)

    @property
    def mean_citation_quality(self) -> float:
        return _mean(r.citation_quality for r in self.answerable)

    @property
    def mean_faithfulness(self) -> float:
        return _mean(r.faithfulness for r in self.answerable)

    @property
    def refusal_rate(self) -> float:
        """Share of unanswerable questions the system honestly declined."""
        return _mean(float(r.refused) for r in self.unanswerable)

    @property
    def false_refusal_rate(self) -> float:
        """Share of answerable questions the system wrongly declined."""
        return _mean(float(r.refused) for r in self.answerable)

    def mean_retrieval(self) -> RetrievalScores | None:
        scored = [r.retrieval for r in self.answerable if r.retrieval is not None]
        if not scored:
            return None
        return RetrievalScores(
            recall=_mean(s.recall for s in scored),
            mrr=_mean(s.mrr for s in scored),
            ndcg=_mean(s.ndcg for s in scored),
        )

    def summary(self) -> str:
        line = (
            f"кейсов={len(self.results)} (с ответом {len(self.answerable)}, "
            f"без ответа {len(self.unanswerable)})\n"
            f"корректность={self.mean_correctness:.2f}  "
            f"обоснованность={self.mean_faithfulness:.2f}  "
            f"ссылки={self.mean_citation_quality:.2f}  "
            f"честный отказ={self.refusal_rate:.0%}  "
            f"ложный отказ={self.false_refusal_rate:.0%}"
        )
        ret = self.mean_retrieval()
        if ret is not None:
            line += (
                f"\nпоиск: recall@{self.k}={ret.recall:.2f}  MRR={ret.mrr:.2f}  "
                f"nDCG@{self.k}={ret.ndcg:.2f}"
            )
        return line


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


class Evaluator:
    def __init__(self, settings: Settings, llm: LLM):
        self.settings = settings
        self.llm = llm

    def judge(self, question: str, expected: str | None, result: AnswerResult) -> dict[str, Any]:
        user = (
            f"Вопрос: {question}\n"
            f"Эталонный ответ: {expected or '(не задан)'}\n\n"
            f"Ответ ассистента: {result.answer}\n"
            f"Ссылок: {len(result.citations)}"
        )
        try:
            return self.llm.structured(
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user}],
                schema=_JUDGE_SCHEMA,
                model=self.settings.judge_model,
                max_tokens=self.settings.judge_max_tokens,
            )
        except Exception:
            return {"correctness": 0.0, "citation_quality": 0.0, "verdict": "ошибка оценщика"}

    def evaluate_case(
        self, case: dict[str, Any], result: AnswerResult, corpus: list[Chunk], k: int
    ) -> CaseResult:
        answerable = bool(case.get("answerable", True))
        if answerable:
            data = self.judge(case["question"], case.get("expected"), result)
        else:
            # nothing to grade except the refusal itself
            data = {"correctness": 0.0, "citation_quality": 0.0, "verdict": "вопрос без ответа в базе"}

        retrieval = None
        relevant = case.get("relevant_sources") or []
        if answerable and relevant and result.trace is not None:
            ranked = [sc.chunk for sc in result.trace.evidence]
            retrieval = score_retrieval(ranked, corpus, relevant, k)

        return CaseResult(
            question=case["question"],
            answer=result.answer,
            answerable=answerable,
            refused=result.refused,
            correctness=float(data["correctness"]),
            citation_quality=float(data["citation_quality"]),
            faithfulness=result.faithfulness or 0.0,
            verdict=str(data["verdict"]),
            retrieval=retrieval,
        )

    def run(
        self,
        ask_fn: Callable[[str], AnswerResult],
        cases: list[dict[str, Any]],
        corpus: list[Chunk] | None = None,
        k: int = 5,
    ) -> EvalReport:
        report = EvalReport(k=k)
        for case in cases:
            result = ask_fn(case["question"])
            report.results.append(self.evaluate_case(case, result, corpus or [], k))
        return report
