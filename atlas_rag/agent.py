"""The retrieval agent — the heart of AtlasRAG.

Instead of a fixed "embed → top-k → stuff into prompt" pipeline, Claude drives
its own retrieval trajectory through a tool-use loop:

1. It decomposes the question and plans an initial set of searches.
2. Each ``search`` call runs the full hybrid + HyDE + rerank/grade pipeline and
   returns passages *with their relevance grades*.
3. Reading those grades, the agent applies corrective-RAG logic: if the best
   passages are weak, it reformulates (synonyms, narrower/broader terms, a
   different sub-topic) and searches again.
4. When it judges the accumulated evidence sufficient — or hits the step budget
   — it stops. Generation then happens separately, grounded in the gathered
   evidence (see ``generator.py``).

This makes retrieval adaptive: multi-hop questions get multiple targeted
searches, and dead-end queries get corrected rather than silently returning
junk.
"""

from __future__ import annotations

from .config import Settings
from .llm import LLM
from .models import RetrievalStep, RetrievalTrace, ScoredChunk
from .query_planner import QueryPlanner
from .reranker import Reranker
from .retriever import HybridRetriever

_SYSTEM = """Ты — планировщик поиска для корпоративного ассистента по базе знаний.

Твоя ЕДИНСТВЕННАЯ задача — собрать достаточно качественных данных, чтобы \
ответить на вопрос пользователя. Ты НЕ пишешь итоговый ответ — это делает \
отдельный шаг на основе собранных тобой данных.

Как работать:
- Сначала реши, состоит ли вопрос из нескольких частей. Если да — ищи по \
каждой части отдельно, а не одним расплывчатым запросом.
- Вызывай инструмент `search` с конкретным запросом и однострочным обоснованием.
- Каждый результат содержит фрагменты с оценкой РЕЛЕВАНТНОСТИ от 0 до 10. \
Прочитай их.
- Корректирующий поиск: если лучшие фрагменты набирают меньше 6 баллов или не \
покрывают часть вопроса — переформулируй запрос (синонимы, более точные термины \
или другая подтема) и ищи снова. Не повторяй запрос, который уже сработал.
- Останавливайся, как только собрано достаточно надёжных данных \
(релевантность >= 6), покрывающих все части вопроса, ИЛИ когда новые поиски \
перестают находить что-то новое.
- Когда останавливаешься, ответь одним коротким предложением: что нашёл и \
полностью ли данные покрывают вопрос. Сам вопрос НЕ отвечай.

Будь экономным: большинству вопросов хватает 1-3 поисков. Расходуй бюджет \
шагов разумно."""

_SEARCH_TOOL = {
    "name": "search",
    "description": (
        "Поиск по базе знаний конкретным запросом на естественном языке. "
        "Возвращает наиболее релевантные фрагменты, каждый с оценкой "
        "релевантности от 0 до 10. Используй несколько вызовов, чтобы покрыть "
        "разные части вопроса или переформулировать запрос при слабых результатах."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Конкретный поисковый запрос (вопрос или ключевые слова).",
            },
            "rationale": {
                "type": "string",
                "description": "Зачем нужен этот поиск (одна строка).",
            },
        },
        "required": ["query", "rationale"],
    },
}


class RetrievalAgent:
    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        retriever: HybridRetriever,
        reranker: Reranker,
        planner: QueryPlanner,
    ):
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.planner = planner

    # --- one search round (the tool implementation) -----------------------
    def _do_search(self, query: str) -> list[ScoredChunk]:
        hyde_text = self.planner.hyde(query) if self.settings.use_hyde else None
        candidates = self.retriever.retrieve(query, hyde_text=hyde_text)
        if self.settings.use_reranker:
            return self.reranker.rerank(query, candidates)
        return candidates[: self.settings.rerank_top_k]

    def _format_results(self, hits: list[ScoredChunk]) -> str:
        if not hits:
            return "Фрагменты не найдены. Попробуй другие формулировки."
        lines = []
        for sc in hits:
            rel = sc.relevance if sc.relevance is not None else sc.score
            snippet = sc.chunk.text.replace("\n", " ").strip()
            if len(snippet) > 500:
                snippet = snippet[:500] + " …"
            lines.append(
                f"[relevance {rel:.1f}] (source: {sc.chunk.source})\n{snippet}"
            )
        avg = sum((h.relevance or 0.0) for h in hits) / len(hits)
        header = f"Топ-{len(hits)} фрагментов (средняя релевантность {avg:.1f}):\n\n"
        return header + "\n\n".join(lines)

    # --- the agentic loop -------------------------------------------------
    def gather(self, question: str) -> RetrievalTrace:
        trace = RetrievalTrace(original_question=question)
        log = print if self.settings.verbose else (lambda *a, **k: None)

        # Seed the model with a decomposition hint (cheap, improves planning).
        subs = self.planner.decompose(question)
        hint = ""
        if len(subs) > 1:
            hint = "\n\nThis question appears to have multiple parts:\n" + "\n".join(
                f"  - {s}" for s in subs
            )
            log(f"  - вопрос разбит на под-вопросы: {len(subs)}")

        messages: list[dict] = [
            {"role": "user", "content": f"Question: {question}{hint}"}
        ]

        step = 0
        while step < self.settings.max_agent_steps:
            resp = self.llm.raw(
                system=_SYSTEM,
                messages=messages,
                tools=[_SEARCH_TOOL],
                model=self.settings.model,
                max_tokens=4096,
                thinking=True,
            )
            messages.append({"role": "assistant", "content": resp.content})

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                # Agent decided it's done gathering.
                final = "".join(
                    b.text for b in resp.content if b.type == "text"
                ).strip()
                trace.notes = final
                trace.sufficient = True
                if final:
                    log(f"  - агент: {final}")
                break

            tool_results = []
            for tu in tool_uses:
                query = str(tu.input.get("query", "")).strip()
                rationale = str(tu.input.get("rationale", "")).strip()
                step += 1
                log(f"  - поиск #{step}: {query!r}  ({rationale})")
                hits = self._do_search(query) if query else []
                trace.steps.append(
                    RetrievalStep(
                        step=step, query=query, rationale=rationale, hits=hits
                    )
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": self._format_results(hits),
                    }
                )
                if step >= self.settings.max_agent_steps:
                    break

            messages.append({"role": "user", "content": tool_results})

        if not trace.steps:
            # Агент ни разу не искал (редкость) — делаем один прямой поиск как страховку.
            log("  - запасной вариант: прямой поиск")
            hits = self._do_search(question)
            trace.steps.append(
                RetrievalStep(step=1, query=question, rationale="fallback", hits=hits)
            )

        return trace
