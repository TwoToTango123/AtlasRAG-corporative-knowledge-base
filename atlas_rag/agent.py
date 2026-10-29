"""The retrieval agent — the heart of AtlasRAG.

Instead of a fixed "embed → top-k → stuff into prompt" pipeline, the LLM drives
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
    "parameters": {
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
    def search_once(self, query: str) -> list[ScoredChunk]:
        hyde_text = self.planner.hyde(query) if self.settings.use_hyde else None
        candidates = self.retriever.retrieve(query, hyde_text=hyde_text)
        if self.settings.use_reranker:
            return self.reranker.rerank(query, candidates)
        return candidates[: self.settings.rerank_top_k]

    def _format_results(self, hits: list[ScoredChunk], new: int) -> str:
        if not hits:
            return "Фрагменты не найдены. Попробуй другие формулировки."
        lines = []
        for sc in hits:
            rel = sc.relevance if sc.relevance is not None else sc.score
            snippet = sc.chunk.text.replace("\n", " ").strip()
            limit = self.settings.agent_preview_chars
            if limit is not None and len(snippet) > limit:
                snippet = snippet[:limit] + " …"
            lines.append(
                f"[relevance {rel:.1f}] (source: {sc.chunk.source})\n{snippet}"
            )
        avg = sum((h.relevance or 0.0) for h in hits) / len(hits)
        header = (
            f"Топ-{len(hits)} фрагментов (средняя релевантность {avg:.1f}), "
            f"новых по сравнению с прошлыми поисками: {new}.\n\n"
        )
        if new == 0:
            header = (
                "Новых фрагментов нет: всё это ты уже видел. Если данных не хватает, "
                "скорее всего, в базе этого нет — заканчивай поиск.\n\n" + header
            )
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
        seen_queries: set[str] = set()
        seen_chunks: set[str] = set()
        while step < self.settings.max_agent_steps:
            turn = self.llm.tool_turn(
                system=_SYSTEM,
                messages=messages,
                tools=[_SEARCH_TOOL],
                model=self.settings.model,
                max_tokens=4096,
                thinking=True,
            )
            messages.append(turn.message)

            if not turn.tool_calls:
                # Agent decided it's done gathering.
                trace.notes = turn.text
                trace.sufficient = True
                if turn.text:
                    log(f"  - агент: {turn.text}")
                break

            results: list[tuple[str, str]] = []
            for call in turn.tool_calls:
                query = str(call.input.get("query", "")).strip()
                rationale = str(call.input.get("rationale", "")).strip()
                step += 1
                log(f"  - поиск #{step}: {query!r}  ({rationale})")
                key = " ".join(query.lower().split())
                if key in seen_queries:
                    # a repeat would cost a HyDE + rerank call and return the same hits
                    results.append((call.id, "Этот запрос уже выполнялся, его результаты выше."))
                else:
                    seen_queries.add(key)
                    hits = self.search_once(query) if query else []
                    new = sum(1 for h in hits if h.chunk_id not in seen_chunks)
                    seen_chunks.update(h.chunk_id for h in hits)
                    trace.steps.append(
                        RetrievalStep(
                            step=step, query=query, rationale=rationale, hits=hits
                        )
                    )
                    results.append((call.id, self._format_results(hits, new)))
                if step >= self.settings.max_agent_steps:
                    break

            messages.extend(self.llm.tool_results(results))

        if not trace.steps:
            # Агент ни разу не искал (редкость) — делаем один прямой поиск как страховку.
            log("  - запасной вариант: прямой поиск")
            hits = self.search_once(question)
            trace.steps.append(
                RetrievalStep(step=1, query=question, rationale="fallback", hits=hits)
            )

        return trace
