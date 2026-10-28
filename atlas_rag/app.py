"""AtlasRAG — the high-level facade.

Wires the components into one object and exposes a small surface:

    rag = AtlasRAG(Settings())
    rag.ingest_path("examples/sample_docs")   # or rag.load() an existing index
    result = rag.ask("…")                      # one-shot, agentic + grounded
    chat = rag.conversation()                  # multi-turn with memory
    report = rag.evaluate(cases)               # offline quality harness

End-to-end ``ask`` flow:
    question
      -> [conversation] condense follow-up to standalone   (chat only)
      -> [agent] plan + iterative hybrid/HyDE/CRAG retrieval
      -> [generator] citation-grounded answer (JSON-constrained)
      -> [generator] faithfulness self-check
      -> AnswerResult (answer + citations + confidence + trace)
"""

from __future__ import annotations

from typing import Any

from .agent import RetrievalAgent
from .bm25 import BM25Index
from .config import Settings
from .embeddings import build_embedder
from .evaluation import EvalReport, Evaluator
from .generator import Generator
from .ingest import Ingestor, load_documents
from .llm import build_llm
from .memory import Conversation
from .models import AnswerResult, ScoredChunk
from .query_planner import QueryPlanner
from .reranker import Reranker
from .retriever import HybridRetriever
from .vector_store import VectorStore


class AtlasRAG:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.llm = build_llm(self.settings)
        self.embedder = build_embedder(self.settings)

        self.store = VectorStore(dim=self.embedder.dim)
        self.bm25 = BM25Index()

        self.ingestor = Ingestor(self.settings, self.embedder, self.store, self.bm25)
        self.retriever = HybridRetriever(
            self.settings, self.store, self.bm25, self.embedder
        )
        self.reranker = Reranker(self.settings, self.llm)
        self.planner = QueryPlanner(self.settings, self.llm)
        self.agent = RetrievalAgent(
            self.settings, self.llm, self.retriever, self.reranker, self.planner
        )
        self.generator = Generator(self.settings, self.llm)
        self.evaluator = Evaluator(self.settings, self.llm)

        self._warn_embedder()

    def _warn_embedder(self) -> None:
        if self.embedder.name.startswith("hashing") and self.settings.verbose:
            print(
                "[AtlasRAG] Используется локальный хеширующий эмбеддер (только "
                "лексический). Для семантического качества уровня production "
                "установите sentence-transformers:\n"
                '           pip install "sentence-transformers>=3.0"\n'
            )

    # --- ingestion / persistence -----------------------------------------
    def ingest_path(self, path: str) -> int:
        docs = load_documents(path)
        n = self.ingestor.add_documents(docs)
        if self.settings.verbose:
            print(f"[AtlasRAG] Загружено документов: {len(docs)} -> фрагментов: "
                  f"{n} (эмбеддер: {self.embedder.name})")
        return n

    def save(self, directory: str | None = None) -> None:
        directory = directory or self.settings.index_dir
        self.store.save(directory, embedder=self.embedder.name)
        if self.settings.verbose:
            print(f"[AtlasRAG] Индекс сохранён в {directory}/")

    def load(self, directory: str | None = None) -> None:
        directory = directory or self.settings.index_dir
        saved = VectorStore.saved_embedder(directory)
        # same dim does not mean same vector space (bge-small and e5-small are both 384)
        if saved and saved != self.embedder.name:
            raise RuntimeError(
                f"Индекс построен эмбеддером {saved}, а сейчас выбран "
                f"{self.embedder.name}. Переиндексируйте данные (`cli.py ingest`) "
                f"или верните прежний ATLAS_EMBEDDING_MODEL."
            )
        self.store = VectorStore.load(directory)
        if self.store.dim != self.embedder.dim:
            raise RuntimeError(
                f"Размерность сохранённого индекса ({self.store.dim}) != "
                f"размерности текущего эмбеддера ({self.embedder.dim}). Индекс "
                f"построен другим бэкендом эмбеддингов; переиндексируйте данные "
                f"или смените ATLAS_EMBEDDING_BACKEND."
            )
        self.bm25.fit([(c.chunk_id, c.text) for c in self.store.chunks])
        # перепривязываем компоненты, которые держат ссылку на хранилище
        self.retriever.store = self.store
        self.ingestor.store = self.store
        if self.settings.verbose:
            print(f"[AtlasRAG] Загружено фрагментов: {len(self.store)} из {directory}/")

    @staticmethod
    def index_exists(directory: str) -> bool:
        return VectorStore.exists(directory)

    # --- question answering ----------------------------------------------
    def search(self, query: str) -> list[ScoredChunk]:
        """One non-agentic retrieval round (HyDE / rerank per settings)."""
        if self.settings.use_hyde or self.settings.use_reranker:
            self.settings.require_api_key()
        return self.agent.search_once(query)

    def ask(self, question: str) -> AnswerResult:
        if len(self.store) == 0:
            raise RuntimeError(
                "База знаний пуста. Сначала вызовите ingest_path(...) или load(...)."
            )
        self.settings.require_api_key()
        if self.settings.verbose:
            print(f"\n[AtlasRAG] Вопрос: {question}")

        trace = self.agent.gather(question)
        evidence = trace.evidence[: max(self.settings.rerank_top_k, 5)]

        result = self.generator.generate(question, evidence)
        result.trace = trace

        # Faithfulness self-check (hallucination guardrail).
        faith, unsupported = self.generator.faithfulness(result, evidence)
        result.faithfulness = faith
        if unsupported:
            result.warnings.append(f"unsupported_claims:{len(unsupported)}")
        # Down-weight confidence if the answer isn't well grounded.
        if faith < 0.5 and result.confidence > 0.4:
            result.confidence = result.confidence * faith * 2

        if self.settings.verbose:
            print(
                f"[AtlasRAG] уверенность={result.confidence:.2f} "
                f"обоснованность={result.faithfulness:.2f} "
                f"ссылок={len(result.citations)}"
            )
        return result

    # --- multi-turn -------------------------------------------------------
    def conversation(self) -> "ChatSession":
        return ChatSession(self)

    # --- evaluation -------------------------------------------------------
    def evaluate(self, cases: list[dict[str, Any]], k: int = 5) -> EvalReport:
        return self.evaluator.run(self.ask, cases, corpus=self.store.chunks, k=k)


class ChatSession:
    """A stateful, multi-turn wrapper around :meth:`AtlasRAG.ask`."""

    def __init__(self, app: AtlasRAG):
        self.app = app
        self.memory = Conversation(app.settings, app.llm)

    def ask(self, question: str) -> AnswerResult:
        standalone = self.memory.condense(question)
        if standalone != question and self.app.settings.verbose:
            print(f"[AtlasRAG] (уточняющий вопрос переписан -> {standalone!r})")
        result = self.app.ask(standalone)
        self.memory.record(question, result.answer)
        return result
