"""Core data structures passed between AtlasRAG components."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1("␟".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


@dataclass
class Document:
    """A source document before chunking."""

    source: str                       # path / URL / logical name
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return _stable_id(self.source)


@dataclass
class Chunk:
    """An indexed unit of retrieval."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    ordinal: int                      # position of the chunk within its doc
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make(doc: Document, ordinal: int, text: str) -> "Chunk":
        cid = _stable_id(doc.source, str(ordinal), text[:64])
        return Chunk(
            chunk_id=cid,
            doc_id=doc.doc_id,
            source=doc.source,
            text=text,
            ordinal=ordinal,
            metadata=dict(doc.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Chunk":
        return Chunk(**d)


@dataclass
class ScoredChunk:
    """A chunk with retrieval/relevance scores attached."""

    chunk: Chunk
    score: float = 0.0                # fused / rerank score (higher = better)
    relevance: float | None = None    # 0-10 LLM relevance grade (CRAG)
    reason: str = ""                  # short rationale from grader/reranker
    sources: dict[str, float] = field(default_factory=dict)  # per-retriever ranks

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class Citation:
    chunk_id: str
    source: str
    quote: str


@dataclass
class RetrievalStep:
    step: int
    query: str
    rationale: str
    hits: list[ScoredChunk]


@dataclass
class RetrievalTrace:
    """A full record of how the agent gathered evidence — for transparency."""

    original_question: str
    steps: list[RetrievalStep] = field(default_factory=list)
    sufficient: bool = False
    notes: str = ""

    @property
    def evidence(self) -> list[ScoredChunk]:
        """Deduplicated, best-scored evidence across all steps."""
        best: dict[str, ScoredChunk] = {}
        for st in self.steps:
            for h in st.hits:
                cur = best.get(h.chunk_id)
                if cur is None or h.score > cur.score:
                    best[h.chunk_id] = h
        return sorted(best.values(), key=lambda s: s.score, reverse=True)


@dataclass
class AnswerResult:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0           # 0-1 self-reported + adjusted
    faithfulness: float | None = None  # 0-1 self-check score
    trace: RetrievalTrace | None = None
    warnings: list[str] = field(default_factory=list)
