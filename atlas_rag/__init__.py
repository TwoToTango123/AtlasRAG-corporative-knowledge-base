"""AtlasRAG — an advanced agentic Retrieval-Augmented Generation platform.

The package is organized as a pipeline of swappable components, with an
LLM-driven *retrieval agent* (``atlas_rag.agent``) at the center that decides
its own retrieval trajectory: decomposing questions, reformulating queries,
grading evidence (corrective RAG), and judging when it has gathered enough to
answer.

Typical usage::

    from atlas_rag import AtlasRAG, Settings

    rag = AtlasRAG(Settings())
    rag.ingest_path("examples/sample_docs")
    result = rag.ask("What is our refund window for enterprise customers?")
    print(result.answer)
    for c in result.citations:
        print(" -", c.source, "::", c.quote)
"""

from .config import Settings
from .models import (
    Document,
    Chunk,
    ScoredChunk,
    Citation,
    AnswerResult,
    RetrievalTrace,
)
from .app import AtlasRAG

__all__ = [
    "AtlasRAG",
    "Settings",
    "Document",
    "Chunk",
    "ScoredChunk",
    "Citation",
    "AnswerResult",
    "RetrievalTrace",
]

__version__ = "0.1.0"
