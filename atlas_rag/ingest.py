"""Document loading and the ingestion pipeline.

Walks a path, loads text/markdown files into ``Document`` objects, chunks them,
embeds the chunks, and populates both the vector store and the BM25 index.
"""

from __future__ import annotations

import os

from .bm25 import BM25Index
from .chunking import chunk_document
from .config import Settings
from .embeddings import EmbeddingProvider
from .models import Chunk, Document
from .vector_store import VectorStore

_TEXT_EXTS = {".md", ".markdown", ".txt", ".rst", ".text"}


def load_documents(path: str) -> list[Document]:
    """Load a single file or recursively load a directory of text documents."""
    docs: list[Document] = []
    if os.path.isfile(path):
        files = [path]
    else:
        files = []
        for root, _dirs, names in os.walk(path):
            for n in names:
                if os.path.splitext(n)[1].lower() in _TEXT_EXTS:
                    files.append(os.path.join(root, n))
    for fp in sorted(files):
        try:
            with open(fp, encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        if text.strip():
            rel = os.path.relpath(fp, path) if os.path.isdir(path) else os.path.basename(fp)
            docs.append(Document(source=rel, text=text, metadata={"path": fp}))
    return docs


class Ingestor:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider,
        store: VectorStore,
        bm25: BM25Index,
    ):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25

    def add_documents(self, docs: list[Document]) -> int:
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(chunk_document(doc, self.settings))
        if not all_chunks:
            return 0

        vectors = self.embedder.embed_documents([c.text for c in all_chunks])
        self.store.add(all_chunks, vectors)

        # Rebuild BM25 over the full corpus (cheap; pure-Python).
        self.bm25.fit([(c.chunk_id, c.text) for c in self.store.chunks])
        return len(all_chunks)
