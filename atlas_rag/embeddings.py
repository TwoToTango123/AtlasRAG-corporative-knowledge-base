"""Embedding providers.

AtlasRAG embeds text for dense (semantic) retrieval. Anthropic does not serve
an embeddings endpoint, so embeddings are a separate, pluggable concern:

* ``SentenceTransformerEmbeddings`` — recommended for real semantic quality.
* ``HashingEmbeddings`` — a deterministic, dependency-free fallback (character
  n-gram hashing into a fixed-dim L2-normalized vector). It is *lexical*, not
  truly semantic, but lets the whole system run offline. The hybrid retriever's
  BM25 component carries lexical matching either way, so the system degrades
  gracefully rather than breaking.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from .config import Settings

# Unicode-aware (\w matches Cyrillic, Latin, digits, ...) so the local fallback
# embedder produces real lexical signal for non-English corpora too.
_WORD = re.compile(r"\w+")


# Asymmetric retrieval models are trained with role prefixes (see model cards);
# without them quality drops noticeably, so the comparison would be unfair.
_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "ai-forever/ru-en-RoSBERTa": ("search_query: ", "search_document: "),
}


class EmbeddingProvider(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class HashingEmbeddings:
    """Hashing trick over word + character trigrams. No external deps."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _tokens(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        grams: list[str] = list(words)
        for w in words:
            padded = f"#{w}#"
            grams.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return grams

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._tokens(text):
                # built-in hash() is salted per process, so a saved index would
                # not match query vectors after restart; blake2b is stable
                h = int.from_bytes(
                    hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest(), "little"
                )
                idx = h % self.dim
                sign = 1.0 if (h >> 32) & 1 == 0 else -1.0
                out[i, idx] += sign
        return _l2_normalize(out)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)


class SentenceTransformerEmbeddings:
    """Wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        # renamed in sentence-transformers 5.x; the old name still exists in 3.x/4.x
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dim = int(get_dim())
        self.name = f"st:{model_name}"
        self.query_prefix, self.doc_prefix = _PREFIXES.get(model_name, ("", ""))

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vecs.astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed([self.query_prefix + t for t in texts])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.embed([self.doc_prefix + t for t in texts])


def build_embedder(settings: Settings) -> EmbeddingProvider:
    backend = settings.embedding_backend
    if backend in ("auto", "sentence-transformers"):
        try:
            emb = SentenceTransformerEmbeddings(settings.embedding_model)
            return emb
        except Exception as e:  # noqa: BLE001 - any import/load failure
            if backend == "sentence-transformers":
                raise RuntimeError(
                    f"Could not load sentence-transformers model "
                    f"'{settings.embedding_model}': {e}"
                ) from e
            # auto: fall through to hashing
    return HashingEmbeddings(settings.hashing_dim)
