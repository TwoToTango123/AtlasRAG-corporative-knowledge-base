"""A small, self-contained Okapi BM25 index for sparse (lexical) retrieval.

No external dependency. Operates over the same chunk set as the vector store so
the hybrid retriever can fuse the two rankings.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Unicode-aware: \w matches Latin, Cyrillic, digits, etc. (str patterns are
# Unicode by default in Python 3), so the index works for multilingual corpora.
_WORD = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.docs_tokens: list[list[str]] = []
        self.doc_freqs: list[Counter] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0

    def fit(self, items: list[tuple[str, str]]) -> None:
        """``items`` is a list of (chunk_id, text)."""
        self.chunk_ids = [cid for cid, _ in items]
        self.docs_tokens = [_tokenize(t) for _, t in items]
        self.doc_freqs = [Counter(toks) for toks in self.docs_tokens]
        self.doc_len = [len(toks) for toks in self.docs_tokens]
        n = len(self.docs_tokens)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0

        self.df = Counter()
        for freqs in self.doc_freqs:
            for term in freqs:
                self.df[term] += 1
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self.chunk_ids:
            return []
        q_terms = _tokenize(query)
        scores: list[float] = [0.0] * len(self.chunk_ids)
        for i, freqs in enumerate(self.doc_freqs):
            dl = self.doc_len[i] or 1
            s = 0.0
            for term in q_terms:
                if term not in freqs:
                    continue
                idf = self.idf.get(term, 0.0)
                tf = freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (tf * (self.k1 + 1)) / denom
            scores[i] = s
        ranked = sorted(
            zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True
        )
        return [(cid, sc) for cid, sc in ranked[:top_k] if sc > 0]
