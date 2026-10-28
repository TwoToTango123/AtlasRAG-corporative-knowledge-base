import math

import pytest

from atlas_rag.evaluation import label_chunks, normalize, retrieval_scores
from atlas_rag.models import Chunk


def chunk(cid: str, source: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=source, source=source, text=text, ordinal=0)


def test_normalize_drops_markdown_and_line_breaks():
    assert normalize("окно **30\nкалендарных**   дней") == "окно 30 календарных дней"


def test_label_requires_same_source_and_quote():
    corpus = [
        chunk("a", "sla.md", "Sev 1: первый ответ в течение **1 часа**"),
        chunk("b", "other.md", "первый ответ в течение 1 часа"),
        chunk("c", "sla.md", "что-то другое"),
    ]
    labels = label_chunks(corpus, [{"source": "sla.md", "quote": "первый ответ в течение 1 часа"}])
    assert labels == {"a": {0}}


def test_perfect_ranking():
    s = retrieval_scores(["a", "x", "y"], {"a": {0}}, n_quotes=1, k=3)
    assert (s.recall, s.mrr, s.ndcg) == (1.0, 1.0, 1.0)


def test_relevant_at_rank_three():
    s = retrieval_scores(["x", "y", "a"], {"a": {0}}, n_quotes=1, k=5)
    assert s.recall == 1.0
    assert s.mrr == pytest.approx(1 / 3)
    assert s.ndcg == pytest.approx(1 / math.log2(4))


def test_nothing_found_in_top_k():
    s = retrieval_scores(["x", "y", "a"], {"a": {0}}, n_quotes=1, k=2)
    assert (s.recall, s.mrr, s.ndcg) == (0.0, 0.0, 0.0)


def test_multi_hop_partial_recall():
    labels = {"a": {0}, "b": {1}}
    s = retrieval_scores(["a", "x", "y"], labels, n_quotes=2, k=3)
    assert s.recall == 0.5
    assert s.mrr == 1.0
    assert s.ndcg == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_one_chunk_can_cover_two_quotes():
    s = retrieval_scores(["a"], {"a": {0, 1}}, n_quotes=2, k=5)
    assert s.recall == 1.0
    assert s.ndcg == 1.0
