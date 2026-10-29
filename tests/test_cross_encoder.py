from atlas_rag.cross_encoder import CrossEncoderReranker
from atlas_rag.models import Chunk, Document, ScoredChunk


def test_pairwise_ranking_uses_same_passages_without_changing_crag_scores():
    class Model:
        def predict(self, pairs):
            assert pairs == [("question", "a" * 1200 + " …"), ("question", "short text")]
            return [-2.0, 4.0]

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = Model()
    doc = Document(source="test.md", text="")
    first = ScoredChunk(Chunk.make(doc, 0, "a" * 1300), score=0.2)
    second = ScoredChunk(Chunk.make(doc, 1, "short\ntext"), score=0.1)
    candidates = [first, second]
    assert reranker.rerank("question", candidates, 1) == [second]
    assert candidates == [first, second]
    assert first.score == 0.2 and second.score == 0.1
    assert first.relevance is None and second.relevance is None
    assert reranker.rerank("question", [], 5) == []
