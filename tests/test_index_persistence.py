import pytest

from atlas_rag import AtlasRAG, Settings


def make_app(dim: int) -> AtlasRAG:
    return AtlasRAG(Settings(embedding_backend="hashing", hashing_dim=dim, verbose=False,
                             use_hyde=False, use_reranker=False))


def test_index_roundtrip_gives_same_results(tmp_path):
    app = make_app(768)
    app.ingest_path("examples/sample_docs")
    before = [h.chunk_id for h in app.search("ошибка ERR-VPN-720")]
    app.save(str(tmp_path))

    loaded = make_app(768)
    loaded.load(str(tmp_path))
    assert [h.chunk_id for h in loaded.search("ошибка ERR-VPN-720")] == before


def test_load_rejects_index_from_other_embedder(tmp_path):
    app = make_app(768)
    app.ingest_path("examples/sample_docs")
    app.save(str(tmp_path))
    with pytest.raises(RuntimeError, match="эмбеддером"):
        make_app(512).load(str(tmp_path))
