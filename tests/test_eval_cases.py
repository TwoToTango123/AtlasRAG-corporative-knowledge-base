import json
from collections import Counter
from pathlib import Path

import pytest

from atlas_rag.chunking import chunk_document
from atlas_rag.config import Settings
from atlas_rag.evaluation import label_chunks
from atlas_rag.ingest import load_documents

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads((ROOT / "examples" / "eval_cases.json").read_text(encoding="utf-8"))
DOCS = load_documents(str(ROOT / "examples" / "sample_docs"))
CHUNKS = [c for d in DOCS for c in chunk_document(d, Settings())]


def test_set_size_and_mix():
    assert len(CASES) >= 30
    assert len({c["id"] for c in CASES}) == len(CASES)
    types = Counter(c["type"] for c in CASES)
    assert types["unanswerable"] >= 6
    assert {"fact", "exact_token", "multi_hop", "paraphrase", "comparison"} <= set(types)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_every_gold_quote_is_in_some_chunk(case):
    relevant = case["relevant_sources"]
    if case.get("answerable", True):
        assert relevant, "answerable case needs relevant_sources"
    else:
        assert not relevant
    labels = label_chunks(CHUNKS, relevant)
    covered = set().union(*labels.values()) if labels else set()
    missing = [r["quote"] for i, r in enumerate(relevant) if i not in covered]
    assert not missing, f"quotes not found in any chunk: {missing}"
