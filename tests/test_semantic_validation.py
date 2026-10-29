from types import SimpleNamespace
import pytest

from atlas_rag.agent import RetrievalAgent
from atlas_rag.config import Settings
from atlas_rag.models import Chunk, Document, ScoredChunk
from atlas_rag.semantic_support import source_units, verify_claims, classification_metrics
from scripts.run_routerai_eval import ROOT
from scripts.validation_fixtures import semantic_cases, sequential_cases
from scripts.run_validation_eval import amount_present, trajectory, paired_interval, entity_literal_present


def test_complete_observation_exposes_fact_after_old_cutoff():
    chunk = Chunk.make(Document("test.md", ""), 0, "x"*510+" Секретный лимит 12345 рублей.")
    agent = RetrievalAgent(Settings(), None, None, None, None)
    hit = ScoredChunk(chunk, relevance=9)
    assert "12345" not in agent._format_results([hit], 1)
    agent.settings.agent_preview_chars = None
    assert "12345" in agent._format_results([hit], 1)


def test_exact_spans_and_nonexistent_evidence_cannot_support_claim():
    chunks = [{"chunk_id": "chunk", "text": "Заголовок\n\nЛимит 30 дней.\n"}]
    for unit in source_units(chunks):
        assert chunks[0]["text"][unit["start"]:unit["end"]] == unit["text"]
    llm = SimpleNamespace(structured=lambda **kw: {"results": [{"claim_id": "c0",
        "label": "supported", "unit_ids": ["fake"], "rationale": "invented"}]})
    result = verify_claims(llm, "model", ["30 дней"], chunks)[0]
    assert result["label"] == "insufficient"
    assert result["raw_label"] == "supported"
    assert result["abstained_invalid_evidence"]


def test_misassigned_claim_verdict_is_rejected():
    llm = SimpleNamespace(structured=lambda **kw: {"results": [{"claim_id": "c1",
        "label": "supported", "unit_ids": ["s0l0"], "rationale": "wrong claim"}]})
    with pytest.raises(ValueError, match="reordered"):
        verify_claims(llm, "model", ["claim"], [{"chunk_id": "c", "text": "text"}])


def test_false_support_metrics_do_not_reward_reject_all():
    metrics = classification_metrics([{"gold": "supported", "pred": "insufficient"},
        {"gold": "contradicted", "pred": "supported"},
        {"gold": "insufficient", "pred": "insufficient"}])
    assert metrics["false_supported"] == 1
    assert metrics["false_supported_rate"] == 0.5
    assert metrics["support_recall"] == 0
    assert metrics["support_precision"] == 0


def test_controls_are_source_exact_and_group_disjoint():
    cases = semantic_cases(ROOT)
    assert len(cases) == 40
    sources = {}
    for split in ["calibration", "heldout"]:
        part = [c for c in cases if c["split"] == split]
        assert len(part) == 20
        assert {label: sum(c["label"] == label for c in part) for label in
            ["supported", "contradicted", "insufficient"]} == {"supported": 8, "contradicted": 8, "insufficient": 4}
        sources[split] = {c["source"] for c in part}
        for c in part:
            assert c["premise"] in (ROOT/"examples/sample_docs"/c["source"]).read_text(encoding="utf-8")
    assert not sources["calibration"] & sources["heldout"]


def test_bridge_fixture_has_exact_gold_and_distinct_archive_limits():
    docs, cases = sequential_cases()
    by_source = {d.source: d.text for d in docs}
    assert len(cases) == 12 and len(docs) == 84
    for case in cases:
        for gold in case["relevant_sources"]:
            assert gold["quote"] in by_source[gold["source"]]
        assert case["expected_code"] not in case["question"]
        assert amount_present(case["expected"], case["expected_amount"])
        assert not amount_present(str(case["expected_amount"]*10), case["expected_amount"])


def test_trajectory_counts_marginal_gold_not_new_distractors():
    gold = Chunk.make(Document("gold.md", ""), 0, "gold fact")
    distractor = Chunk.make(Document("other.md", ""), 0, "other")
    hit = lambda chunk: {"chunk": {"chunk_id": chunk.chunk_id}}
    entry = {"result": {"trace": {"steps": [{"query": "first", "hits": [hit(gold)]},
        {"query": "repeat", "hits": [hit(gold), hit(distractor)]}]}}}
    audit = trajectory(entry, [gold, distractor], {"relevant_sources": [{"source": "gold.md", "quote": "gold fact"}]})
    assert audit["rounds"][1]["new_chunks"] == 1
    assert audit["rounds"][1]["new_gold_chunks"] == 0


def test_paired_comparison_uses_matching_case_ids_only():
    result = paired_interval({"a": 0.1, "b": 1, "unmatched": 100}, {"b": 0.9, "a": 0.2})
    assert result["n"] == 2
    assert abs(result["mean_change"]) < 1e-12


def test_entity_probe_allows_typographic_hyphen_but_not_another_model():
    assert entity_literal_present("NW-200", "Гарантия NW‑200?")
    assert not entity_literal_present("NW-200", "Гарантия NW‑100?")
