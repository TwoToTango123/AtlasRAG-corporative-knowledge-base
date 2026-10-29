from dataclasses import asdict
import json
from types import SimpleNamespace

from atlas_rag.context import select_context
from atlas_rag.models import Chunk, Document, ScoredChunk, RetrievalStep, RetrievalTrace
from atlas_rag.semantic_support_v2 import extract_claims, verify_claims
from scripts.run_followup_eval import evidence_from


def hit(source, text, score):
    return ScoredChunk(Chunk.make(Document(source, text), 0, text), score=score)


def test_low_graded_subject_and_late_rule_survive_without_gold_labels():
    bridge = hit("a", "Проект Берилл использует схему Z-83.", 1)
    distractors = [hit(str(i), f"Лимит компенсации схемы Z-{i} равен 40.", 10) for i in range(7)]
    rule = hit("rule", "Для Z-83 лимит компенсации равен 600.", 9)
    trace = RetrievalTrace("Какой лимит компенсации проекта Берилл?", steps=[
        RetrievalStep(1, "проект Берилл", "", distractors[:4]+[bridge]),
        RetrievalStep(2, "Z-83 лимит", "", [rule]+distractors[4:])])
    assert bridge not in select_context(trace, 5, "legacy")
    selected = select_context(trace, 5)
    assert bridge in selected and rule in selected and len(selected) == 5
    for step in trace.steps:
        for h in step.hits:
            h.score *= 0.01 if step.step == 1 else 100
    assert bridge in select_context(trace, 5) and rule in select_context(trace, 5)


def test_context_is_bounded_deduplicated_and_explicit_in_saved_results():
    a, b = hit("a", "entity fact", 1), hit("b", "later fact", 10)
    trace = RetrievalTrace("entity", steps=[RetrievalStep(1, "q1", "", [a,b]),
        RetrievalStep(2, "q2", "", [b,a])])
    selected = select_context(trace, 2)
    assert len(selected) == 2 and len({h.chunk_id for h in selected}) == 2
    trace.selected_context_ids = [h.chunk_id for h in selected]
    assert trace.evidence == selected
    assert [c["chunk_id"] for c in evidence_from({"result": {"trace": asdict(trace)}})] == trace.selected_context_ids
    assert select_context(trace, 0) == []


def test_one_search_order_is_preserved():
    hits = [hit("a", "match", 1), hit("b", "match match", 0.9)]
    trace = RetrievalTrace("match", steps=[RetrievalStep(1,"match","",hits)])
    assert select_context(trace, 5) == hits


def test_score_order_preserves_anchor_membership():
    a,b=hit('a','Берилл project',1),hit('b','numeric rule',10)
    trace=RetrievalTrace('Берилл',steps=[RetrievalStep(1,'q1','',[a,b]),RetrievalStep(2,'q2','',[b,a])])
    first=select_context(trace,5,'anchored')
    scored=select_context(trace,5,'anchored_score')
    assert {h.chunk_id for h in first}=={h.chunk_id for h in scored}
    assert [h.score for h in scored]==sorted([h.score for h in first],reverse=True)


def test_v2_extractor_receives_question_and_does_not_synthesize_claims():
    seen = []
    def structured(**kwargs):
        seen.append(json.loads(kwargs["messages"][0]["content"]))
        return {"claims": ["Для нового поставщика нужна проверка."]}
    claims = extract_claims(SimpleNamespace(structured=structured), "judge", "Нужна проверка.", "Поставщик новый.")
    assert seen == [{"question": "Поставщик новый.", "answer": "Нужна проверка."}]
    assert claims == ["Для нового поставщика нужна проверка."]


def test_v2_scenario_does_not_bypass_source_span_requirement():
    seen = []
    def structured(**kwargs):
        seen.append(json.loads(kwargs["messages"][0]["content"]))
        return {"results": [{"claim_id": "c0", "label": "supported", "unit_ids": [], "rationale": "guess"}]}
    result = verify_claims(SimpleNamespace(structured=structured), "judge", ["claim"],
                           [{"chunk_id": "x", "text": "source"}], "question")
    assert seen[0]["question"] == "question"
    assert result[0]["label"] == "insufficient" and result[0]["raw_label"] == "supported"
