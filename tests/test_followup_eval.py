from dataclasses import asdict
from types import SimpleNamespace

from atlas_rag.models import AnswerResult, Chunk, Document, ScoredChunk
from scripts.run_followup_eval import evidence_from, judge_payload, one_pass, review


def test_one_pass_has_no_agent_planning_and_preserves_faithfulness_handling():
    chunk=Chunk.make(Document("test.md","fact"),0,"fact")
    hit=ScoredChunk(chunk=chunk,score=1)
    calls=[]

    class Agent:
        def search_once(self,question):
            calls.append(("search",question))
            return [hit]

        def gather(self,question):
            raise AssertionError("Single-search ablation must not invoke the agent")

    class Generator:
        def generate(self,question,evidence):
            assert evidence==[hit]
            calls.append(("generate",question))
            return AnswerResult(question=question,answer="fact",confidence=0.9)

        def faithfulness(self,result,evidence):
            calls.append(("audit",result.question))
            return 0.2,["unsupported"]

    result=one_pass(SimpleNamespace(agent=Agent(),generator=Generator()),"question")
    assert [c[0] for c in calls]==["search","generate","audit"]
    assert len(result.trace.steps)==1
    assert result.confidence==0.36000000000000004
    assert result.warnings==["unsupported_claims:1"]


def test_saved_evidence_uses_best_duplicate_score_and_top_five():
    doc=Document("test.md","")
    hits=[ScoredChunk(Chunk.make(doc,i,str(i)),score=float(i)) for i in range(7)]
    duplicate=ScoredChunk(hits[0].chunk,score=10)
    entry={"result":{"trace":{"steps":[{"hits":[asdict(h) for h in hits]}, {"hits":[asdict(duplicate)]}]}}}
    evidence=evidence_from(entry)
    assert [c["text"] for c in evidence]==["0","6","5","4","3"]


def test_judge_is_blinded_to_models_and_old_ratings():
    payload=judge_payload({"question":"q","expected":"a","model":"secret"},
                          {"answer":"a","citations":[],"model":"secret","faithfulness":0.99},[])
    assert set(payload)=={"question","answerable","expected","answer","citations","evidence"}


def test_singleton_review_wrapper_preserves_all_scores():
    scores={"correctness":0.8,"faithfulness":0.5,"citation_quality":0.0,
            "unsupported_claims":["claim"],"is_refusal":False,"rationale":"reason"}
    llm=SimpleNamespace(structured=lambda **kwargs:[scores])
    assert review(llm,{})==scores
