from typing import Any

from atlas_rag import AtlasRAG, Settings
from atlas_rag.llm import LLM, ToolCall, ToolTurn


class FakeLLM(LLM):
    """Asks for one search, then stops; answers structured calls by schema."""

    provider = "fake"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.turns = 0

    def _call_tool_turn(self, *, messages, **kw: Any) -> ToolTurn:
        self.turns += 1
        if self.turns == 1:
            call = ToolCall(id="c1", name="search", input={"query": "ERR-VPN-720", "rationale": "код"})
            return ToolTurn(text="", tool_calls=[call], message={"role": "assistant", "content": ""})
        assert messages[-1] == {"role": "tool", "tool_call_id": "c1", "content": messages[-1]["content"]}
        return ToolTurn(text="нашёл", message={"role": "assistant", "content": "нашёл"})

    def tool_results(self, results: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{"role": "tool", "tool_call_id": cid, "content": c} for cid, c in results]

    def _call_structured(self, *, schema, messages, **kw: Any) -> str:
        props = schema["properties"]
        if "sub_questions" in props:
            return '{"is_complex": false, "sub_questions": ["q"]}'
        if "answer_found" in props:
            cid = messages[0]["content"].split("chunk_id: ")[1].split("\n")[0]
            return '{"answer": "Истёк сертификат", "citations": [{"chunk_id": "%s", "quote": "истёк"}], ' \
                   '"answer_found": true, "confidence": 0.9}' % cid
        return '{"faithfulness": 1.0, "unsupported_claims": []}'


def test_agent_loop_runs_search_and_cites(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    settings = Settings(embedding_backend="hashing", verbose=False, use_hyde=False, use_reranker=False)
    app = AtlasRAG(settings)
    fake = FakeLLM(settings)
    for part in (app, app.agent, app.planner, app.generator):
        part.llm = fake
    app.ingest_path("examples/sample_docs")

    result = app.ask("VPN пишет ERR-VPN-720, что делать?")

    assert [s.query for s in result.trace.steps] == ["ERR-VPN-720"]
    assert result.trace.evidence[0].chunk.source == "it_helpdesk_faq.md"
    assert result.citations and result.citations[0].source == "it_helpdesk_faq.md"
    assert not result.refused
    assert fake.failures == 0
