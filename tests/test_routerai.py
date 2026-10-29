import json

import httpx
import pytest

from atlas_rag.config import Settings
from atlas_rag.routerai import BudgetExceeded, RouterAILLM


def adapter(tmp_path, monkeypatch, credits=100.0):
    monkeypatch.setenv("ROUTERAI_API_KEY", "test")
    settings = Settings(llm_provider="routerai", routerai_ledger_path=str(tmp_path / "usage.jsonl"),
                        routerai_providers={"test/model": "test"})
    llm = RouterAILLM(settings)
    posts = []

    def transport(request):
        if request.url.path.endswith("/credits"):
            return httpx.Response(200, json={"data": {"credits": credits - len(posts)*0.01}})
        if request.url.path.endswith("/endpoints"):
            return httpx.Response(200, json={"data": {"endpoints": [{
                "tag": "test", "status": 0, "supported_parameters": [],
                "pricing": {"prompt": 0.00001, "completion": 0.00002},
                "pricing_units": {"prompt": "token", "completion": "token"}}]}})
        posts.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
            "model": "test/model", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cost": 0.01}})

    llm.http = httpx.Client(transport=httpx.MockTransport(transport))
    return llm, posts


def invoke(llm):
    return llm.complete(system="test", messages=[{"role": "user", "content": "test"}],
                        model="test/model", max_tokens=20)


def test_protected_balance_prevents_paid_call(tmp_path, monkeypatch):
    llm, posts = adapter(tmp_path, monkeypatch, credits=10.01)
    with pytest.raises(BudgetExceeded):
        invoke(llm)
    assert not posts


def test_success_settles_reserve_and_records_usage(tmp_path, monkeypatch):
    llm, posts = adapter(tmp_path, monkeypatch)
    assert invoke(llm) == "ok"
    assert llm.spent_rub == pytest.approx(0.01)
    assert llm.tokens == 7
    assert posts[0]["provider"] == {"only": ["test"], "allow_fallbacks": False}
    assert posts[0]["reasoning"] == {"enabled": False}
    assert [e["state"] for e in llm.events] == ["reserved", "settled"]


def test_interrupted_request_reserve_survives_restart(tmp_path, monkeypatch):
    llm, posts = adapter(tmp_path, monkeypatch)
    llm._event({"request_id": "interrupted", "state": "reserved", "budget_charge_rub": 90.0})
    restarted, posts = adapter(tmp_path, monkeypatch)
    assert restarted.spent_rub == 90.0
    with pytest.raises(BudgetExceeded):
        invoke(restarted)
    assert not posts


def test_http_200_api_error_is_not_counted_as_a_completion(tmp_path, monkeypatch):
    llm, posts = adapter(tmp_path, monkeypatch)
    previous = llm.http

    def transport(request):
        if request.method == "POST":
            return httpx.Response(200, json={"error": {"message": "Upstream failed"}})
        return previous.send(request)

    llm.http = httpx.Client(transport=httpx.MockTransport(transport))
    with pytest.raises(RuntimeError, match="Upstream failed"):
        invoke(llm)
    assert llm.tokens == 0
    assert llm.spent_rub == 0
    assert llm.events[-1]["state"] == "api_error"


def test_default_routing_selects_a_compatible_host(tmp_path, monkeypatch):
    llm, posts = adapter(tmp_path, monkeypatch)
    llm.settings.routerai_providers = {}
    assert invoke(llm) == "ok"
    assert posts[0]["provider"]["only"] == ["test"]


def test_empty_router_usage_breakdown_does_not_discard_a_valid_answer(tmp_path, monkeypatch):
    llm, _ = adapter(tmp_path, monkeypatch)
    previous = llm.http

    def transport(request):
        response = previous.send(request)
        if request.method == "POST":
            data = response.json()
            data["usage"]["completion_tokens_details"] = {}
            return httpx.Response(200, json=data)
        return response

    llm.http = httpx.Client(transport=httpx.MockTransport(transport))
    assert invoke(llm) == "ok"
    assert llm.tokens == 7
    assert llm.spent_rub == pytest.approx(0.01)


def test_unbilled_rate_limit_retries_are_bounded(tmp_path, monkeypatch):
    llm, _ = adapter(tmp_path, monkeypatch)
    monkeypatch.setattr("atlas_rag.routerai.time.sleep", lambda seconds: None)
    previous = llm.http
    attempts = []

    def transport(request):
        if request.method == "POST":
            attempts.append(1)
            return httpx.Response(200, json={"error": {"message": "Upstream rate-limited"}})
        return previous.send(request)

    llm.http = httpx.Client(transport=httpx.MockTransport(transport))
    with pytest.raises(RuntimeError, match="rate-limited"):
        invoke(llm)
    assert len(attempts) == 4
    assert llm.spent_rub == 0


def test_finish_error_is_saved_and_never_accepted_as_a_valid_answer(tmp_path, monkeypatch):
    llm, _ = adapter(tmp_path, monkeypatch)
    previous = llm.http

    def transport(request):
        response = previous.send(request)
        if request.method == "POST":
            data = response.json()
            data["choices"][0]["finish_reason"] = "error"
            return httpx.Response(200, json=data)
        return response

    llm.http = httpx.Client(transport=httpx.MockTransport(transport))
    with pytest.raises(RuntimeError, match="upstream completion error"):
        invoke(llm)
    assert not llm._cache
    assert llm.spent_rub == pytest.approx(0.01)
    assert llm.events[-1]["state"] == "completion_error"
    assert len(list((tmp_path/"completion_errors").glob("*.json"))) == 1
