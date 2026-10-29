"""RouterAI adapter with RUB accounting and a persistent, fail-closed budget.

Use one process per ledger. Reservations survive network errors and restarts.
Token cost bounds include all published upstreams, including routing fallbacks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

import httpx
from groq.types.chat import ChatCompletion

from .llm import GroqLLM


class BudgetExceeded(RuntimeError):
    pass


class RouterAILLM(GroqLLM):
    provider = "routerai"
    base_url = "https://routerai.ru/api/v1"

    def __init__(self, settings):
        super().__init__(settings)
        self.http = httpx.Client(timeout=httpx.Timeout(180, connect=20))
        self.ledger_path = Path(settings.routerai_ledger_path)
        self.events = []
        if self.ledger_path.exists():
            self.events = [json.loads(line) for line in self.ledger_path.read_text(encoding="utf-8").splitlines()]
        self.endpoints = {}

    def balance(self) -> float:
        response = self.http.get(self.base_url + "/credits", headers=self._headers())
        response.raise_for_status()
        value = float(response.json()["data"]["credits"])
        if not math.isfinite(value):
            raise RuntimeError("Invalid RouterAI balance")
        return value

    def _headers(self):
        return {"Authorization": "Bearer " + self.settings.require_api_key()}

    @property
    def spent_rub(self):
        latest = {event["request_id"]: event for event in self.events}
        return sum(event["budget_charge_rub"] for event in latest.values())

    def _event(self, event):
        event = {"date": datetime.now(timezone.utc).isoformat(), **event}
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.events.append(event)

    def _cached(self, payload, call):
        if self.failures:
            raise RuntimeError("Previous RouterAI call failed; stopping paid calls: " + self.last_error)
        return super()._cached({**payload, "routerai_providers": self.settings.routerai_providers}, call)

    def model_endpoints(self, model):
        if model not in self.endpoints:
            response = self.http.get(self.base_url + "/models/" + model + "/endpoints")
            response.raise_for_status()
            self.endpoints[model] = response.json()["data"]["endpoints"]
        return self.endpoints[model]

    def _cost_bound(self, model, payload, max_tokens):
        endpoints = self.model_endpoints(model)
        # UTF-8 bytes give a deliberately generous token bound for the text
        # tokenizers used by these models; include space for chat serialization.
        input_bound = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 4096
        maxima = {"prompt": 0.0, "completion": 0.0}
        for endpoint in endpoints:
            for key in maxima:
                if endpoint.get("pricing_units", {}).get(key) != "token":
                    raise RuntimeError("Budget requires per-token RUB pricing")
                price = float(endpoint["pricing"][key])
                for tier in endpoint.get("variable_pricings", []):
                    price = max(price, float(tier.get(key, price)))
                if not math.isfinite(price) or price < 0:
                    raise RuntimeError("Invalid RouterAI tariff")
                maxima[key] = max(maxima[key], price)
        if not endpoints or max(maxima.values()) <= 0:
            raise RuntimeError("Missing RouterAI tariffs")
        return 1.25 * (input_bound * maxima["prompt"] + max_tokens * maxima["completion"])

    def _create(self, model, system, messages, max_tokens, thinking, _attempt=0, **extra):
        endpoint = self.settings.routerai_providers.get(model)
        if not endpoint:
            required = []
            if extra.get("tools"):
                required.append("tools")
            if extra.get("response_format"):
                required.extend(["response_format", "structured_outputs"])
            available = [e for e in self.model_endpoints(model) if e.get("status") == 0 and
                         all(p in e.get("supported_parameters", []) for p in required)]
            if not available:
                raise RuntimeError(f"No compatible RouterAI endpoints for {model}")
            endpoint = min(available, key=lambda e: float(e["pricing"]["prompt"])*5 +
                           float(e["pricing"]["completion"]))["tag"]
        payload = {
            "model": model, "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens, "provider": {"only": [endpoint], "allow_fallbacks": False},
            "reasoning": ({"effort": "medium" if thinking else "low"}
                          if model.startswith("openai/gpt-oss") else
                          ({"effort": "medium"} if thinking else {"enabled": False})), **extra,
        }
        bound = self._cost_bound(model, payload, max_tokens)
        before = self.balance()
        if self.spent_rub + bound > self.settings.routerai_budget_rub:
            raise BudgetExceeded("Persistent experiment budget would be exceeded")
        if before - bound < self.settings.routerai_min_balance_rub:
            raise BudgetExceeded("Request could consume the protected balance reserve")
        request_id = uuid.uuid4().hex
        self._event({"request_id": request_id, "state": "reserved", "model": model,
                     "budget_charge_rub": bound, "balance_before_rub": before})
        # No automatic transport retries: an ambiguous timeout keeps its reserve.
        response = self.http.post(self.base_url + "/chat/completions", headers=self._headers(), json=payload)
        if not response.is_success:
            # Documented unsuccessful HTTP responses are not billed. Verify balance.
            after = self.balance()
            self._event({"request_id": request_id, "state": "http_error", "model": model,
                         "status": response.status_code, "budget_charge_rub": max(0.0, before-after)})
            message = response.json().get("error", {})
            if isinstance(message, dict):
                message = message.get("message", "Request rejected")
            message = str(message).replace(self.settings.require_api_key(), "[REDACTED]")
            if response.status_code == 429 and _attempt < 3 and before-after < 1e-8:
                time.sleep(10 * (_attempt+1))
                return self._create(model, system, messages, max_tokens, thinking, _attempt+1, **extra)
            raise RuntimeError(f"RouterAI HTTP {response.status_code}: {message[:600]}")
        data = response.json()
        after = self.balance()
        if "error" in data:
            message = data["error"]
            if isinstance(message, dict):
                message = message.get("message", str(message))
            message = str(message).replace(self.settings.require_api_key(), "[REDACTED]")
            self._event({"request_id": request_id, "state": "api_error", "model": model,
                         "budget_charge_rub": max(0.0, before-after), "error": str(message)[:1000]})
            if ("rate-limit" in message.lower() or "rate limit" in message.lower()) and _attempt < 3 and before-after < 1e-8:
                print(f"RouterAI {model}: rate limited, retry {_attempt+1}/3 after {10*(_attempt+1)}s", flush=True)
                time.sleep(10 * (_attempt+1))
                return self._create(model, system, messages, max_tokens, thinking, _attempt+1, **extra)
            raise RuntimeError(f"RouterAI API error: {str(message)[:1000]}")
        usage = data.get("usage") or {}
        reported = usage.get("cost")
        if reported is None:
            reported = max(0.0, before-after) if after < before else bound
        cost = max(float(reported), before-after, 0.0)
        if not math.isfinite(cost):
            raise RuntimeError("Invalid RouterAI cost; reservation retained")
        self.tokens += int(usage.get("total_tokens", 0))
        self._event({"request_id": request_id, "state": "settled", "model": model,
                     "response_id": data.get("id"), "provider": data.get("provider"),
                     "budget_charge_rub": cost, "reported_cost_rub": usage.get("cost"),
                     "prompt_tokens": usage.get("prompt_tokens", 0),
                     "completion_tokens": usage.get("completion_tokens", 0),
                     "balance_after_rub": after})
        if any(choice.get("finish_reason") == "error" for choice in data.get("choices", [])):
            diagnostic = self.ledger_path.parent / "completion_errors" / (request_id + ".json")
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            diagnostic.write_text(json.dumps(data, ensure_ascii=False, indent=2).replace(
                self.settings.require_api_key(), "[REDACTED]"), encoding="utf-8")
            self._event({"request_id": request_id, "state": "completion_error", "model": model,
                         "budget_charge_rub": cost, "response_id": data.get("id"),
                         "error": "Upstream returned finish_reason=error"})
            raise RuntimeError("RouterAI upstream completion error (finish_reason=error); response saved")
        # RouterAI may return an empty optional usage breakdown, which Groq's
        # stricter SDK rejects. Usage is already accounted from the raw payload;
        # only the completion/message needs SDK parsing here.
        completion = ChatCompletion.model_validate({**data, "usage": None})
        choice = completion.choices[0]
        self._last_message = choice.message
        if choice.finish_reason == "length":
            diagnostic = self.ledger_path.parent / "truncated" / (request_id + ".json")
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            diagnostic.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(f"RouterAI output truncated ({model}, limit={max_tokens})")
        if extra.get("response_format") and not (choice.message.content or "").strip():
            raise RuntimeError("Empty structured response: " + choice.message.model_dump_json()[:1200])
        return choice

    def _call_tool_turn(self, **kwargs):
        turn = super()._call_tool_turn(**kwargs)
        # Thinking-mode tool loops may require the returned reasoning on replay.
        for key in ("reasoning_content", "reasoning_details"):
            value = getattr(self._last_message, key, None)
            if value is not None:
                turn.message[key] = value
        return turn
