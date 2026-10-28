"""Provider-neutral LLM layer.

Every model call in AtlasRAG goes through ``LLM`` so model selection,
structured output and tool use are configured in one place. Two providers sit
behind the same interface:

* ``GroqLLM`` — Groq's OpenAI-compatible API (gpt-oss, Qwen), the default;
* ``AnthropicLLM`` — Claude via the Anthropic SDK.

The agent loop only sees :class:`ToolTurn` / :class:`ToolCall`, so it does not
care which message format the provider uses for tool calls.

The base class also counts API failures and tokens and can cache responses on
disk. The cache matters for evaluation on free-tier quotas: an interrupted run
resumes without paying for the calls it already made.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    message: dict[str, Any] = field(default_factory=dict)  # provider-native, appended to history


class LLM:
    """Base class: caching, failure and token accounting. Providers implement ``_call_*``."""

    provider = "base"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.failures = 0
        self.last_error = ""
        self.tokens = 0
        self._cache: dict[str, Any] = {}
        if settings.llm_cache_path and os.path.exists(settings.llm_cache_path):
            with open(settings.llm_cache_path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self._cache[row["key"]] = row["value"]

    def _cached(self, payload: dict[str, Any], call: Callable[[], Any]) -> Any:
        key = hashlib.sha256(
            json.dumps([self.provider, payload], ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]
        try:
            value = call()
        except Exception as e:
            self.failures += 1
            self.last_error = f"{type(e).__name__}: {e}"[:500]
            raise
        if self.settings.llm_cache_path:
            self._cache[key] = value
            os.makedirs(os.path.dirname(self.settings.llm_cache_path) or ".", exist_ok=True)
            with open(self.settings.llm_cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        return value

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
        thinking: bool = False,
    ) -> str:
        """Return the text of a single (non-tool) completion."""
        args = dict(system=system, messages=messages, model=model or self.settings.model,
                    max_tokens=max_tokens, thinking=thinking)
        return self._cached({"op": "complete", **args}, lambda: self._call_complete(**args))

    def structured(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int = 4096,
        thinking: bool = False,
    ) -> dict[str, Any]:
        """Return a parsed object matching ``schema``."""
        args = dict(system=system, messages=messages, schema=schema, model=model or self.settings.model,
                    max_tokens=max_tokens, thinking=thinking)

        def call() -> dict[str, Any]:
            text = self._call_structured(**args)
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Model did not return valid JSON: {e}\n{text[:500]}") from e

        return self._cached({"op": "structured", **args}, call)

    def tool_turn(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
        thinking: bool = False,
    ) -> ToolTurn:
        """One assistant turn that may request tools.

        ``tools`` use a neutral format: ``{"name", "description", "parameters"}``
        where ``parameters`` is a JSON schema.
        """
        args = dict(system=system, messages=messages, tools=tools, model=model or self.settings.model,
                    max_tokens=max_tokens, thinking=thinking)
        data = self._cached({"op": "tool_turn", **args}, lambda: asdict(self._call_tool_turn(**args)))
        return ToolTurn(
            text=data["text"],
            tool_calls=[ToolCall(**c) for c in data["tool_calls"]],
            message=data["message"],
        )

    def tool_results(self, results: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Messages that hand ``(tool_call_id, content)`` pairs back to the model."""
        raise NotImplementedError

    def _call_complete(self, **kw: Any) -> str:
        raise NotImplementedError

    def _call_structured(self, **kw: Any) -> str:
        raise NotImplementedError

    def _call_tool_turn(self, **kw: Any) -> ToolTurn:
        raise NotImplementedError


class GroqLLM(LLM):
    provider = "groq"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._client = None

    @property
    def client(self):
        # created on first call so indexing and pure retrieval work without a key
        if self._client is None:
            from groq import Groq

            self.settings.require_api_key()
            # free tier is ~8k tokens/min, so 429s are normal: the SDK waits retry-after
            self._client = Groq(max_retries=self.settings.llm_max_retries)
        return self._client

    def _create(self, model: str, system: str, messages: list[dict[str, Any]], max_tokens: int,
                thinking: bool, **extra: Any):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_completion_tokens": max_tokens,
            **extra,
        }
        if model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "medium" if thinking else "low"
        resp = self.client.chat.completions.create(**kwargs)
        if resp.usage:
            self.tokens += resp.usage.total_tokens
        return resp.choices[0]

    def _call_complete(self, *, system, messages, model, max_tokens, thinking) -> str:
        return (self._create(model, system, messages, max_tokens, thinking).message.content or "").strip()

    def _call_structured(self, *, system, messages, schema, model, max_tokens, thinking) -> str:
        fmt = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True}}
        choice = self._create(model, system, messages, max_tokens, thinking, response_format=fmt)
        return choice.message.content or ""

    def _call_tool_turn(self, *, system, messages, tools, model, max_tokens, thinking) -> ToolTurn:
        specs = [{"type": "function", "function": t} for t in tools]
        msg = self._create(model, system, messages, max_tokens, thinking, tools=specs).message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments or "{}"))
            for tc in (msg.tool_calls or [])
        ]
        native: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            native["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        return ToolTurn(text=(msg.content or "").strip(), tool_calls=calls, message=native)

    def tool_results(self, results: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [{"role": "tool", "tool_call_id": cid, "content": content} for cid, content in results]


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self.settings.require_api_key()
            self._client = anthropic.Anthropic(max_retries=self.settings.llm_max_retries)
        return self._client

    def _create(self, model: str, system: str, messages: list[dict[str, Any]], max_tokens: int,
                thinking: bool, **extra: Any):
        kwargs: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens, "system": system, "messages": messages, **extra,
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = self.client.messages.create(**kwargs)
        self.tokens += resp.usage.input_tokens + resp.usage.output_tokens
        return resp

    def _call_complete(self, *, system, messages, model, max_tokens, thinking) -> str:
        return _text_of(self._create(model, system, messages, max_tokens, thinking))

    def _call_structured(self, *, system, messages, schema, model, max_tokens, thinking) -> str:
        fmt = {"format": {"type": "json_schema", "schema": schema}}
        return _text_of(self._create(model, system, messages, max_tokens, thinking, output_config=fmt))

    def _call_tool_turn(self, *, system, messages, tools, model, max_tokens, thinking) -> ToolTurn:
        specs = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                 for t in tools]
        resp = self._create(model, system, messages, max_tokens, thinking, tools=specs)
        calls = [ToolCall(id=b.id, name=b.name, input=dict(b.input))
                 for b in resp.content if b.type == "tool_use"]
        native = {"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in resp.content]}
        return ToolTurn(text=_text_of(resp), tool_calls=calls, message=native)

    def tool_results(self, results: list[tuple[str, str]]) -> list[dict[str, Any]]:
        blocks = [{"type": "tool_result", "tool_use_id": cid, "content": content} for cid, content in results]
        return [{"role": "user", "content": blocks}]


def _text_of(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def build_llm(settings: Settings) -> LLM:
    providers: dict[str, type[LLM]] = {"groq": GroqLLM, "anthropic": AnthropicLLM}
    if settings.llm_provider not in providers:
        raise ValueError(f"unknown llm_provider: {settings.llm_provider!r}")
    return providers[settings.llm_provider](settings)
