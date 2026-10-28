"""Thin wrapper around the Anthropic SDK.

Every Claude call in AtlasRAG goes through ``LLM`` so model selection, thinking,
structured output, and tool use are configured consistently in one place.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from .config import Settings


class LLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        # created on first call so indexing and pure retrieval work without a key
        if self._client is None:
            self.settings.require_api_key()
            self._client = anthropic.Anthropic()
        return self._client

    # --- plain completion -------------------------------------------------
    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 4096,
        thinking: bool = False,
        effort: str | None = None,
    ) -> str:
        """Return the concatenated text of a single (non-tool) completion."""
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if effort:
            kwargs.setdefault("output_config", {})["effort"] = effort

        resp = self.client.messages.create(**kwargs)
        return _text_of(resp)

    # --- structured (JSON-schema constrained) output ----------------------
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
        """Return a parsed object guaranteed to match ``schema``."""
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        resp = self.client.messages.create(**kwargs)
        text = _text_of(resp)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:  # pragma: no cover - defensive
            raise RuntimeError(f"Model did not return valid JSON: {e}\n{text[:500]}")

    # --- raw message (for tool-use loops) ---------------------------------
    def raw(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        model: str | None = None,
        max_tokens: int = 8192,
        thinking: bool = False,
    ):
        """Return the raw SDK response object (used by the agent loop)."""
        kwargs: dict[str, Any] = {
            "model": model or self.settings.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        return self.client.messages.create(**kwargs)


def _text_of(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text").strip()
