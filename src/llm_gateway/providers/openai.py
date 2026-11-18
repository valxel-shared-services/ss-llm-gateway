"""OpenAI-compatible providers (OpenAI, Azure OpenAI, vLLM, and others that speak it).

Messages: chat completions, translated from and to the Anthropic Messages API (the common subset:
system, text, tool definitions, tool calls and tool results); not streamed.

Responses: OpenAI's Responses API, passed through as it is, streamed or not: what agents built
on it send (Codex). Usage is read from the response, or the `response.completed` event, and
given to the ledger in the Messages API's shape."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..policy import Provider
from . import Result, auth

STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "refusal"}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def to_openai(body: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if body.get("system"):
        messages.append({"role": "system", "content": _text(body["system"])})
    for m in body.get("messages", []):
        content = m["content"]
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        calls = [b for b in content if b.get("type") == "tool_use"]
        results = [b for b in content if b.get("type") == "tool_result"]
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": _text(content) or None}
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c.get("input") or {})},
                    }
                    for c in calls
                ]
            messages.append(msg)
        else:
            for r in results:
                messages.append(
                    {"role": "tool", "tool_call_id": r["tool_use_id"], "content": _text(r.get("content"))}
                )
            if _text(content):
                messages.append({"role": "user", "content": _text(content)})
    out: dict[str, Any] = {"model": body["model"], "messages": messages, "max_tokens": body.get("max_tokens")}
    if body.get("tools"):
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {},
                },
            }
            for t in body["tools"]
        ]
    for k in ("temperature", "top_p"):
        if k in body:
            out[k] = body[k]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    return out


def from_openai(data: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if msg.get("content"):
        content.append({"type": "text", "text": msg["content"]})
    for c in msg.get("tool_calls") or []:
        content.append(
            {
                "type": "tool_use",
                "id": c["id"],
                "name": c["function"]["name"],
                "input": json.loads(c["function"].get("arguments") or "{}"),
            }
        )
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": STOP.get(choice.get("finish_reason", "stop"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": (u.get("prompt_tokens") or 0) - cached,
            "output_tokens": u.get("completion_tokens") or 0,
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        },
    }


class OpenAIAdapter:
    def __init__(self, provider: Provider, client: httpx.AsyncClient) -> None:
        self.provider = provider
        self.client = client

    async def create(self, body: dict[str, Any], headers: dict[str, str]) -> Result:
        res = await self.client.post(
            f"{self.provider.base_url}/v1/chat/completions",
            json=to_openai(body),
            headers=auth(self.provider),
            timeout=600,
        )
        data = res.json()
        if res.status_code >= 400:
            message = (data.get("error") or {}).get("message", "upstream error")
            return Result(
                status=res.status_code,
                usage={},
                body={"type": "error", "error": {"type": "api_error", "message": message}},
            )
        out = from_openai(data, body["model"])
        return Result(status=200, body=out, usage=out["usage"])

    async def stream(
        self, body: dict[str, Any], headers: dict[str, str], usage: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("streaming is only available from Anthropic-type providers")
        yield b""  # pragma: no cover


def responses_usage(response: dict[str, Any]) -> dict[str, int]:
    """A Responses API usage block in the shape the ledger prices (the Messages API's)."""
    u = response.get("usage") or {}
    cached = int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    return {
        "input_tokens": max(0, int(u.get("input_tokens") or 0) - cached),
        "output_tokens": int(u.get("output_tokens") or 0),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
    }


class ResponsesAdapter:
    """OpenAI's Responses API, passed through."""

    def __init__(self, provider: Provider, client: httpx.AsyncClient) -> None:
        self.provider = provider
        self.client = client

    @property
    def url(self) -> str:
        return f"{self.provider.base_url}/v1/responses"

    async def create(self, body: dict[str, Any]) -> Result:
        res = await self.client.post(self.url, json=body, headers=auth(self.provider), timeout=600)
        try:
            data = res.json()
        except ValueError:
            data = {"error": {"message": res.text[:500] or "upstream error"}}
        if res.status_code >= 400:
            return Result(status=res.status_code, usage={}, body=data)
        return Result(status=200, body=data, usage=responses_usage(data))

    async def stream(self, body: dict[str, Any], usage: dict[str, Any]) -> AsyncIterator[bytes]:
        async with self.client.stream(
            "POST", self.url, json=body, headers=auth(self.provider), timeout=600
        ) as res:
            buffer = b""
            async for chunk in res.aiter_bytes():
                yield chunk
                buffer += chunk
                while b"\n\n" in buffer:  # usage arrives with response.completed
                    event, buffer = buffer.split(b"\n\n", 1)
                    for line in event.split(b"\n"):
                        if not line.startswith(b"data:"):
                            continue
                        try:
                            data = json.loads(line[5:])
                        except ValueError:
                            continue
                        if data.get("type") in ("response.completed", "response.incomplete"):
                            usage.update(responses_usage(data.get("response") or {}))
