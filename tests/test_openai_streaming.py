"""Streaming from OpenAI-compatible providers, as the Messages API's events."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.providers.openai import StreamTranslator

from .conftest import call, policy

CHUNKS = [
    {"id": "chatcmpl-9", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
    {"id": "chatcmpl-9", "choices": [{"index": 0, "delta": {"content": "Looking "}}]},
    {"id": "chatcmpl-9", "choices": [{"index": 0, "delta": {"content": "it up."}}]},
    # Two calls, their arguments interleaved: each still arrives whole and in order.
    {
        "id": "chatcmpl-9",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q": '},
                        }
                    ]
                },
            }
        ],
    },
    {
        "id": "chatcmpl-9",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "fetch", "arguments": '{"id"'},
                        }
                    ]
                },
            }
        ],
    },
    {
        "id": "chatcmpl-9",
        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]}}],
    },
    {
        "id": "chatcmpl-9",
        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": ": 7}"}}]}}],
    },
    {"id": "chatcmpl-9", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    {
        "id": "chatcmpl-9",
        "choices": [],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 9,
            "prompt_tokens_details": {"cached_tokens": 10},
        },
    },
]


def chat_stream(status=200, chunks=CHUNKS):
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if status >= 400:
            return httpx.Response(status, json={"error": {"message": "model overloaded"}})
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    return handler, sent


def events(raw: bytes) -> list[dict]:
    return [json.loads(line[5:]) for line in raw.split(b"\n") if line.startswith(b"data:")]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("T_OPENAI", "sk-openai")

    def make(handler):
        return TestClient(create_app(policy("general"), transport=httpx.MockTransport(handler)))

    return make


def body():
    return {
        "model": "gpt-5",
        "max_tokens": 50,
        "stream": True,
        "messages": [{"role": "user", "content": "find x"}],
    }


def test_a_chat_stream_arrives_as_messages_events_and_is_charged(client, caplog):
    handler, sent = chat_stream()
    with client(handler) as c, caplog.at_level("INFO", logger="llm_gateway.usage"):
        res = call(c, body())
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    assert sent[0]["stream"] is True and sent[0]["stream_options"] == {"include_usage": True}
    ev = events(res.content)
    assert [e["type"] for e in ev] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert (
        "".join(e["delta"]["text"] for e in ev if e.get("delta", {}).get("type") == "text_delta")
        == "Looking it up."
    )
    tools = [e["content_block"] for e in ev if e["type"] == "content_block_start"][1:]
    args = [
        json.loads(e["delta"]["partial_json"])
        for e in ev
        if e.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert [(t["id"], t["name"]) for t in tools] == [("call_a", "lookup"), ("call_b", "fetch")]
    assert args == [{"q": "x"}, {"id": 7}]
    assert [e["index"] for e in ev if e["type"] == "content_block_start"] == [0, 1, 2]
    delta = ev[-2]
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert delta["usage"]["input_tokens"] == 40 and delta["usage"]["output_tokens"] == 9
    record = json.loads(next(r.message for r in caplog.records if "llm_usage" in r.message))
    assert (record["input_tokens"], record["output_tokens"], record["cache_read_input_tokens"]) == (40, 9, 10)


def test_an_upstream_error_is_an_error_event(client):
    handler, _ = chat_stream(status=503)
    with client(handler) as c:
        ev = events(call(c, body()).content)
    assert ev == [{"type": "error", "error": {"type": "api_error", "message": "model overloaded"}}]


def test_a_reply_with_text_only_and_no_usage():
    usage: dict = {}
    t = StreamTranslator("m", usage)
    out = t.feed({"id": "x", "choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]}) + t.finish()
    ev = events(b"".join(out))
    assert [e["type"] for e in ev] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert ev[-2]["delta"]["stop_reason"] == "end_turn" and usage == {}
