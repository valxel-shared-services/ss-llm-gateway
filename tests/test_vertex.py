"""Claude on Vertex AI (providers/vertex.py, CP-112): the same Messages API in front, Vertex AI's
rawPredict behind, with the pod's own Google token."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.policy import Provider
from llm_gateway.providers import vertex

from .conftest import call, policy

BASE = "https://us-east5-aiplatform.googleapis.com/v1/projects/acme-ai/locations/us-east5/publishers/anthropic/models"


def vertex_policy():
    pol = policy()
    pol.providers = {
        "vertex": Provider(
            type="vertex",
            project="acme-ai",
            region="us-east5",
            baa=True,
            models=["claude-*"],
            model_versions={"claude-sonnet-5-5": "claude-sonnet-5-5@20260601"},
        )
    }
    return pol


class Vertex:
    def __init__(self):
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((str(request.url), dict(request.headers), body))
        if request.url.path.endswith(":streamRawPredict"):
            events = [
                {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1}}},
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}},
                {"type": "message_delta", "usage": {"output_tokens": 5}},
            ]
            sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
            return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
        if "count-tokens" in request.url.path:
            return httpx.Response(200, json={"input_tokens": 12})
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 200},
            },
        )


@pytest.fixture
def upstream(monkeypatch):
    calls = {"n": 0}

    def token():
        calls["n"] += 1
        return f"ya29.token-{calls['n']}", 4_000_000_000.0

    monkeypatch.setattr(vertex.VertexAdapter, "token_source", staticmethod(token))
    monkeypatch.setitem(vertex._cache, "token", "")
    monkeypatch.setitem(vertex._cache, "lock", None)
    up = Vertex()
    up.token_calls = calls
    with TestClient(create_app(vertex_policy(), transport=httpx.MockTransport(up))) as c:
        up.client = c
        yield up


def test_messages_go_to_raw_predict_with_the_pods_token(upstream):
    res = call(upstream.client, **{"anthropic-beta": "context-1m-2025-08-07"})
    assert res.status_code == 200 and res.json()["content"][0]["text"] == "Hello"
    url, headers, body = upstream.requests[0]
    assert url == f"{BASE}/claude-sonnet-5-5@20260601:rawPredict"
    assert headers["authorization"] == "Bearer ya29.token-1" and "x-api-key" not in headers
    assert headers["anthropic-beta"] == "context-1m-2025-08-07"
    assert "model" not in body and body["anthropic_version"] == "vertex-2023-10-16"
    call(upstream.client)
    assert upstream.token_calls["n"] == 1  # the token is reused until it nears its end


def test_streaming_uses_stream_raw_predict_and_counts_usage(upstream):
    body = {
        "model": "claude-sonnet-5-5",
        "max_tokens": 50,
        "stream": True,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    res = call(upstream.client, body=body)
    assert res.status_code == 200 and b"message_delta" in res.content
    assert upstream.requests[0][0].endswith(":streamRawPredict")


def test_token_counting_names_the_model_in_the_body(upstream):
    res = upstream.client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-sonnet-5-5", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"x-api-key": "agent-key-123", "anthropic-version": "2023-06-01"},
    )
    assert res.status_code == 200 and res.json() == {"input_tokens": 12}
    url, _, body = upstream.requests[0]
    assert url == f"{BASE}/count-tokens:rawPredict" and body["model"] == "claude-sonnet-5-5@20260601"


def test_a_vertex_provider_is_checked():
    assert (
        Provider(type="vertex", project="p", region="global").base_url == "https://aiplatform.googleapis.com"
    )
    for bad, says in (
        ({}, "project and region"),
        ({"project": "p", "region": "r", "api_key_env": "K"}, "no key"),
        ({"project": "p", "region": "r", "operations": ["embeddings"]}, "messages only"),
    ):
        with pytest.raises(ValueError, match=says):
            Provider(type="vertex", **bad)
    with pytest.raises(ValueError, match="base_url"):
        Provider(type="anthropic")
