"""Claude in Microsoft Foundry (providers/foundry.py, CP-409): the Messages API passed through to the
customer's Foundry resource, with the deployment's name and the pod's own Entra ID token."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.policy import Provider
from llm_gateway.providers import foundry

from .conftest import call, policy

BASE = "https://acme-ai.services.ai.azure.com/anthropic"


def foundry_policy():
    pol = policy()
    pol.providers = {
        "foundry": Provider(
            type="foundry",
            base_url=BASE + "/",
            baa=True,
            models=["claude-*"],
            model_versions={"claude-sonnet-5-5": "acme-claude-sonnet"},
        )
    }
    return pol


class Foundry:
    def __init__(self):
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((str(request.url), dict(request.headers), body))
        if body.get("stream"):
            events = [
                {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 1}}},
                {"type": "message_delta", "usage": {"output_tokens": 5}},
            ]
            sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
            return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
        if request.url.path.endswith("/count_tokens"):
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
        return f"eyJ.entra-{calls['n']}", 4_000_000_000.0

    monkeypatch.setattr(foundry.FoundryAdapter, "token_source", staticmethod(token))
    monkeypatch.setitem(foundry._cache, "token", "")
    monkeypatch.setitem(foundry._cache, "lock", None)
    up = Foundry()
    up.token_calls = calls
    with TestClient(create_app(foundry_policy(), transport=httpx.MockTransport(up))) as c:
        up.client = c
        yield up


def test_messages_pass_through_with_the_deployment_and_the_pods_token(upstream):
    res = call(upstream.client, **{"anthropic-beta": "context-1m-2025-08-07"})
    assert res.status_code == 200 and res.json()["content"][0]["text"] == "Hello"
    url, headers, body = upstream.requests[0]
    assert url == f"{BASE}/v1/messages" and body["model"] == "acme-claude-sonnet"
    assert headers["authorization"] == "Bearer eyJ.entra-1" and "x-api-key" not in headers
    assert (
        headers["anthropic-version"] == "2023-06-01" and headers["anthropic-beta"] == "context-1m-2025-08-07"
    )
    call(upstream.client)
    assert upstream.token_calls["n"] == 1  # the token is reused until it nears its end


def test_streaming_and_token_counting(upstream):
    body = {
        "model": "claude-sonnet-5-5",
        "max_tokens": 50,
        "stream": True,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    res = call(upstream.client, body=body)
    assert res.status_code == 200 and b"message_delta" in res.content
    res = upstream.client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-sonnet-5-5", "messages": [{"role": "user", "content": "Hi"}]},
        headers={"x-api-key": "agent-key-123", "anthropic-version": "2023-06-01"},
    )
    assert res.status_code == 200 and res.json() == {"input_tokens": 12}
    url, _, body = upstream.requests[-1]
    assert url == f"{BASE}/v1/messages/count_tokens" and body["model"] == "acme-claude-sonnet"


@pytest.mark.parametrize(
    ("kwargs", "says"),
    [
        ({"base_url": "https://acme.openai.azure.com/openai"}, "services.ai.azure.com/anthropic"),
        ({"base_url": BASE, "api_key_env": "FOUNDRY_KEY"}, "takes no key"),
        ({"base_url": BASE, "operations": ["embeddings"]}, "serves messages only"),
    ],
)
def test_a_foundry_provider_is_checked(kwargs, says):
    with pytest.raises(ValueError, match=says):
        Provider(type="foundry", **kwargs)
