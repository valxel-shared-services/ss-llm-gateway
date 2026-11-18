import hashlib

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.audit import Auditor, MemorySink

from .conftest import KEY, call, policy

TOOL_KEY = "sdlc-tool-key"
GPT = {"model": "gpt-5", "max_tokens": 10, "messages": [{"role": "user", "content": "Hello"}]}


def tools_gateway(**tool):
    """A gateway set up for general data, with an orchestrator that may also declare payments and PHI."""
    pol = policy("general")
    callers = {
        **{k: v.model_dump() for k, v in pol.callers.items()},
        "sdlc-tool": {
            "key_sha256": hashlib.sha256(TOOL_KEY.encode()).hexdigest(),
            "models": ["*"],
            "data_classes": ["payments", "phi"],
            "sessions": {"max_budget_usd": 2.0, "max_ttl_s": 3600},
            **tool,
        },
    }
    return policy("general", callers=callers)


@pytest.fixture
def client(upstream, monkeypatch):
    monkeypatch.setenv("T_ANTHROPIC", "sk-real")
    monkeypatch.setenv("T_OPENAI", "sk-openai")
    sink = MemorySink()
    app = create_app(
        tools_gateway(),
        transport=httpx.MockTransport(upstream),
        auditor=Auditor(sink, service="llm-gateway", product="emr", data_class="general"),
    )
    with TestClient(app) as c:
        c.sink = sink
        yield c


def test_a_declared_class_gets_its_own_rule(client, upstream):
    # General data: OpenAI is allowed, and secrets are redacted.
    assert call(client, GPT, key=TOOL_KEY).status_code == 200
    # PHI: the rule needs a BAA, so OpenAI is refused and the request never leaves.
    res = call(client, GPT, key=TOOL_KEY, **{"x-valxel-data-class": "phi"})
    assert res.status_code == 403 and "phi" in res.json()["error"]["message"]
    assert len(upstream.requests) == 1
    # ...and Anthropic (under a BAA) serves it, with the PHI rule's redaction (off).
    assert call(client, key=TOOL_KEY, **{"x-valxel-data-class": "phi"}).status_code == 200
    classes = [e["data"]["data_class"] for e in client.sink.events if e["data"]["action"] == "llm.request"]
    assert classes == ["general", "phi", "phi"]


def test_a_caller_declares_only_the_classes_it_may(client, upstream):
    res = call(client, key=KEY, **{"x-valxel-data-class": "phi"})
    assert res.status_code == 403 and "may not send phi" in res.json()["error"]["message"]
    assert call(client, key=TOOL_KEY, **{"x-valxel-data-class": "secret-stuff"}).status_code == 403
    assert not upstream.requests


def test_a_session_is_held_to_the_class_it_was_minted_for(client, upstream):
    s = client.post(
        "/v1/sessions",
        headers={"x-api-key": TOOL_KEY},
        json={"budget_usd": 1, "ttl_s": 60, "data_class": "phi", "models": ["*"]},
    ).json()
    # Declaring general on the request doesn't loosen it: OpenAI stays refused.
    res = call(client, GPT, key=s["token"], **{"x-valxel-data-class": "general"})
    assert res.status_code == 403 and not upstream.requests


def test_a_policy_cant_let_a_caller_declare_a_class_with_no_rule():
    pol = policy("general").model_dump()
    pol["data_classes"].pop("phi")
    pol["callers"]["intake-agent"]["data_classes"] = ["phi"]
    with pytest.raises(ValueError, match="no rule"):
        type(policy("general")).model_validate(pol)
