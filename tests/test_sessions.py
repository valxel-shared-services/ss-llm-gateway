import hashlib

import pytest

from llm_gateway.sessions import SessionStore

from .conftest import KEY, call, policy

TOOL_KEY = "sdlc-tool-key"


def with_tool(data_class="phi", **grant):
    pol = policy(data_class)
    callers = {
        **{k: v.model_dump() for k, v in pol.callers.items()},
        "sdlc-tool": {
            "key_sha256": hashlib.sha256(TOOL_KEY.encode()).hexdigest(),
            "budget_usd_per_day": 5.0,
            "models": ["claude-*"],
            "max_tokens": 32000,
            "sessions": {"max_budget_usd": 2.0, "max_ttl_s": 3600, "max_open": 2, **grant},
        },
    }
    return policy(data_class, callers=callers)


def mint(c, key=TOOL_KEY, **body):
    body = {
        "budget_usd": 0.01,
        "ttl_s": 600,
        "data_class": "phi",
        "models": ["claude-sonnet-*"],
        "labels": {"feature": "F-1", "agent": "coder"},
        **body,
    }
    return c.post("/v1/sessions", json=body, headers={"x-api-key": key})


def test_a_session_token_calls_the_model_as_its_caller_and_accounts_to_the_session(make_client, upstream):
    with make_client(with_tool()) as c:
        res = mint(c)
        assert res.status_code == 201
        s = res.json()
        assert s["token"].startswith("vgs_") and s["budget_usd"] == 0.01
        # The session's labels are fixed; what it sends can't move its spend elsewhere.
        assert call(c, key=s["token"], **{"x-valxel-feature": "OTHER"}).status_code == 200
        assert upstream.requests[0][1]["x-api-key"] == "sk-real"  # the token never leaves the gateway
        seen = c.get(f"/v1/sessions/{s['id']}", headers={"x-api-key": TOOL_KEY}).json()
        assert seen["calls"] == 1 and seen["spent_usd"] == pytest.approx(0.006)
        # One more call takes it over its budget; the next is refused before it reaches a model.
        assert call(c, key=s["token"]).status_code == 200
        over = call(c, key=s["token"])
        assert over.status_code == 429 and "session" in over.json()["error"]["message"]
        assert len(upstream.requests) == 2


def test_revoked_and_expired_tokens_stop_working(make_client):
    with make_client(with_tool()) as c:
        s = mint(c).json()
        done = c.delete(f"/v1/sessions/{s['id']}", headers={"x-api-key": TOOL_KEY})
        assert done.status_code == 200 and done.json()["revoked"] is True
        assert call(c, key=s["token"]).status_code == 401
    clock = [1000.0]
    store = SessionStore(clock=lambda: clock[0])
    sess, token = store.mint(
        "sdlc-tool", budget_usd=1, ttl_s=60, models=["*"], labels={}, data_class="phi", max_open=5
    )
    assert store.find(token) is sess
    clock[0] += 61
    assert store.find(token) is None


def test_a_session_is_held_to_its_models_and_its_grant(make_client):
    with make_client(with_tool()) as c:
        s = mint(c, models=["claude-sonnet-*"]).json()
        res = call(c, {"model": "claude-opus-5-5", "max_tokens": 10, "messages": []}, key=s["token"])
        assert res.status_code == 403 and "session" in res.json()["error"]["message"]
        assert mint(c, budget_usd=3).status_code == 400  # over the grant's budget
        assert mint(c, ttl_s=7200).status_code == 400  # over the grant's lifetime
        assert mint(c, models=["gpt-5"]).status_code == 400  # a model the caller itself may not use
        assert mint(c).status_code == 201
        assert mint(c).status_code == 429  # two open already


def test_only_a_caller_with_a_grant_mints_and_never_with_a_session_token(make_client):
    with make_client(with_tool()) as c:
        assert mint(c, key=KEY).status_code == 403
        s = mint(c).json()
        assert mint(c, key=s["token"]).status_code == 401
        assert c.delete(f"/v1/sessions/{s['id']}", headers={"x-api-key": s["token"]}).status_code == 401
        assert c.get(f"/v1/sessions/{s['id']}", headers={"x-api-key": KEY}).status_code == 403


def test_a_gateway_serves_sessions_of_its_own_data_class_or_general_only(make_client):
    with make_client(with_tool("general")) as c:
        res = mint(c, data_class="phi")
        assert res.status_code == 403 and "general" in res.json()["error"]["message"]
        assert mint(c, data_class="general").status_code == 201
    with make_client(with_tool("phi")) as c:
        assert mint(c, data_class="payments").status_code == 403
        assert mint(c, data_class="general").status_code == 201


def test_streamed_calls_count_against_the_session(make_client):
    with make_client(with_tool()) as c:
        s = mint(c).json()
        res = c.post(
            "/v1/messages",
            headers={"x-api-key": s["token"]},
            json={"model": "claude-sonnet-5-5", "max_tokens": 10, "stream": True, "messages": []},
        )
        assert res.status_code == 200
        seen = c.get(f"/v1/sessions/{s['id']}", headers={"x-api-key": TOOL_KEY}).json()
        assert seen["calls"] == 1 and seen["spent_usd"] > 0
