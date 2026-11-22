"""OpenAI's Responses API through the gateway (Codex): the same rules as messages."""

import hashlib

from .conftest import KEY, policy

TOOL_KEY = "sdlc-tool-key"
SESSION = {"budget_usd": 0.01, "ttl_s": 600, "data_class": "general", "models": ["gpt-*"]}


def body(**kw):
    return {
        "model": "gpt-5.4",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Fix the bug"}]}],
        "instructions": "You are a coder",
        **kw,
    }


def post(c, b, key=KEY):
    return c.post("/v1/responses", json=b, headers={"authorization": f"Bearer {key}"})


PRICES = {"claude-sonnet-*": {"input": 3.0, "output": 15.0}, "gpt-*": {"input": 2.0, "output": 8.0}}


def with_tool(data_class="general"):
    pol = policy(data_class)
    callers = {
        **{k: v.model_dump() for k, v in pol.callers.items()},
        "sdlc-tool": {
            "key_sha256": hashlib.sha256(TOOL_KEY.encode()).hexdigest(),
            "models": ["*"],
            "max_tokens": 32000,
            "sessions": {"max_budget_usd": 1.0, "max_ttl_s": 3600},
        },
    }
    return policy(data_class, callers=callers, pricing=PRICES)


def test_a_responses_call_goes_to_the_provider_with_its_key_and_is_metered(make_client, upstream):
    with make_client(policy("general")) as c:
        res = post(c, body())
        assert res.status_code == 200 and res.json()["output"][0]["content"][0]["text"] == "Done"
        url, headers, sent = upstream.requests[0]
        assert url == "https://openai.test/v1/responses" and headers["authorization"] == "Bearer sk-openai"
        assert sent["model"] == "gpt-5.4"


def test_a_streamed_call_counts_against_its_session(make_client, upstream):
    with make_client(with_tool()) as c:
        s = c.post("/v1/sessions", json=SESSION, headers={"x-api-key": TOOL_KEY}).json()
        res = post(c, body(stream=True, max_output_tokens=1000), key=s["token"])
        assert res.status_code == 200 and b"response.completed" in res.content
        seen = c.get(f"/v1/sessions/{s['id']}", headers={"x-api-key": TOOL_KEY}).json()
        assert seen["calls"] == 1 and seen["spent_usd"] > 0


def test_a_session_is_held_to_its_budget_before_anything_is_sent(make_client, upstream):
    """A hard cap: a call that could take the session past its budget is refused unsent, and calls
    that fit are let through until what they cost leaves no room for the next."""
    with make_client(with_tool()) as c:
        s = c.post(
            "/v1/sessions", json={**SESSION, "budget_usd": 0.02}, headers={"x-api-key": TOOL_KEY}
        ).json()
        big = post(c, body(max_output_tokens=4000), key=s["token"])
        assert big.status_code == 429 and "this session's budget" in big.json()["error"]["message"]
        assert not upstream.requests
        sent = 0
        while post(c, body(max_output_tokens=1000), key=s["token"]).status_code == 200:
            sent += 1
        seen = c.get(f"/v1/sessions/{s['id']}", headers={"x-api-key": TOOL_KEY}).json()
        assert sent >= 1 and seen["calls"] == sent == len(upstream.requests)
        assert seen["spent_usd"] <= 0.02


def test_secrets_in_the_input_are_redacted_before_they_leave(make_client, upstream):
    leaked = "AKIA" + "ABCDEFGH" + "IJKLMNOP"
    with make_client(policy("general")) as c:
        b = body(
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": f"key {leaked}"}]},
                {"type": "function_call_output", "call_id": "c1", "output": f"env: {leaked}"},
            ]
        )
        assert post(c, b).status_code == 200
        sent = str(upstream.requests[0][2])
        assert leaked not in sent


def test_phi_never_goes_to_a_provider_without_a_baa_and_anthropic_doesnt_serve_it(make_client, upstream):
    with make_client(policy("phi")) as c:
        assert post(c, body()).status_code == 403
        assert post(c, body(model="claude-sonnet-5-5")).status_code == 403
    assert not upstream.requests


def test_the_output_cap_applies(make_client, upstream):
    with make_client(policy("general")) as c:
        assert post(c, body(max_output_tokens=10**6)).status_code == 400
    assert not upstream.requests


def test_a_session_may_use_only_priced_models(make_client, upstream):
    pol = with_tool()
    pol.pricing.pop("gpt-*")
    with make_client(pol) as c:
        s = c.post("/v1/sessions", json=SESSION, headers={"x-api-key": TOOL_KEY}).json()
        res = post(c, body(), key=s["token"])
        assert res.status_code == 403 and "no price" in res.json()["error"]["message"]
    assert not upstream.requests
