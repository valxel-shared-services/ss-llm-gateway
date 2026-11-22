"""Embeddings and rerank: the same rules as messages (keys, models, the data class's providers,
redaction, budgets, audit first), and one answer shape whichever provider serves them."""

import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import check_texts, create_app, labels_of
from llm_gateway.audit import Auditor, AuditUnavailable
from llm_gateway.ledger import cost_usd
from llm_gateway.policy import Policy, Provider
from llm_gateway.redaction import scrub_fields

KEY = "knowledge-key-1"

PROVIDERS = {
    "anthropic": {
        "type": "anthropic",
        "base_url": "https://anthropic.test",
        "api_key_env": "T_ANTHROPIC",
        "baa": True,
        "models": ["claude-*"],
    },
    "openai": {
        "type": "openai",
        "base_url": "https://openai.test",
        "api_key_env": "T_OPENAI",
        "models": ["text-embedding-*"],
    },
    "vllm": {
        "type": "openai",
        "base_url": "http://vllm.local:8000/",
        "self_hosted": True,
        "models": ["bge-*"],
        "operations": ["embeddings", "rerank"],
    },
    "voyage": {
        "type": "openai",
        "base_url": "https://voyage.test",
        "api_key_env": "T_VOYAGE",
        "baa": True,
        "models": ["rerank-*"],
        "operations": ["rerank"],
    },
}


def policy(data_class="phi", budget=1.0, **caller):
    return Policy.model_validate(
        {
            "product": "emr",
            "data_class": data_class,
            "providers": PROVIDERS,
            "data_classes": {
                "phi": {"require_baa": True, "redaction": "off"},
                "payments": {"redaction": "block", "detectors": ["card_number"]},
                "general": {"redaction": "redact", "detectors": ["us_ssn"]},
            },
            "callers": {
                "notes-knowledge": {
                    "key_sha256": hashlib.sha256(KEY.encode()).hexdigest(),
                    "budget_usd_per_day": budget,
                    "models": ["bge-*", "text-embedding-*", "rerank-*", "claude-*"],
                    **caller,
                }
            },
            "pricing": {
                "bge-*": {"input": 0.0},
                "text-embedding-*": {"input": 0.02},
                "rerank-*": {"input": 0.05, "per_request": 0.002},
                "claude-*": {"input": 3.0, "output": 15.0},
            },
        }
    )


class Upstream:
    def __init__(self, status=200):
        self.requests = []
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((str(request.url), dict(request.headers), body))
        if self.status >= 400:
            return httpx.Response(self.status, json={"error": {"message": "model overloaded"}})
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [0.1 * (i + 1)] * 4}
                        for i in range(len(body["input"]))
                    ],
                    "model": "upstream-name",
                    "usage": {"prompt_tokens": 12, "total_tokens": 12},
                },
            )
        docs = body["documents"]
        rows = [{"index": i, "relevance_score": round(1 / (i + 2), 3)} for i in reversed(range(len(docs)))]
        if request.url.host == "voyage.test":  # Voyage answers in `data`, with `usage.total_tokens`
            return httpx.Response(200, json={"data": rows, "usage": {"total_tokens": 40}})
        rows.append({"index": 99, "relevance_score": 0.9})  # out of range: dropped
        return httpx.Response(
            200, json={"id": "rr-1", "results": rows, "meta": {"billed_units": {"total_tokens": 30}}}
        )


class Sink:
    def __init__(self, fail=False):
        self.events, self.fail = [], fail

    def publish(self, key, event):
        if self.fail:
            raise AuditUnavailable("down")
        self.events.append(event)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("T_ANTHROPIC", "sk-a")
    monkeypatch.setenv("T_OPENAI", "sk-o")
    monkeypatch.setenv("T_VOYAGE", "sk-v")


def client(pol, upstream, sink=None):
    auditor = Auditor(
        sink or Sink(), service="llm-gateway", product="emr", environment="test", data_class=pol.data_class
    )
    return TestClient(create_app(pol, transport=httpx.MockTransport(upstream), auditor=auditor))


def post(c, path, body, key=KEY, **headers):
    return c.post(path, json=body, headers={"authorization": f"Bearer {key}", **headers})


EMBED = {
    "model": "bge-m3",
    "input": ["Patient reports chest pain", "Follow up in two weeks"],
    "dimensions": 4,
}
RERANK = {
    "model": "bge-reranker-v2",
    "query": "chest pain",
    "documents": ["a", {"text": "b"}, "c"],
    "top_n": 2,
}


def test_embeddings_go_to_a_self_hosted_model_for_phi_without_a_key(env):
    up = Upstream()
    with client(policy(), up) as c:
        res = post(
            c,
            "/v1/embeddings",
            EMBED,
            **{"x-valxel-component": "notes-knowledge", "x-valxel-purpose": "index"},
        )
    assert res.status_code == 200, res.text
    url, headers, sent = up.requests[0]
    assert url == "http://vllm.local:8000/v1/embeddings"
    assert "authorization" not in headers  # a self-hosted model takes no key; the caller's never leaves
    assert sent == EMBED
    data = res.json()
    assert [r["index"] for r in data["data"]] == [0, 1] and data["model"] == "bge-m3"
    assert data["usage"] == {"prompt_tokens": 12, "total_tokens": 12}
    assert res.headers["request-id"]


def test_phi_embeddings_never_reach_a_provider_without_a_baa(env):
    up = Upstream()
    with client(policy(), up) as c:
        res = post(c, "/v1/embeddings", {**EMBED, "model": "text-embedding-3-large"})
    assert res.status_code == 403 and "phi" in res.json()["error"]["message"]
    assert "for embeddings" in res.json()["error"]["message"]
    assert up.requests == []


def test_general_products_may_embed_with_a_hosted_provider_using_the_platform_key(env):
    up = Upstream()
    with client(policy("general"), up) as c:
        res = post(c, "/v1/embeddings", {"model": "text-embedding-3-small", "input": "SSN 123-45-6789"})
    assert res.status_code == 200
    url, headers, sent = up.requests[0]
    assert url == "https://openai.test/v1/embeddings" and headers["authorization"] == "Bearer sk-o"
    assert sent["input"] == "SSN [US_SSN REDACTED]"


def test_a_provider_that_doesnt_do_the_operation_is_not_chosen(env):
    up = Upstream()
    with client(policy(), up) as c:
        # Anthropic serves claude-* but only messages; voyage serves rerank-* but not embeddings.
        assert post(c, "/v1/embeddings", {**EMBED, "model": "claude-sonnet-5-5"}).status_code == 403
        assert post(c, "/v1/embeddings", {**EMBED, "model": "rerank-2"}).status_code == 403
    assert up.requests == []


def test_rerank_answers_in_one_shape_best_first(env):
    up = Upstream()
    with client(policy(), up) as c:
        res = post(c, "/v1/rerank", RERANK)
    assert res.status_code == 200, res.text
    url, _, sent = up.requests[0]
    assert url == "http://vllm.local:8000/v1/rerank"
    assert sent == {
        "model": "bge-reranker-v2",
        "query": "chest pain",
        "documents": ["a", "b", "c"],
        "top_n": 2,
    }
    body = res.json()
    assert [r["index"] for r in body["results"]] == [0, 1]
    assert body["results"][0]["relevance_score"] == 0.5
    assert body["usage"] == {"input_tokens": 30} and body["model"] == "bge-reranker-v2"
    assert "documents" not in body


def test_voyage_style_rerank_is_normalized_and_priced_per_search(env, caplog):
    up = Upstream()
    caplog.set_level("INFO", logger="llm_gateway.usage")
    with client(policy(), up) as c:
        res = post(c, "/v1/rerank", {"model": "rerank-2", "query": "q", "documents": ["a", "b"]})
    assert res.status_code == 200
    assert up.requests[0][1]["authorization"] == "Bearer sk-v"
    assert [r["index"] for r in res.json()["results"]] == [0, 1]
    usage = next(json.loads(r.message) for r in caplog.records if "llm_usage" in r.message)
    assert usage["input_tokens"] == 40 and usage["cost_usd"] == pytest.approx(0.002 + 40 * 0.05 / 1e6)


@pytest.mark.parametrize(
    ("path", "body", "message"),
    [
        ("/v1/embeddings", {"model": "bge-m3", "input": []}, "input must be"),
        ("/v1/embeddings", {"model": "bge-m3", "input": [1, 2]}, "input must be"),
        ("/v1/embeddings", {"model": "bge-m3", "input": ["x"] * 4}, "the cap is 3"),
        ("/v1/rerank", {"model": "bge-r", "query": "", "documents": ["a"]}, "query must be"),
        ("/v1/rerank", {"model": "bge-r", "query": "q", "documents": []}, "documents must be"),
        ("/v1/rerank", {"model": "bge-r", "query": "q", "documents": ["a"] * 4}, "the cap is 3"),
    ],
)
def test_texts_are_checked_and_capped(env, path, body, message):
    up = Upstream()
    with client(policy(max_inputs=3), up) as c:
        res = post(c, path, body)
    assert res.status_code == 400 and message in res.json()["error"]["message"]
    assert up.requests == []


def test_keys_models_budgets_and_bodies_are_checked_as_for_messages(env):
    up = Upstream()
    with client(policy(budget=0.0), up) as c:
        assert post(c, "/v1/embeddings", EMBED, key="wrong").status_code == 401
        assert post(c, "/v1/rerank", {**RERANK, "model": "gpt-5"}).status_code == 403
        assert post(c, "/v1/embeddings", EMBED).status_code == 429
        res = c.post("/v1/embeddings", content=b"nope", headers={"authorization": f"Bearer {KEY}"})
        assert res.status_code == 400
        res = c.post("/v1/embeddings", json=["a"], headers={"authorization": f"Bearer {KEY}"})
        assert res.status_code == 400 and "object" in res.json()["error"]["message"]
    assert up.requests == []


def test_payments_block_card_numbers_in_documents(env):
    up = Upstream()
    pol = policy("payments")
    with client(pol, up) as c:
        res = post(
            c, "/v1/rerank", {"model": "bge-r", "query": "q", "documents": [{"text": "4111 1111 1111 1111"}]}
        )
    assert res.status_code == 400 and "card_number" in res.json()["error"]["message"]
    assert up.requests == []


def test_every_call_is_audited_first_and_refused_when_the_trail_is_down(env):
    up, sink = Upstream(), Sink()
    with client(policy(), up, sink) as c:
        post(c, "/v1/rerank", RERANK, **{"x-valxel-component": "notes-knowledge"})
    event = sink.events[-1]
    details = event["data"]["details"] if "data" in event else event["details"]
    assert details["operation"] == "rerank" and details["labels"]["component"] == "notes-knowledge"
    assert "chest pain" not in json.dumps(event)  # never content

    up = Upstream()
    with client(policy(), up, Sink(fail=True)) as c:
        assert post(c, "/v1/embeddings", EMBED).status_code == 503
    assert up.requests == []


@pytest.mark.parametrize("path", ["/v1/embeddings", "/v1/rerank"])
def test_upstream_errors_and_outages_are_reported_not_charged(env, path, caplog):
    caplog.set_level("INFO", logger="llm_gateway.usage")
    body = EMBED if path.endswith("embeddings") else RERANK
    with client(policy(), Upstream(status=503)) as c:
        res = post(c, path, body)
    assert res.status_code == 503 and res.json()["error"]["message"] == "model overloaded"
    assert not [r for r in caplog.records if "llm_usage" in r.message]

    def down(request):
        raise httpx.ConnectError("refused")

    pol = policy()
    with TestClient(
        create_app(
            pol,
            transport=httpx.MockTransport(down),
            auditor=Auditor(Sink(), service="g", product="emr", environment="t", data_class="phi"),
        )
    ) as c:
        res = post(c, path, body)
    assert res.status_code == 502 and "ConnectError" in res.json()["error"]["message"]


def test_non_json_upstream_errors_are_still_reported(env):
    def broken(request):
        return httpx.Response(500, content=b"<html>bad gateway</html>")

    pol = policy()
    with TestClient(
        create_app(
            pol,
            transport=httpx.MockTransport(broken),
            auditor=Auditor(Sink(), service="g", product="emr", environment="t", data_class="phi"),
        )
    ) as c:
        res = post(c, "/v1/embeddings", EMBED)
    assert res.status_code == 500 and res.json()["error"]["message"] == "upstream error"


def test_readiness_ignores_self_hosted_models_without_keys(env, monkeypatch):
    with client(policy(), Upstream()) as c:
        assert c.get("/readyz").status_code == 200
    monkeypatch.delenv("T_VOYAGE")
    with client(policy(), Upstream()) as c:
        res = c.get("/readyz")
    assert res.status_code == 503 and res.json()["missing_keys"] == ["voyage"]


def test_policy_rules():
    with pytest.raises(ValueError, match="messages only"):
        Provider(type="anthropic", base_url="x", operations=["embeddings"])
    assert Provider(type="openai", base_url="x").does("embeddings")
    assert not Provider(type="openai", base_url="x").does("rerank")
    assert not Provider(type="anthropic", base_url="x").does("embeddings")
    pol = policy()
    assert pol.provider_for("bge-m3", "rerank") == ("vllm", pol.providers["vllm"])
    assert pol.provider_for("claude-sonnet-5-5")[0] == "anthropic"
    assert cost_usd(pol, "rerank-2", {"input_tokens": 1_000_000}) == pytest.approx(0.052)


def test_helpers():
    assert labels_of({"X-Valxel-Component": "k", "x-vrixel-feature": "F-1", "x-other": "n"}) == {
        "component": "k",
        "feature": "F-1",
    }
    assert check_texts({"input": "one"}, "embeddings", 1) == ""
    out, findings = scrub_fields(
        {"query": "SSN 123-45-6789", "documents": ["ok", {"text": "123-45-6789", "id": 1}]},
        ("query", "documents"),
        ["us_ssn"],
    )
    assert findings.counts == {"us_ssn": 2}
    assert out["documents"][1] == {"text": "[US_SSN REDACTED]", "id": 1}
