"""Every credential the platform hands out, and the common ones a repo or log may hold, is redacted
before text reaches a model (review finding 2). A credential type the platform starts issuing gets
a real-shaped example here first. The examples are made up: right prefix, length and alphabet."""

from __future__ import annotations

import hashlib

import pytest

from llm_gateway.policy import Policy
from llm_gateway.redaction import Findings, scrub

from .conftest import KEY, policy

A36 = "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5"  # 36 alphanumerics
CORPUS = {
    "gateway session (vgs_)": "vgs_" + "Xy7_kP-2mQ9rT4wZ8vB1nC6dF3gH5jK0lM2",
    "GitHub installation (ghs_)": "ghs_" + A36,
    "GitHub personal (ghp_)": "ghp_" + A36,
    "GitHub OAuth (gho_)": "gho_" + A36,
    "GitHub user-to-server (ghu_)": "ghu_" + A36,
    "GitHub refresh (ghr_)": "ghr_" + A36 + "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9vW1xY3zA5bC7d",
    "GitHub fine-grained": "github_pat_11ABCDEFG0aBcDeFgHiJkL_"
    + "mNoPqRsTuVwXyZ0123456789aBcDeFgHiJkLmNoPqRsTuVwXyZ01234567",
    "AWS access key": "AKIA" + "IOSFODNN7EXAMPLE",
    "AWS temporary key": "ASIA" + "IOSFODNN7EXAMPLE",
    "AWS session token": "IQoJb3JpZ2luX2VjEJr" + "a" * 120 + "/b+c==",
    "Anthropic": "sk-ant-api03-" + "x" * 40,
    "OpenAI project": "sk-proj-" + "Ab1_" * 12,
    "Slack": "xoxb-123456789012-123456789012-" + "a" * 24,
    "JWT": "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJhZ2VudCJ9.c2lnbmF0dXJlLXZhbHVl",
    "private key": "-----BEGIN OPENSSH PRIVATE KEY-----",
}


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_each_credential_is_redacted_whole(name: str) -> None:
    secret = CORPUS[name]
    f = Findings()
    out = scrub(f"git push failed: remote: token {secret} was rejected", ["secret"], f)
    assert f.counts.get("secret") == 1, name
    assert secret not in out and secret[8:24] not in out, f"{name}: part of it survived: {out}"


@pytest.mark.parametrize(
    "text",
    [
        "the ghs_ prefix marks installation tokens",  # the prefix alone
        "AKIA is how access key ids start",
        "sk-learn is a library",
        "ghs_short",
    ],
)
def test_words_about_credentials_are_left_alone(text: str) -> None:
    f = Findings()
    assert scrub(text, ["secret"], f) == text and not f.counts


def test_a_caller_glob_no_price_could_cover_is_refused_at_load() -> None:
    with pytest.raises(ValueError, match="caller intake-agent may use mistral-"):
        policy(
            callers={
                "intake-agent": {
                    "key_sha256": hashlib.sha256(KEY.encode()).hexdigest(),
                    "budget_usd_per_day": 1.0,
                    "models": ["claude-sonnet-*", "mistral-*"],
                }
            }
        )
    pol: Policy = policy()  # claude-sonnet-* and gpt-* both priced
    assert pol.price("gpt-5") is not None


def test_an_unpriced_model_is_refused_for_every_caller_not_only_sessions(make_client, upstream) -> None:
    pol = policy("general")  # OpenAI may serve general data; only the price is missing
    pol.pricing.pop("gpt-*")
    with make_client(pol) as c:
        res = c.post(
            "/v1/messages",
            json={"model": "gpt-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": KEY},
        )
    assert res.status_code == 403 and "no price" in res.json()["error"]["message"]
    assert not upstream.requests  # refused before any provider was called
