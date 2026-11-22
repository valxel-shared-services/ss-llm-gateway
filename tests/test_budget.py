"""Budgets as a hard cap, and spend and sessions shared by replicas (review finding 9, T-137)."""

from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.app import create_app
from llm_gateway.budget import (
    RESERVATION_TTL_S,
    MemoryBudget,
    OverBudget,
    PostgresBudget,
    PostgresSessionStore,
    worst_case_usd,
)
from llm_gateway.policy import Policy
from llm_gateway.sessions import SessionRefused, SessionStore

from .conftest import KEY, call, policy

DB = os.environ.get("GATEWAY_TEST_DATABASE_URL", "")
needs_db = pytest.mark.skipif(not DB, reason="set GATEWAY_TEST_DATABASE_URL to run the Postgres tests")


def test_the_worst_case_is_the_whole_request_as_input_and_the_whole_output_allowance() -> None:
    pol: Policy = policy()
    body = {"model": "claude-sonnet-5-5", "messages": [{"role": "user", "content": "x" * 4000}]}
    price = pol.price("claude-sonnet-5-5")
    assert price is not None
    w = worst_case_usd(pol, "claude-sonnet-5-5", body, 1000)
    assert w >= (2000 * price.input + 1000 * price.output) / 1_000_000  # 4,000 characters: 2,000+ tokens
    assert worst_case_usd(pol, "unpriced-model", body, 1000) == 0.0


def test_calls_in_flight_together_cant_pass_the_cap() -> None:
    clock = [1000.0]
    b = MemoryBudget(SessionStore(), clock=lambda: clock[0])
    held = [b.reserve("agent", 1.0, 0.3) for _ in range(3)]
    with pytest.raises(OverBudget, match="agent's daily budget"):
        b.reserve("agent", 1.0, 0.3)  # 0.9 held: a fourth could pass 1.0
    b.settle(held[0], 0.05)  # answered: it cost less than it held
    b.release(held[1])  # failed: costs nothing
    assert b.spent_today("agent") == pytest.approx(0.05)
    b.reserve("agent", 1.0, 0.6)
    clock[0] += RESERVATION_TTL_S + 1  # never settled (a crash): stops counting
    b.reserve("agent", 1.0, 0.9)


def test_a_spent_budget_refuses_even_a_free_call() -> None:
    b = MemoryBudget(SessionStore())
    with pytest.raises(OverBudget):
        b.reserve("agent", 0.0, 0.0)


def test_a_session_is_held_to_its_own_budget_too() -> None:
    sessions = SessionStore()
    b = MemoryBudget(sessions)
    s, _ = sessions.mint(
        "tool", budget_usd=0.5, ttl_s=60, models=["*"], labels={}, data_class="general", max_open=5
    )
    r = b.reserve("tool", 100.0, 0.4, s)
    with pytest.raises(OverBudget, match="this session's budget"):
        b.reserve("tool", 100.0, 0.2, s)
    b.settle(r, 0.1)
    view = sessions.get(s.id, "tool")
    assert view is not None and view.calls == 1 and view.spent_usd == pytest.approx(0.1)
    b.reserve("tool", 100.0, 0.2, view)


def test_many_concurrent_calls_take_exactly_what_fits(make_client: Any, upstream: Any) -> None:
    """Through the app: twenty concurrent callers against a budget that fits only some of them."""
    pol = policy()
    pol.callers["intake-agent"].budget_usd_per_day = 0.05
    real = upstream.__call__

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.5)  # every call is admitted, and holds its reservation, before any answers
        return real(request)

    app = create_app(pol, transport=httpx.MockTransport(slow))
    with TestClient(app) as c:
        body = {
            "model": "claude-sonnet-5-5",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        with ThreadPoolExecutor(20) as pool:
            futures = [pool.submit(call, c, body) for _ in range(20)]
            codes = [f.result().status_code for f in futures]
    worst = worst_case_usd(pol, "claude-sonnet-5-5", body, 1000)
    assert codes.count(200) == int(0.05 // worst) and codes.count(429) == 20 - codes.count(200)


# --------------------------------------------------------------------------- Postgres: every replica


@pytest.fixture
def pool() -> Any:
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    with ConnectionPool(
        DB, open=True, max_size=20, kwargs={"autocommit": True, "row_factory": dict_row}
    ) as p:
        yield p


@needs_db
def test_two_replicas_share_one_cap(pool: Any) -> None:
    a, b = PostgresBudget(pool), PostgresBudget(pool)
    caller = f"agent-{uuid.uuid4()}"

    def take(budget: PostgresBudget) -> bool:
        try:
            budget.reserve(caller, 1.0, 0.1)
            return True
        except OverBudget:
            return False

    with ThreadPoolExecutor(20) as ex:
        results = list(ex.map(take, [a, b] * 15))
    assert results.count(True) == 10  # 30 tries at 0.1 against 1.0, across both replicas


@needs_db
def test_spend_survives_a_restart_and_settles_and_releases(pool: Any) -> None:
    caller = f"agent-{uuid.uuid4()}"
    first = PostgresBudget(pool)
    r = first.reserve(caller, 1.0, 0.5)
    first.settle(r, 0.25)
    first.release(first.reserve(caller, 1.0, 0.7))
    again = PostgresBudget(pool)  # a new replica, or the same one restarted
    assert again.spent_today(caller) == pytest.approx(0.25)
    with pytest.raises(OverBudget):
        again.reserve(caller, 1.0, 0.8)
    again.reserve(caller, 1.0, 0.75)


@needs_db
def test_sessions_are_shared_by_replicas(pool: Any) -> None:
    budget = PostgresBudget(pool)
    one, two = PostgresSessionStore(pool), PostgresSessionStore(pool)
    caller = f"tool-{uuid.uuid4()}"
    s, token = one.mint(
        caller,
        budget_usd=0.5,
        ttl_s=60,
        models=["gpt-*"],
        labels={"run": "r1"},
        data_class="general",
        max_open=1,
    )
    with pytest.raises(SessionRefused):
        two.mint(caller, budget_usd=0.5, ttl_s=60, models=["*"], labels={}, data_class="general", max_open=1)
    found = two.find(token)
    assert (
        found is not None and found.id == s.id and found.labels == {"run": "r1"} and two.open_for(caller) == 1
    )
    r = budget.reserve(caller, 100.0, 0.4, found)
    with pytest.raises(OverBudget, match="this session's"):
        budget.reserve(caller, 100.0, 0.2, found)
    budget.settle(r, 0.1)
    seen = one.get(s.id, caller)
    assert seen is not None and seen.calls == 1 and seen.spent_usd == pytest.approx(0.1)
    two.spend(s.id, 0.05)
    assert one.revoke(s.id, caller) is not None and two.find(token) is None
    assert one.get(s.id, "someone-else") is None and one.revoke(s.id, "someone-else") is None


@needs_db
def test_the_app_uses_the_database_when_told(make_client: Any, upstream: Any, monkeypatch: Any) -> None:
    monkeypatch.setenv("T_ANTHROPIC", "sk-real")
    app = create_app(policy(), transport=httpx.MockTransport(upstream), database_url=DB)
    with TestClient(app) as c:
        assert c.get("/readyz").json()["state"] == "postgres"
        assert call(c, key=KEY).status_code == 200
