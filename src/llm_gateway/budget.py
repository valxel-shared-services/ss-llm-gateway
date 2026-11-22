"""Budgets as a hard cap (review finding 9): a call reserves its worst-case cost before it's sent,
and settles to its real cost when it's answered.

Checking "spent so far < cap" before a call and adding the cost after lets concurrent calls, or one
large call, run past the cap. So each call first reserves what it could cost at most (its input,
estimated generously from the request's size, at the dearer of the input and cache-write prices,
plus its whole output allowance at the output price), and is refused if what's spent plus what's
reserved plus this would pass the caller's daily budget or its session's budget. When it's answered
the reservation becomes the real cost; a call that fails, or is never sent, releases it.

The estimate can still be low (a request made of very short tokens), so spend can pass a cap by
that one call's error, never by the sum of everything in flight. A reservation a crashed replica
never settled stops counting after RESERVATION_TTL_S.

MemoryBudget is one process's; PostgresBudget (GATEWAY_DATABASE_URL) is shared by every replica,
and also keeps sessions there (PostgresSessionStore), so a restart loses neither spend nor sessions.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .policy import Policy
from .sessions import Session, SessionStore

RESERVATION_TTL_S = 900  # a call's longest life; a reservation older than this was never settled
CHARS_PER_TOKEN = 2  # generous: English runs about 4 characters a token, code about 3


def day() -> str:
    return datetime.now(UTC).date().isoformat()


def worst_case_usd(policy: Policy, model: str, body: dict[str, Any], output_tokens: int) -> float:
    """The most this request could cost: its whole size as input, its whole output allowance."""
    price = policy.price(model)
    if price is None:
        return 0.0
    input_tokens = len(json.dumps(body, separators=(",", ":"), default=str)) // CHARS_PER_TOKEN + 1
    input_rate = max(price.input, price.cache_write if price.cache_write is not None else price.input * 1.25)
    return (input_tokens * input_rate + max(0, output_tokens) * price.output) / 1_000_000 + price.per_request


@dataclass
class Reservation:
    id: str
    caller: str
    day: str
    session_id: str
    amount: float


class OverBudget(Exception):
    """The call could take spend past a budget."""

    def __init__(self, whose: str, spent: float, held: float, amount: float, cap: float) -> None:
        self.whose = whose
        super().__init__(
            f"{whose} budget: ${spent:.4f} spent and ${held:.4f} held by calls in flight; this call "
            f"could cost up to ${amount:.4f}, past ${cap:.4f}"
        )


class MemoryBudget:
    def __init__(self, sessions: SessionStore, clock: Callable[[], float] = time.time) -> None:
        self.sessions, self.clock = sessions, clock
        self._spent: dict[tuple[str, str], float] = {}
        self._held: dict[str, tuple[Reservation, float]] = {}  # id -> (reservation, when)
        self._lock = threading.Lock()

    def _live(self) -> list[Reservation]:
        now = self.clock()
        for rid in [r for r, (_, at) in self._held.items() if at + RESERVATION_TTL_S < now]:
            del self._held[rid]
        return [r for r, _ in self._held.values()]

    def spent_today(self, caller: str) -> float:
        with self._lock:
            return self._spent.get((caller, day()), 0.0)

    def reserve(self, caller: str, cap: float, amount: float, session: Session | None = None) -> Reservation:
        r = Reservation(str(uuid.uuid4()), caller, day(), session.id if session else "", amount)
        with self._lock:
            live = self._live()
            spent = self._spent.get((caller, r.day), 0.0)
            held = sum(x.amount for x in live if x.caller == caller and x.day == r.day)
            if spent >= cap or spent + held + amount > cap:
                raise OverBudget(f"{caller}'s daily", spent, held, amount, cap)
            if session is not None:
                s_held = sum(x.amount for x in live if x.session_id == session.id)
                if (
                    session.spent_usd >= session.budget_usd
                    or session.spent_usd + s_held + amount > session.budget_usd
                ):
                    raise OverBudget("this session's", session.spent_usd, s_held, amount, session.budget_usd)
            self._held[r.id] = (r, self.clock())
        return r

    def settle(self, r: Reservation, cost: float, *, called: bool = True) -> None:
        """The reservation becomes the call's real cost (0 and called=False: released unused)."""
        with self._lock:
            self._held.pop(r.id, None)
            self._spent[(r.caller, r.day)] = self._spent.get((r.caller, r.day), 0.0) + cost
        if r.session_id and called:
            self.sessions.spend(r.session_id, cost)

    def release(self, r: Reservation) -> None:
        self.settle(r, 0.0, called=False)


class PostgresSessionStore(SessionStore):
    """Sessions in Postgres: every replica finds a token any of them minted, and a restart ends none."""

    def __init__(self, pool: Any, clock: Callable[[], float] = time.time) -> None:
        super().__init__(clock)
        self.pool = pool

    @staticmethod
    def _of(row: dict[str, Any]) -> Session:
        return Session(
            id=row["id"],
            caller=row["caller"],
            digest=row["digest"],
            budget_usd=float(row["budget_usd"]),
            expires_at=float(row["expires_at"]),
            models=list(row["models"]),
            labels=dict(row["labels"]),
            data_class=row["data_class"],
            spent_usd=float(row["spent_usd"]),
            calls=int(row["calls"]),
            revoked_at=None if row["revoked_at"] is None else float(row["revoked_at"]),
            created_at=float(row["created_at"]),
        )

    def open_for(self, caller: str) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "select count(*) as n from gateway_sessions where caller = %s and revoked_at is null"
                " and expires_at > %s",
                (caller, self.clock()),
            ).fetchone()
        return int(row["n"])

    def mint(
        self,
        caller: str,
        *,
        budget_usd: float,
        ttl_s: int,
        models: list[str],
        labels: dict[str, str],
        data_class: str,
        max_open: int,
    ) -> tuple[Session, str]:
        from psycopg.types.json import Jsonb

        from .sessions import KEEP_AFTER_S, SessionRefused, mint_token

        s, token = mint_token(self.clock, caller, budget_usd, ttl_s, models, labels, data_class)
        now = self.clock()
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("select pg_advisory_xact_lock(hashtext(%s))", (f"gateway-sessions:{caller}",))
            conn.execute(
                "delete from gateway_sessions where least(expires_at, coalesce(revoked_at, %s)) + %s < %s",
                (now, KEEP_AFTER_S, now),
            )
            n = conn.execute(
                "select count(*) as n from gateway_sessions where caller = %s and revoked_at is null"
                " and expires_at > %s",
                (caller, now),
            ).fetchone()["n"]
            if n >= max_open:
                raise SessionRefused(f"{caller} already has {max_open} open sessions")
            conn.execute(
                "insert into gateway_sessions (id, caller, digest, budget_usd, expires_at, models, labels,"
                " data_class, created_at) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    s.id,
                    caller,
                    s.digest,
                    s.budget_usd,
                    s.expires_at,
                    Jsonb(s.models),
                    Jsonb(s.labels),
                    s.data_class,
                    s.created_at,
                ),
            )
        return s, token

    def find(self, token: str) -> Session | None:
        from .sessions import digest

        with self.pool.connection() as conn:
            row = conn.execute(
                "select * from gateway_sessions where digest = %s", (digest(token),)
            ).fetchone()
        s = self._of(row) if row else None
        return s if s is not None and self._live(s) else None

    def get(self, session_id: str, caller: str) -> Session | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "select * from gateway_sessions where id = %s and caller = %s", (session_id, caller)
            ).fetchone()
        return self._of(row) if row else None

    def revoke(self, session_id: str, caller: str) -> Session | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "update gateway_sessions set revoked_at = coalesce(revoked_at, %s)"
                " where id = %s and caller = %s"
                " returning *",
                (self.clock(), session_id, caller),
            ).fetchone()
        return self._of(row) if row else None

    def spend(self, session_id: str, cost: float) -> None:  # PostgresBudget.settle does it in one go
        with self.pool.connection() as conn:
            conn.execute(
                "update gateway_sessions set spent_usd = spent_usd + %s, calls = calls + 1 where id = %s",
                (cost, session_id),
            )


class PostgresBudget:
    DDL = (
        "create table if not exists gateway_spend (caller text not null, day date not null,"
        " spent_usd double precision not null default 0, primary key (caller, day))",
        "create table if not exists gateway_reservations (id text primary key, caller text not null,"
        " day date not null, session_id text not null default '', amount double precision not null,"
        " at timestamptz not null default now())",
        "create index if not exists gateway_reservations_caller on gateway_reservations (caller, day)",
        "create index if not exists gateway_reservations_session on gateway_reservations (session_id)",
        "create table if not exists gateway_sessions (id text primary key, caller text not null,"
        " digest text not null unique, budget_usd double precision not null,"
        " expires_at double precision not null,"
        " models jsonb not null, labels jsonb not null, data_class text not null default '',"
        " spent_usd double precision not null default 0, calls integer not null default 0,"
        " revoked_at double precision, created_at double precision not null)",
        "create index if not exists gateway_sessions_caller on gateway_sessions (caller)",
    )

    def __init__(self, pool: Any) -> None:
        self.pool = pool
        with pool.connection() as conn:
            conn.execute("select pg_advisory_lock(hashtext('gateway-ddl'))")
            try:
                for statement in self.DDL:
                    conn.execute(statement)
            finally:
                conn.execute("select pg_advisory_unlock(hashtext('gateway-ddl'))")

    def spent_today(self, caller: str) -> float:
        with self.pool.connection() as conn:
            row = conn.execute(
                "select spent_usd from gateway_spend where caller = %s and day = %s", (caller, day())
            ).fetchone()
        return float(row["spent_usd"]) if row else 0.0

    def reserve(self, caller: str, cap: float, amount: float, session: Session | None = None) -> Reservation:
        r = Reservation(str(uuid.uuid4()), caller, day(), session.id if session else "", amount)
        stale = f"{RESERVATION_TTL_S} seconds"
        with self.pool.connection() as conn, conn.transaction():
            # Locks in one order everywhere: the caller's day, then the session.
            conn.execute(
                "insert into gateway_spend (caller, day) values (%s, %s) on conflict do nothing",
                (caller, r.day),
            )
            spent = float(
                conn.execute(
                    "select spent_usd from gateway_spend where caller = %s and day = %s for update",
                    (caller, r.day),
                ).fetchone()["spent_usd"]
            )
            conn.execute("delete from gateway_reservations where at < now() - %s::interval", (stale,))
            held = float(
                conn.execute(
                    "select coalesce(sum(amount), 0) as held from gateway_reservations"
                    " where caller = %s and day = %s",
                    (caller, r.day),
                ).fetchone()["held"]
            )
            if spent >= cap or spent + held + amount > cap:
                raise OverBudget(f"{caller}'s daily", spent, held, amount, cap)
            if session is not None:
                row = conn.execute(
                    "select budget_usd, spent_usd from gateway_sessions where id = %s for update",
                    (session.id,),
                ).fetchone()
                s_held = float(
                    conn.execute(
                        "select coalesce(sum(amount), 0) as held from gateway_reservations"
                        " where session_id = %s",
                        (session.id,),
                    ).fetchone()["held"]
                )
                s_cap = float(row["budget_usd"]) if row else 0.0
                s_now = float(row["spent_usd"]) if row else 0.0
                if row is None or s_now >= s_cap or s_now + s_held + amount > s_cap:
                    s_spent = float(row["spent_usd"]) if row else session.spent_usd
                    raise OverBudget("this session's", s_spent, s_held, amount, session.budget_usd)
            conn.execute(
                "insert into gateway_reservations (id, caller, day, session_id, amount)"
                " values (%s, %s, %s, %s, %s)",
                (r.id, caller, r.day, r.session_id, amount),
            )
        return r

    def settle(self, r: Reservation, cost: float, *, called: bool = True) -> None:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                "insert into gateway_spend (caller, day, spent_usd) values (%s, %s, %s)"
                " on conflict (caller, day)"
                " do update set spent_usd = gateway_spend.spent_usd + excluded.spent_usd",
                (r.caller, r.day, cost),
            )
            if r.session_id and called:
                conn.execute(
                    "update gateway_sessions set spent_usd = spent_usd + %s, calls = calls + 1 where id = %s",
                    (cost, r.session_id),
                )
            conn.execute("delete from gateway_reservations where id = %s", (r.id,))

    def release(self, r: Reservation) -> None:
        self.settle(r, 0.0, called=False)


def shared(url: str) -> tuple[PostgresBudget, PostgresSessionStore]:
    """The budget and sessions every replica shares, in the database at `url`."""
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        url, min_size=1, max_size=10, open=True, kwargs={"autocommit": True, "row_factory": dict_row}
    )
    return PostgresBudget(pool), PostgresSessionStore(pool)
