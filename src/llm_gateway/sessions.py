"""Session tokens: short-lived gateway keys a trusted caller mints for one agent session, so the
session never holds a provider key or the caller's own gateway key.

A caller whose policy entry has `sessions:` may mint them (POST /v1/sessions). A token works like
its caller's key, narrowed: it expires, it carries its own budget on top of the caller's daily one,
it may use only the models it was minted for, its labels (feature, run, agent) are fixed when it is
minted and override whatever the session sends, and it serves only data of the class it was minted
for. The caller revokes it when the session ends (DELETE /v1/sessions/{id}) and reads what it spent.

Tokens are kept as SHA-256 digests, in memory, like the ledger: run one replica, or move both to
Redis or Postgres before scaling out. A restart revokes every open session.
"""

from __future__ import annotations

import fnmatch
import hashlib
import secrets
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

TOKEN_PREFIX = "vgs_"  # noqa: S105 - a prefix that marks session tokens, not a secret
KEEP_AFTER_S = 3600  # an ended session stays readable this long, then it's dropped


class SessionRefused(ValueError):
    """The mint asks for more than the caller's grant allows."""


@dataclass
class Session:
    id: str
    caller: str
    digest: str
    budget_usd: float
    expires_at: float
    models: list[str]
    labels: dict[str, str]
    data_class: str
    spent_usd: float = 0.0
    calls: int = 0
    revoked_at: float | None = None
    created_at: float = field(default_factory=time.time)

    def may_use(self, model: str) -> bool:
        return any(fnmatch.fnmatch(model, p) for p in self.models)

    def view(self) -> dict[str, object]:
        return {
            "id": self.id,
            "budget_usd": round(self.budget_usd, 6),
            "spent_usd": round(self.spent_usd, 6),
            "calls": self.calls,
            "expires_at": int(self.expires_at),
            "models": self.models,
            "data_class": self.data_class,
            "revoked": self.revoked_at is not None,
        }


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class SessionStore:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._by_digest: dict[str, Session] = {}
        self._by_id: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _live(self, s: Session) -> bool:
        return s.revoked_at is None and self.clock() < s.expires_at

    def _sweep(self) -> None:
        now = self.clock()
        for s in [
            s for s in self._by_id.values() if min(s.expires_at, s.revoked_at or now) + KEEP_AFTER_S < now
        ]:
            self._by_id.pop(s.id, None)
            self._by_digest.pop(s.digest, None)

    def open_for(self, caller: str) -> int:
        with self._lock:
            return sum(1 for s in self._by_id.values() if s.caller == caller and self._live(s))

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
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        s = Session(
            id=str(uuid.uuid4()),
            caller=caller,
            digest=digest(token),
            budget_usd=float(budget_usd),
            expires_at=self.clock() + int(ttl_s),
            models=list(models),
            labels=dict(labels),
            data_class=data_class,
            created_at=self.clock(),
        )
        with self._lock:
            self._sweep()
            if sum(1 for o in self._by_id.values() if o.caller == caller and self._live(o)) >= max_open:
                raise SessionRefused(f"{caller} already has {max_open} open sessions")
            self._by_digest[s.digest] = s
            self._by_id[s.id] = s
        return s, token

    def find(self, token: str) -> Session | None:
        """The live session a token belongs to: None if it's unknown, revoked or expired."""
        with self._lock:
            s = self._by_digest.get(digest(token))
            return s if s is not None and self._live(s) else None

    def get(self, session_id: str, caller: str) -> Session | None:
        with self._lock:
            s = self._by_id.get(session_id)
            return s if s is not None and s.caller == caller else None

    def revoke(self, session_id: str, caller: str) -> Session | None:
        with self._lock:
            s = self._by_id.get(session_id)
            if s is None or s.caller != caller:
                return None
            if s.revoked_at is None:
                s.revoked_at = self.clock()
            return s

    def spend(self, session_id: str, cost: float) -> None:
        with self._lock:
            s = self._by_id.get(session_id)
            if s is not None:
                s.spent_usd += float(cost)
                s.calls += 1
