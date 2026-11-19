"""The gateway's HTTP API. Callers use the Anthropic SDK with base_url pointed here and their
gateway key as the API key; nothing else changes in agent code.

  POST /v1/messages     Anthropic Messages API, streamed or not (OpenAI-compatible providers translated)
  POST /v1/messages/count_tokens   Anthropic token counting (Anthropic providers)
  POST /v1/sessions     mint a session token (callers with a `sessions` grant; sessions.py)
  GET  /v1/sessions/{id}, DELETE /v1/sessions/{id}   read or revoke one, with what it spent
  POST /v1/responses    OpenAI Responses API, passed through (OpenAI-type providers; Codex), streamed or not
  POST /v1/embeddings   OpenAI-style embeddings, for knowledge components
  POST /v1/rerank       Cohere-style reranking, for knowledge components
  GET  /healthz         liveness;  GET /readyz  readiness (policy loaded, provider keys present)
  GET  /metrics         Prometheus

All three routes apply the same rules: the caller's key and model list, the data class's provider
rule (PHI only to a provider under a BAA or a self-hosted model), redaction, the daily budget, the
audit record before anything is sent, and the usage ledger.

Every request is audited (audit.py) before it goes to a provider: which caller, which model and
provider, which kinds of data were redacted or blocked, and requests that were refused. Spend is
in the usage ledger. With GATEWAY_AUDIT_REQUIRED (the default) a request whose audit record
can't be published is refused, so nothing reaches a model unrecorded. Requests are traced with
OpenTelemetry (telemetry.py); the call to the provider is a child span.

Audit settings come from the environment: GATEWAY_AUDIT_SINK (kafka | log),
GATEWAY_AUDIT_KAFKA_BOOTSTRAP, GATEWAY_AUDIT_TOPIC (default audit.v1), GATEWAY_AUDIT_KAFKA_CONFIG
(JSON of extra client settings), GATEWAY_AUDIT_REQUIRED (default true), GATEWAY_ENVIRONMENT.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import telemetry
from .audit import Actor, Auditor, AuditUnavailable, Outcome, make_sink
from .ledger import REDACTIONS, REQUESTS, Ledger
from .policy import Caller, Operation, Policy, load_policy
from .providers import Adapter, Result, retrieval
from .providers.anthropic import AnthropicAdapter
from .providers.openai import OpenAIAdapter, ResponsesAdapter
from .redaction import Findings, scrub_fields, scrub_request, scrub_responses
from .sessions import TOKEN_PREFIX, Session, SessionRefused, SessionStore

# Callers label requests with x-valxel-* headers (x-vrixel-* from before the rename still works).
LABEL_PREFIXES = ("x-valxel-", "x-vrixel-")
SPAN_LABELS = ("agent", "variant", "run", "thread", "feature", "tenant", "component", "purpose")


def error(status: int, kind: str, message: str) -> JSONResponse:
    """Errors in the Anthropic API's shape, so SDKs raise the usual exception types."""
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


def make_auditor(pol: Policy, env: Any = os.environ) -> Auditor:
    sink = make_sink(
        env.get("GATEWAY_AUDIT_SINK", "log"),
        bootstrap=env.get("GATEWAY_AUDIT_KAFKA_BOOTSTRAP", ""),
        topic=env.get("GATEWAY_AUDIT_TOPIC", "audit.v1"),
        config=json.loads(env.get("GATEWAY_AUDIT_KAFKA_CONFIG") or "{}"),
    )
    return Auditor(
        sink,
        service="llm-gateway",
        product=pol.product,
        environment=env.get("GATEWAY_ENVIRONMENT", ""),
        data_class=pol.data_class,
        required=env.get("GATEWAY_AUDIT_REQUIRED", "true").lower() != "false",
    )


def labels_of(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers.items():
        for prefix in LABEL_PREFIXES:
            if k.lower().startswith(prefix):
                out.setdefault(k.lower()[len(prefix) :], v)
    return out


@dataclass
class Admitted:
    """A request that passed the gateway's rules and was audited: safe to send."""

    caller_name: str
    caller: Caller
    body: dict[str, Any]
    model: str
    provider_name: str
    labels: dict[str, str]
    request_id: str
    redacted: dict[str, int] = field(default_factory=dict)
    session_id: str = ""


def annotate(
    span: Any,
    *,
    caller: str,
    model: str,
    provider: str,
    usage: dict[str, Any],
    cost: float,
    labels: dict[str, str],
    operation: str = "chat",
) -> None:
    """The model call on the request's span, in OpenTelemetry's GenAI terms: which model and
    provider, the tokens and the cost, and the caller's labels (agent, variant, run). Never the
    prompt or the reply: the gateway's traces are structure only, whatever the data class."""
    if span is None or not span.is_recording():
        return
    span.set_attribute("gen_ai.operation.name", operation)
    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", model)
    for key, attr in (
        ("input_tokens", "gen_ai.usage.input_tokens"),
        ("output_tokens", "gen_ai.usage.output_tokens"),
        ("cache_read_input_tokens", "gen_ai.usage.cache_read_input_tokens"),
        ("cache_creation_input_tokens", "gen_ai.usage.cache_creation_input_tokens"),
    ):
        if usage.get(key) is not None:
            span.set_attribute(attr, int(usage.get(key) or 0))
    span.set_attribute("valxel.cost_usd", round(float(cost), 6))
    span.set_attribute("valxel.caller", caller)
    for k, v in labels.items():
        if k in SPAN_LABELS:
            span.set_attribute(f"valxel.{k}", str(v)[:200])


def check_texts(body: dict[str, Any], operation: str, max_inputs: int) -> str:
    """What's wrong with an embeddings or rerank request's texts, or ''."""
    if operation == "embeddings":
        texts = body.get("input")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not texts or not all(isinstance(t, str) for t in texts):
            return "input must be a string or a non-empty list of strings"
        n = len(texts)
    else:
        if not isinstance(body.get("query"), str) or not body["query"]:
            return "query must be a non-empty string"
        docs = body.get("documents")
        if not isinstance(docs, list) or not docs:
            return "documents must be a non-empty list"
        n = len(docs)
    if n > max_inputs:
        return f"{n} texts in one call; the cap is {max_inputs}"
    return ""


def create_app(
    policy: Policy | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    auditor: Auditor | None = None,
) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["policy"] = policy or load_policy(
            os.environ.get("GATEWAY_POLICY", "/etc/llm-gateway/policy.yaml")
        )
        state["client"] = httpx.AsyncClient(transport=transport)
        state["ledger"] = Ledger(state["policy"])
        state["audit"] = auditor or make_auditor(state["policy"])
        state["sessions"] = SessionStore()
        yield
        await state["client"].aclose()

    app = FastAPI(title="Valxel LLM gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
    telemetry.setup("llm-gateway", app=app)

    def audited(
        caller: str,
        model: str,
        outcome: Outcome,
        reason: str = "",
        action: str = "llm.request",
        data_class: str = "",
        **details: Any,
    ) -> None:
        state["audit"].record(
            action,
            actor=Actor(id=caller, type="agent" if caller != "unknown" else "service"),
            resource=("model", [model] if model else []),
            outcome=outcome,
            reason=reason,
            details={k: v for k, v in details.items() if v},
            data_class=data_class,
        )

    def adapter(provider_name: str) -> Adapter:
        p = state["policy"].providers[provider_name]
        cls = AnthropicAdapter if p.type == "anthropic" else OpenAIAdapter
        return cls(p, state["client"])

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> Response:
        pol: Policy = state["policy"]
        missing = [n for n, p in pol.providers.items() if p.missing_key()]
        if missing:
            return JSONResponse({"status": "not ready", "missing_keys": missing}, status_code=503)
        return JSONResponse({"status": "ready", "product": pol.product, "data_class": pol.data_class})

    @app.get("/metrics")
    def metrics() -> Response:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    def key_of(request: Request) -> str:
        return request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix(
            "Bearer "
        )

    def caller_of(request: Request) -> tuple[str, Caller, Session | None] | None:
        """Who is calling: a caller by its own key, or by a live session token minted for it."""
        pol: Policy = state["policy"]
        key = key_of(request)
        if not key:
            return None
        if key.startswith(TOKEN_PREFIX):
            sess = state["sessions"].find(key)
            caller = pol.callers.get(sess.caller) if sess is not None else None
            return (sess.caller, caller, sess) if sess is not None and caller is not None else None
        found = pol.caller_for(key)
        return (found[0], found[1], None) if found else None

    async def admit(
        request: Request, operation: Operation, texts: tuple[str, ...] = ()
    ) -> Admitted | Response:
        """The rules every route applies, in order, then the audit record. Returns the request to
        send, or the refusal to return. `texts` are the fields that embeddings and rerank send."""
        pol: Policy = state["policy"]
        ledger: Ledger = state["ledger"]
        found = caller_of(request)
        if found is None:
            REQUESTS.labels(pol.product, "unknown", "unauthenticated").inc()
            with contextlib.suppress(AuditUnavailable):  # refused either way
                audited("unknown", "", "denied", "invalid gateway key", operation=operation)
            return error(
                401, "authentication_error", "invalid gateway key, or a session token that has ended"
            )
        caller_name, caller, sess = found
        try:
            body = await request.json()
        except ValueError:
            return error(400, "invalid_request_error", "body must be JSON")
        if not isinstance(body, dict):
            return error(400, "invalid_request_error", "body must be a JSON object")
        model = str(body.get("model", ""))
        labels = labels_of(request.headers)
        if sess is not None:  # a session's labels are the ones it was minted with, whatever it sends
            labels = {**labels, **sess.labels, "session": sess.id}
        # The data class it's held to: a session's own; else what a caller that may declare one did.
        declared = (
            sess.data_class if sess is not None else request.headers.get("x-valxel-data-class", "").strip()
        )
        data_class = pol.class_for(declared, caller) or ""
        labels.pop("data-class", None)  # recorded as the audit record's data class, not a label

        def refuse(status: int, kind: str, message: str, metric: str, **details: Any) -> Response:
            if metric:
                REQUESTS.labels(pol.product, caller_name, metric).inc()
            with contextlib.suppress(AuditUnavailable):  # refused either way
                audited(
                    caller_name,
                    model,
                    "denied",
                    message,
                    labels=labels,
                    operation=operation,
                    data_class=data_class or declared,
                    **details,
                )
            return error(status, kind, message)

        if not data_class:
            return refuse(
                403,
                "permission_error",
                f"{caller_name} may not send {declared} data through this gateway",
                "data_class_denied",
            )

        if not caller.may_use(model) or (sess is not None and not sess.may_use(model)):
            who = f"this session of {caller_name}" if sess is not None else caller_name
            return refuse(403, "permission_error", f"{who} may not use {model}", "model_denied")
        route = pol.provider_for(model, operation, data_class)
        if route is None:
            what = "" if operation == "messages" else f" for {operation}"
            return refuse(
                403,
                "permission_error",
                f"no provider approved for {data_class} data serves {model}{what}",
                "no_provider",
            )
        provider_name, _ = route
        cap_field = {"messages": "max_tokens", "responses": "max_output_tokens"}.get(operation)
        if cap_field and int(body.get(cap_field) or 0) > caller.max_tokens:
            return refuse(
                400,
                "invalid_request_error",
                f"max_tokens is capped at {caller.max_tokens} for {caller_name}",
                "",
            )
        if operation in ("embeddings", "rerank"):
            problem = check_texts(body, operation, caller.max_inputs)
            if problem:
                return refuse(400, "invalid_request_error", problem, "")
        if ledger.spent_today(caller_name) >= caller.budget_usd_per_day:
            return refuse(429, "rate_limit_error", f"{caller_name} has used its daily budget", "over_budget")
        if sess is not None and pol.price(model) is None:  # unpriced, a session's budget couldn't hold it
            return refuse(
                403,
                "permission_error",
                f"{model} has no price in this gateway's policy: sessions use priced models",
                "unpriced",
            )
        if sess is not None and sess.spent_usd >= sess.budget_usd:
            return refuse(429, "rate_limit_error", "this session has used its budget", "session_over_budget")

        rule = pol.rule(data_class)
        redacted: dict[str, int] = {}
        if rule.redaction != "off":
            findings: Findings
            if operation == "messages":
                scrubbed, findings = scrub_request(body, rule.detectors)
            elif operation == "responses":
                scrubbed, findings = scrub_responses(body, rule.detectors)
            else:
                scrubbed, findings = scrub_fields(body, texts, rule.detectors)
            for kind, n in findings.counts.items():
                REDACTIONS.labels(pol.product, caller_name, kind).inc(n)
            if findings.total and rule.redaction == "block":
                return refuse(
                    400,
                    "invalid_request_error",
                    "request contains data this product may not send to a model: "
                    + ", ".join(sorted(findings.counts)),
                    "blocked",
                    blocked=dict(findings.counts),
                )
            redacted = dict(findings.counts)
            body = scrubbed

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        # Nothing reaches a model unrecorded.
        try:
            audited(
                caller_name,
                model,
                "success",
                provider=provider_name,
                redacted=redacted,
                request_id=request_id,
                labels=labels,
                operation=operation,
                stream=bool(body.get("stream")),
                data_class=data_class,
            )
        except AuditUnavailable:
            REQUESTS.labels(pol.product, caller_name, "audit_unavailable").inc()
            return error(503, "api_error", "the audit trail is unavailable; try again shortly")
        return Admitted(
            caller_name,
            caller,
            body,
            model,
            provider_name,
            labels,
            request_id,
            redacted,
            session_id=sess.id if sess is not None else "",
        )

    def charge(a: Admitted, cost: float) -> None:
        if a.session_id:
            state["sessions"].spend(a.session_id, cost)

    def settle(a: Admitted, res: Result, operation: str) -> Response:
        """Record usage and cost for an answered call and return the provider's answer."""
        pol: Policy = state["policy"]
        if res.status < 400:
            cost = state["ledger"].record(
                caller=a.caller_name,
                model=a.model,
                provider=a.provider_name,
                usage=res.usage,
                request_id=a.request_id,
                labels=a.labels,
            )
            charge(a, cost)
            annotate(
                trace.get_current_span(),
                caller=a.caller_name,
                model=a.model,
                provider=a.provider_name,
                usage=res.usage,
                cost=cost,
                labels=a.labels,
                operation=operation,
            )
        REQUESTS.labels(
            pol.product, a.caller_name, "ok" if res.status < 400 else f"upstream_{res.status}"
        ).inc()
        return JSONResponse(res.body, status_code=res.status, headers={"request-id": a.request_id})

    def unreachable(a: Admitted, exc: Exception) -> Response:
        REQUESTS.labels(state["policy"].product, a.caller_name, "upstream_error").inc()
        return error(502, "api_error", f"provider unreachable: {type(exc).__name__}")

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> Response:
        a = await admit(request, "messages")
        if not isinstance(a, Admitted):
            return a
        ad = adapter(a.provider_name)
        if not isinstance(ad, AnthropicAdapter):
            return error(
                400, "invalid_request_error", f"token counting is not available for {a.provider_name}"
            )
        try:
            res = await ad.count_tokens(a.body, dict(request.headers.items()))
        except httpx.HTTPError as exc:
            return unreachable(a, exc)
        return JSONResponse(res.body, status_code=res.status, headers={"request-id": a.request_id})

    # ------------------------------------------------------------------ session tokens

    def minter(request: Request) -> tuple[str, Caller] | Response:
        """The caller minting or managing sessions: by its own key, never by a session token."""
        found = caller_of(request)
        if found is None or found[2] is not None:
            return error(
                401, "authentication_error", "sessions are managed with the caller's own gateway key"
            )
        name, caller, _ = found
        if caller.sessions is None:
            return error(403, "permission_error", f"{name} may not mint session tokens")
        return name, caller

    @app.post("/v1/sessions")
    async def mint_session(request: Request) -> Response:
        pol: Policy = state["policy"]
        who = minter(request)
        if isinstance(who, Response):
            return who
        name, caller = who
        grant = caller.sessions
        assert grant is not None
        try:
            body = await request.json()
        except ValueError:
            return error(400, "invalid_request_error", "body must be JSON")
        if not isinstance(body, dict):
            return error(400, "invalid_request_error", "body must be a JSON object")
        try:
            budget = float(body.get("budget_usd", 0))
            ttl = int(body.get("ttl_s", 0))
        except (TypeError, ValueError):
            return error(400, "invalid_request_error", "budget_usd and ttl_s must be numbers")
        data_class = str(body.get("data_class", ""))
        models = body.get("models") or list(caller.models)
        labels = body.get("labels") or {}
        problem = ""
        if not 0 < budget <= grant.max_budget_usd:
            problem = f"budget_usd must be more than 0 and at most {grant.max_budget_usd}"
        elif not 0 < ttl <= grant.max_ttl_s:
            problem = f"ttl_s must be more than 0 and at most {grant.max_ttl_s}"
        elif not isinstance(models, list) or not all(isinstance(m, str) and m for m in models):
            problem = "models must be a list of model names"
        elif not all(caller.may_use(m) for m in models):
            problem = f"{name} may not use every model asked for"
        elif not isinstance(labels, dict) or not all(isinstance(v, str) for v in labels.values()):
            problem = "labels must map names to strings"
        elif not pol.serves_data_class(data_class, caller):
            return error(
                403,
                "permission_error",
                f"this gateway serves {pol.data_class} data; not a {data_class or '?'} session",
            )
        if problem:
            return error(400, "invalid_request_error", problem)
        labels = {str(k).lower(): str(v)[:200] for k, v in labels.items()}
        try:
            sess, token = state["sessions"].mint(
                name,
                budget_usd=budget,
                ttl_s=ttl,
                models=models,
                labels=labels,
                data_class=data_class,
                max_open=grant.max_open,
            )
        except SessionRefused as exc:
            return error(429, "rate_limit_error", str(exc))
        try:
            audited(
                name,
                "",
                "success",
                action="llm.session.minted",
                session=sess.id,
                budget_usd=budget,
                ttl_s=ttl,
                models=models,
                labels=labels,
                data_class=data_class,
            )
        except AuditUnavailable:
            state["sessions"].revoke(sess.id, name)
            return error(503, "api_error", "the audit trail is unavailable; try again shortly")
        return JSONResponse({**sess.view(), "token": token}, status_code=201)

    @app.get("/v1/sessions/{session_id}")
    def read_session(session_id: str, request: Request) -> Response:
        who = minter(request)
        if isinstance(who, Response):
            return who
        sess = state["sessions"].get(session_id, who[0])
        if sess is None:
            return error(404, "not_found_error", "no such session")
        return JSONResponse(sess.view())

    @app.delete("/v1/sessions/{session_id}")
    def revoke_session(session_id: str, request: Request) -> Response:
        who = minter(request)
        if isinstance(who, Response):
            return who
        sess = state["sessions"].revoke(session_id, who[0])
        if sess is None:
            return error(404, "not_found_error", "no such session")
        with contextlib.suppress(AuditUnavailable):  # the token is dead either way
            audited(
                who[0],
                "",
                "success",
                action="llm.session.revoked",
                session=sess.id,
                spent_usd=round(sess.spent_usd, 6),
                calls=sess.calls,
            )
        return JSONResponse(sess.view())

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        a = await admit(request, "embeddings", ("input",))
        if not isinstance(a, Admitted):
            return a
        try:
            res = await retrieval.embed(state["policy"].providers[a.provider_name], state["client"], a.body)
        except httpx.HTTPError as exc:
            return unreachable(a, exc)
        return settle(a, res, "embeddings")

    @app.post("/v1/rerank")
    async def rerank(request: Request) -> Response:
        a = await admit(request, "rerank", ("query", "documents"))
        if not isinstance(a, Admitted):
            return a
        try:
            res = await retrieval.rerank(state["policy"].providers[a.provider_name], state["client"], a.body)
        except httpx.HTTPError as exc:
            return unreachable(a, exc)
        return settle(a, res, "rerank")

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        """OpenAI's Responses API (Codex), passed through to an OpenAI-type provider under the same
        rules as messages: the caller's key or session token, its models, the data class's provider
        rule and redaction, the budgets, the audit record and the ledger."""
        a = await admit(request, "responses")
        if not isinstance(a, Admitted):
            return a
        p = state["policy"].providers[a.provider_name]
        if p.type != "openai":
            return error(
                400, "invalid_request_error", f"the Responses API is not available for {a.provider_name}"
            )
        ad = ResponsesAdapter(p, state["client"])
        if a.body.get("stream"):
            usage: dict[str, Any] = {}
            span = trace.get_current_span()

            async def relay() -> AsyncIterator[bytes]:
                try:
                    async for chunk in ad.stream(a.body, usage):
                        yield chunk
                finally:
                    cost = state["ledger"].record(
                        caller=a.caller_name,
                        model=a.model,
                        provider=a.provider_name,
                        usage=usage,
                        request_id=a.request_id,
                        labels=a.labels,
                    )
                    charge(a, cost)
                    annotate(
                        span,
                        caller=a.caller_name,
                        model=a.model,
                        provider=a.provider_name,
                        usage=usage,
                        cost=cost,
                        labels=a.labels,
                        operation="responses",
                    )
                    REQUESTS.labels(state["policy"].product, a.caller_name, "ok").inc()

            return StreamingResponse(
                relay(), media_type="text/event-stream", headers={"request-id": a.request_id}
            )
        try:
            res = await ad.create(a.body)
        except httpx.HTTPError as exc:
            return unreachable(a, exc)
        return settle(a, res, "responses")

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        a = await admit(request, "messages")
        if not isinstance(a, Admitted):
            return a
        pol: Policy = state["policy"]
        ledger: Ledger = state["ledger"]
        body, model, caller_name, provider_name = a.body, a.model, a.caller_name, a.provider_name
        labels, request_id = a.labels, a.request_id
        headers = {k: v for k, v in request.headers.items()}
        ad = adapter(provider_name)

        if body.get("stream"):
            usage: dict[str, Any] = {}

            span = trace.get_current_span()

            async def relay() -> AsyncIterator[bytes]:
                try:
                    async for chunk in ad.stream(body, headers, usage):
                        yield chunk
                finally:
                    cost = ledger.record(
                        caller=caller_name,
                        model=model,
                        provider=provider_name,
                        usage=usage,
                        request_id=request_id,
                        labels=labels,
                    )
                    charge(a, cost)
                    annotate(
                        span,
                        caller=caller_name,
                        model=model,
                        provider=provider_name,
                        usage=usage,
                        cost=cost,
                        labels=labels,
                    )
                    REQUESTS.labels(pol.product, caller_name, "ok").inc()

            if not isinstance(ad, (AnthropicAdapter, OpenAIAdapter)):
                return error(400, "invalid_request_error", f"streaming is not available for {provider_name}")
            return StreamingResponse(
                relay(), media_type="text/event-stream", headers={"request-id": request_id}
            )

        try:
            res = await ad.create(body, headers)
        except httpx.HTTPError as exc:
            return unreachable(a, exc)
        return settle(a, res, "chat")

    return app


def main() -> None:  # pragma: no cover
    import uvicorn

    telemetry.setup_logging("llm-gateway")
    uvicorn.run(
        create_app(),
        host="0.0.0.0",  # noqa: S104 - a container listens on all interfaces
        port=int(os.environ.get("PORT", "8080")),
        proxy_headers=True,
        log_config=None,
    )
