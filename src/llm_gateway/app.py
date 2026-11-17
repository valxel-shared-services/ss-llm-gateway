"""The gateway's HTTP API. Callers use the Anthropic SDK with base_url pointed here and their
gateway key as the API key; nothing else changes in agent code.

  POST /v1/messages     Anthropic Messages API (streaming included for Anthropic providers)
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
from .providers.openai import OpenAIAdapter
from .redaction import Findings, scrub_fields, scrub_request

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
        yield
        await state["client"].aclose()

    app = FastAPI(title="Valxel LLM gateway", lifespan=lifespan, docs_url=None, redoc_url=None)
    telemetry.setup("llm-gateway", app=app)

    def audited(caller: str, model: str, outcome: Outcome, reason: str = "", **details: Any) -> None:
        state["audit"].record(
            "llm.request",
            actor=Actor(id=caller, type="agent" if caller != "unknown" else "service"),
            resource=("model", [model] if model else []),
            outcome=outcome,
            reason=reason,
            details={k: v for k, v in details.items() if v},
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

    async def admit(
        request: Request, operation: Operation, texts: tuple[str, ...] = ()
    ) -> Admitted | Response:
        """The rules every route applies, in order, then the audit record. Returns the request to
        send, or the refusal to return. `texts` are the fields that embeddings and rerank send."""
        pol: Policy = state["policy"]
        ledger: Ledger = state["ledger"]
        key = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix(
            "Bearer "
        )
        found = pol.caller_for(key) if key else None
        if found is None:
            REQUESTS.labels(pol.product, "unknown", "unauthenticated").inc()
            with contextlib.suppress(AuditUnavailable):  # refused either way
                audited("unknown", "", "denied", "invalid gateway key", operation=operation)
            return error(401, "authentication_error", "invalid gateway key")
        caller_name, caller = found
        try:
            body = await request.json()
        except ValueError:
            return error(400, "invalid_request_error", "body must be JSON")
        if not isinstance(body, dict):
            return error(400, "invalid_request_error", "body must be a JSON object")
        model = str(body.get("model", ""))
        labels = labels_of(request.headers)

        def refuse(status: int, kind: str, message: str, metric: str, **details: Any) -> Response:
            if metric:
                REQUESTS.labels(pol.product, caller_name, metric).inc()
            with contextlib.suppress(AuditUnavailable):  # refused either way
                audited(caller_name, model, "denied", message, labels=labels, operation=operation, **details)
            return error(status, kind, message)

        if not caller.may_use(model):
            return refuse(403, "permission_error", f"{caller_name} may not use {model}", "model_denied")
        route = pol.provider_for(model, operation)
        if route is None:
            what = "" if operation == "messages" else f" for {operation}"
            return refuse(
                403,
                "permission_error",
                f"no provider approved for {pol.data_class} data serves {model}{what}",
                "no_provider",
            )
        provider_name, _ = route
        if operation == "messages" and int(body.get("max_tokens") or 0) > caller.max_tokens:
            return refuse(
                400,
                "invalid_request_error",
                f"max_tokens is capped at {caller.max_tokens} for {caller_name}",
                "",
            )
        if operation != "messages":
            problem = check_texts(body, operation, caller.max_inputs)
            if problem:
                return refuse(400, "invalid_request_error", problem, "")
        if ledger.spent_today(caller_name) >= caller.budget_usd_per_day:
            return refuse(429, "rate_limit_error", f"{caller_name} has used its daily budget", "over_budget")

        rule = pol.rule()
        redacted: dict[str, int] = {}
        if rule.redaction != "off":
            findings: Findings
            if operation == "messages":
                scrubbed, findings = scrub_request(body, rule.detectors)
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
            )
        except AuditUnavailable:
            REQUESTS.labels(pol.product, caller_name, "audit_unavailable").inc()
            return error(503, "api_error", "the audit trail is unavailable; try again shortly")
        return Admitted(caller_name, caller, body, model, provider_name, labels, request_id, redacted)

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

            if not isinstance(ad, AnthropicAdapter):
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
