"""Claude on Vertex AI (agentic-sdlc ADR 0010; tracker CP-112): for an installation on Google
Cloud, model calls stay in the customer's own project, under its Google Cloud terms.

Callers still speak the Messages API. Vertex AI takes the same request with two changes: the model
goes in the URL (`publishers/anthropic/models/<model>:rawPredict`, or `:streamRawPredict` to
stream), and the body names `anthropic_version: vertex-2023-10-16` in place of the header.
`anthropic-beta` passes through. Credentials are the pod's own (GKE Workload Identity, role
aiplatform.user): an access token from Application Default Credentials, refreshed before it ends.

A model's Vertex AI name may carry a version (`claude-sonnet-4-5@20250929`): the provider's
`model_versions` maps the name callers use to it."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import httpx

from ..policy import Provider
from .anthropic import AnthropicAdapter

VERSION = "vertex-2023-10-16"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def adc_token() -> tuple[str, float]:
    """(access token, expiry as a UNIX time) from Application Default Credentials."""
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(scopes=SCOPES)
    credentials.refresh(google.auth.transport.requests.Request())  # type: ignore[no-untyped-call]
    expiry = credentials.expiry.timestamp() if credentials.expiry else time.time() + 300
    return str(credentials.token), expiry


# One token for the process, shared by every adapter (each request makes a new adapter).
_cache: dict[str, Any] = {"token": "", "until": 0.0, "lock": None}


class VertexAdapter(AnthropicAdapter):
    token_source: Callable[[], tuple[str, float]] = staticmethod(adc_token)

    def __init__(self, provider: Provider, client: httpx.AsyncClient) -> None:
        super().__init__(provider, client)
        self.token = ""

    async def _ready(self) -> None:
        if _cache["lock"] is None:
            _cache["lock"] = asyncio.Lock()
        async with _cache["lock"]:
            if not _cache["token"] or _cache["until"] - time.time() < 120:
                _cache["token"], _cache["until"] = await asyncio.to_thread(type(self).token_source)
            self.token = _cache["token"]

    def _model_url(self, model: str, verb: str) -> str:
        p = self.provider
        name = p.model_versions.get(model, model)
        where = f"projects/{p.project}/locations/{p.region}/publishers/anthropic/models"
        return f"{p.base_url}/v1/{where}/{name}:{verb}"

    def _target(self, op: str, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if op == "count_tokens":  # Vertex AI counts with the model in the body
            model = str(body.get("model", ""))
            out = {
                **body,
                "model": self.provider.model_versions.get(model, model),
                "anthropic_version": VERSION,
            }
            return self._model_url("count-tokens", "rawPredict"), out
        out = {k: v for k, v in body.items() if k != "model"}
        out["anthropic_version"] = VERSION
        verb = "streamRawPredict" if op == "stream" else "rawPredict"
        return self._model_url(str(body.get("model", "")), verb), out

    def _headers(self, incoming: dict[str, str]) -> dict[str, str]:
        out = {"authorization": f"Bearer {self.token}", "content-type": "application/json"}
        if incoming.get("anthropic-beta"):
            out["anthropic-beta"] = incoming["anthropic-beta"]
        return out
