"""Claude in Microsoft Foundry (agentic-sdlc ADR 0010; tracker CP-409): for an installation on Azure,
model calls stay in the customer's own Foundry resource, under its Azure terms.

Foundry serves the Messages API itself at https://<resource>.services.ai.azure.com/anthropic, so
requests pass through as they do to Anthropic: prompt caching, tools, thinking, streaming and
token counting unchanged. Two things differ: the model is the deployment's name, which the
provider's `model_versions` maps from the name callers use; and the credential is an Entra ID
token for the pod's own identity (AKS Workload Identity, role Azure AI User on the resource),
from DefaultAzureCredential with the scope https://ai.azure.com/.default, refreshed before it ends.
No API key."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import httpx

from ..policy import Provider
from .anthropic import AnthropicAdapter

SCOPE = "https://ai.azure.com/.default"


def entra_token() -> tuple[str, float]:
    """(access token, expiry as a UNIX time) for the pod's own identity."""
    from azure.identity import DefaultAzureCredential

    got = DefaultAzureCredential().get_token(SCOPE)
    return str(got.token), float(got.expires_on)


# One token for the process, shared by every adapter (each request makes a new adapter).
_cache: dict[str, Any] = {"token": "", "until": 0.0, "lock": None}


class FoundryAdapter(AnthropicAdapter):
    token_source: Callable[[], tuple[str, float]] = staticmethod(entra_token)

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

    def _target(self, op: str, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        url, out = super()._target(op, body)
        model = str(body.get("model", ""))
        return url, {**out, "model": self.provider.model_versions.get(model, model)}

    def _headers(self, incoming: dict[str, str]) -> dict[str, str]:
        out = super()._headers(incoming)
        del out["x-api-key"]
        out["authorization"] = f"Bearer {self.token}"
        return out
