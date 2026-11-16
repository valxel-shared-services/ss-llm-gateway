"""Embeddings and reranking, for knowledge (RAG) components.

  Embeddings: the OpenAI shape (POST /v1/embeddings), which OpenAI, Azure OpenAI, vLLM, Hugging Face
  TEI and Voyage all serve. The answer is passed back as the provider gave it.
  Rerank: the Cohere shape (POST /v1/rerank: model, query, documents, top_n), which Cohere, vLLM,
  TEI and Voyage serve with small differences. The answer is normalized to
  {"results": [{"index", "relevance_score"}], ...}, best first, so callers see one shape.

Usage is mapped to `input_tokens` for the ledger. Documents are never echoed back.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..policy import Provider
from . import Result, auth

EMBEDDING_FIELDS = ("model", "input", "dimensions", "encoding_format")


def _tokens(usage: dict[str, Any]) -> int:
    return int(usage.get("prompt_tokens") or usage.get("total_tokens") or usage.get("input_tokens") or 0)


def _upstream_error(res: httpx.Response) -> Result:
    try:
        data = res.json()
    except ValueError:
        data = {}
    err = data.get("error") if isinstance(data, dict) else None
    message = (
        (err.get("message") if isinstance(err, dict) else err) or data.get("message") or "upstream error"
    )
    return Result(
        status=res.status_code,
        usage={},
        body={"type": "error", "error": {"type": "api_error", "message": str(message)[:500]}},
    )


async def embed(provider: Provider, client: httpx.AsyncClient, body: dict[str, Any]) -> Result:
    payload = {k: body[k] for k in EMBEDDING_FIELDS if k in body}
    res = await client.post(
        provider.base_url.rstrip("/") + provider.embeddings_path,
        json=payload,
        headers=auth(provider),
        timeout=120,
    )
    if res.status_code >= 400:
        return _upstream_error(res)
    data: dict[str, Any] = res.json()
    usage = {"input_tokens": _tokens(data.get("usage") or {})}
    data["model"] = body["model"]
    data["usage"] = {"prompt_tokens": usage["input_tokens"], "total_tokens": usage["input_tokens"]}
    return Result(status=200, body=data, usage=usage)


def _document(d: Any) -> str:
    return str(d.get("text", "")) if isinstance(d, dict) else str(d)


async def rerank(provider: Provider, client: httpx.AsyncClient, body: dict[str, Any]) -> Result:
    documents = [_document(d) for d in body.get("documents") or []]
    payload: dict[str, Any] = {
        "model": body["model"],
        "query": str(body.get("query", "")),
        "documents": documents,
    }
    if body.get("top_n") is not None:
        payload["top_n"] = int(body["top_n"])
    res = await client.post(
        provider.base_url.rstrip("/") + provider.rerank_path, json=payload, headers=auth(provider), timeout=60
    )
    if res.status_code >= 400:
        return _upstream_error(res)
    data: dict[str, Any] = res.json()
    rows = data.get("results") if "results" in data else data.get("data")  # Voyage answers in `data`
    results = sorted(
        (
            {
                "index": int(r["index"]),
                "relevance_score": float(r.get("relevance_score", r.get("score", 0.0))),
            }
            for r in rows or []
            if 0 <= int(r["index"]) < len(documents)
        ),
        key=lambda r: r["relevance_score"],
        reverse=True,
    )
    if payload.get("top_n") is not None:
        results = results[: payload["top_n"]]
    tokens = _tokens(data.get("usage") or {}) or _tokens((data.get("meta") or {}).get("billed_units") or {})
    out = {
        "id": str(data.get("id", "")),
        "model": body["model"],
        "results": results,
        "usage": {"input_tokens": tokens},
    }
    return Result(status=200, body=out, usage={"input_tokens": tokens})
