"""Usage and cost. Every call is logged as one JSON line (who, which model, tokens, cost, cache use)
and counted in Prometheus, so each product sees its own spend by agent and model. Budgets, and the
spend they're held to, are budget.py's."""

from __future__ import annotations

import json
import logging
from typing import Any

from prometheus_client import Counter

from .policy import Policy

log = logging.getLogger("llm_gateway.usage")

TOKENS = Counter("llm_gateway_tokens_total", "Tokens by kind", ["product", "caller", "model", "kind"])
COST = Counter("llm_gateway_cost_usd_total", "Estimated cost in US dollars", ["product", "caller", "model"])
REQUESTS = Counter("llm_gateway_requests_total", "Requests by outcome", ["product", "caller", "outcome"])
REDACTIONS = Counter(
    "llm_gateway_redactions_total", "Redacted spans by detector", ["product", "caller", "detector"]
)


def cost_usd(policy: Policy, model: str, usage: dict[str, Any]) -> float:
    price = policy.price(model)
    if price is None:
        return 0.0
    cache_write = price.cache_write if price.cache_write is not None else price.input * 1.25
    cache_read = price.cache_read if price.cache_read is not None else price.input * 0.1
    tokens = {
        k: int(usage.get(k) or 0)
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    }
    return (
        tokens["input_tokens"] * price.input
        + tokens["output_tokens"] * price.output
        + tokens["cache_creation_input_tokens"] * cache_write
        + tokens["cache_read_input_tokens"] * cache_read
    ) / 1_000_000 + price.per_request


class Ledger:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def record(
        self,
        *,
        caller: str,
        model: str,
        provider: str,
        usage: dict[str, Any],
        request_id: str,
        labels: dict[str, str],
    ) -> float:
        cost = cost_usd(self.policy, model, usage)
        product = self.policy.product
        for kind in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            TOKENS.labels(product, caller, model, kind).inc(usage.get(kind) or 0)
        COST.labels(product, caller, model).inc(cost)
        log.info(
            json.dumps(
                {
                    "event": "llm_usage",
                    "product": product,
                    "caller": caller,
                    "provider": provider,
                    "model": model,
                    "request_id": request_id,
                    "cost_usd": round(cost, 6),
                    **{
                        k: usage.get(k) or 0
                        for k in (
                            "input_tokens",
                            "output_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        )
                    },
                    **labels,
                }
            )
        )
        return cost
