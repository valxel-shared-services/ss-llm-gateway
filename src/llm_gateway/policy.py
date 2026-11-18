"""Gateway policy: which providers and models a product may use, how outbound text is treated,
who may call, and what models cost. One YAML file per deployment (a ConfigMap), so each product
runs the gateway with its own data class and budgets."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

DataClass = Literal["general", "payments", "phi"]
RedactionMode = Literal["off", "redact", "block"]
Operation = Literal["messages", "embeddings", "rerank"]

# What each provider type can do unless the policy narrows it. Rerank is opt-in: only some
# OpenAI-compatible servers (vLLM, TEI, Cohere, Voyage) serve it.
DEFAULT_OPERATIONS: dict[str, list[Operation]] = {
    "anthropic": ["messages"],
    "openai": ["messages", "embeddings"],
}


class Provider(BaseModel):
    type: Literal["anthropic", "openai"]
    base_url: str
    api_key_env: str = ""  # empty for a self-hosted model that takes no key
    baa: bool = False  # a business associate agreement (or DPA) covers this provider
    self_hosted: bool = False  # runs inside the product's own network: data never leaves it
    models: list[str] = Field(default_factory=lambda: ["*"])
    operations: list[Operation] | None = None  # defaults by type (DEFAULT_OPERATIONS)
    embeddings_path: str = "/v1/embeddings"
    rerank_path: str = "/v1/rerank"

    @model_validator(mode="after")
    def _anthropic_only_messages(self) -> Provider:
        if self.type == "anthropic" and set(self.operations or []) - {"messages"}:
            raise ValueError("an anthropic provider serves messages only; embeddings and rerank need another")
        return self

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    def missing_key(self) -> bool:
        return bool(self.api_key_env) and not self.api_key()

    def does(self, operation: Operation) -> bool:
        ops = self.operations if self.operations is not None else DEFAULT_OPERATIONS[self.type]
        return operation in ops

    def serves(self, model: str) -> bool:
        return any(fnmatch.fnmatch(model, p) for p in self.models)


class DataClassRule(BaseModel):
    require_baa: bool = False
    redaction: RedactionMode = "redact"
    detectors: list[str] = Field(
        default_factory=lambda: ["secret", "card_number", "us_ssn", "email", "phone"]
    )


class SessionGrant(BaseModel):
    """What a caller may hand out as session tokens (sessions.py): the most any one may spend and
    how long it may live, and how many may be open at once."""

    max_budget_usd: float = 20.0
    max_ttl_s: int = 7200
    max_open: int = 50


class Caller(BaseModel):
    key_sha256: str
    budget_usd_per_day: float = 25.0
    models: list[str] = Field(default_factory=lambda: ["*"])
    max_tokens: int = 8192
    max_inputs: int = 256  # texts per embeddings call, documents per rerank call
    sessions: SessionGrant | None = None  # may mint session tokens (POST /v1/sessions)

    def may_use(self, model: str) -> bool:
        return any(fnmatch.fnmatch(model, p) for p in self.models)


class Price(BaseModel):
    """US dollars per million tokens."""

    input: float
    output: float = 0.0  # embeddings and rerank models have no output tokens
    per_request: float = 0.0  # a flat price per call, e.g. a rerank search
    cache_write: float | None = None  # defaults to 1.25 x input (5-minute cache writes)
    cache_read: float | None = None  # defaults to 0.1 x input


class Policy(BaseModel):
    product: str
    data_class: DataClass
    providers: dict[str, Provider]
    data_classes: dict[str, DataClassRule] = Field(default_factory=dict)
    callers: dict[str, Caller]
    pricing: dict[str, Price] = Field(default_factory=dict)  # model glob -> price

    @field_validator("callers")
    @classmethod
    def _hex_keys(cls, v: dict[str, Caller]) -> dict[str, Caller]:
        for name, c in v.items():
            if len(c.key_sha256) != 64:
                raise ValueError(f"caller {name}: key_sha256 must be a hex SHA-256 of its key")
        return v

    def serves_data_class(self, data_class: str) -> bool:
        """A session of this data class may use this gateway: its own class, or general data,
        which every gateway's rules cover. PHI never goes through a gateway set up for less."""
        return data_class in (self.data_class, "general")

    def rule(self) -> DataClassRule:
        return self.data_classes.get(self.data_class, DataClassRule())

    def caller_for(self, key: str) -> tuple[str, Caller] | None:
        digest = hashlib.sha256(key.encode()).hexdigest()
        for name, c in self.callers.items():
            if c.key_sha256 == digest:
                return name, c
        return None

    def provider_for(self, model: str, operation: Operation = "messages") -> tuple[str, Provider] | None:
        """The first provider that serves the model for this operation and is allowed for this data
        class. Where a BAA is required, a self-hosted model qualifies too: the data stays inside."""
        rule = self.rule()
        for name, p in self.providers.items():
            allowed = p.baa or p.self_hosted or not rule.require_baa
            if p.serves(model) and p.does(operation) and allowed:
                return name, p
        return None

    def price(self, model: str) -> Price | None:
        for pattern, price in self.pricing.items():
            if fnmatch.fnmatch(model, pattern):
                return price
        return None


def load_policy(path: str | Path) -> Policy:
    return Policy.model_validate(yaml.safe_load(Path(path).read_text()))
