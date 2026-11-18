# Shared LLM gateway

One way for every Valxel agent to reach a model. Agents use the standard Anthropic SDK with
`base_url` pointed at the gateway and their own gateway key; the gateway holds the provider keys
and applies the product's rules. Shared services ship no runtime, so **each product runs its own
gateway** in its OpenChoreo project, with its own policy, keys and budgets. PHI and spend stay
inside the product.

```python
client = anthropic.Anthropic(base_url=os.environ["LLM_GATEWAY_URL"], api_key=os.environ["LLM_GATEWAY_KEY"])
```

## What it enforces

| Rule | How |
| --- | --- |
| Approved providers only | A model is served only by a provider listed in the policy that does the operation (messages, embeddings, rerank); PHI products only by providers under a BAA or self-hosted models (`require_baa`) |
| Who may call what | Each agent or knowledge component has its own key (only its SHA-256 is in the policy), its own model list, a `max_tokens` cap and a `max_inputs` cap (texts per embeddings or rerank call) |
| Data that must not reach a model | Payments: card numbers (Luhn-checked) block the request. General: secrets, card numbers and SSNs are redacted. PHI: provider-level control instead, since the BAA covers the content |
| Spend | A daily budget per agent; the request is refused with `rate_limit_error` once it is used |
| Session tokens | A caller with a `sessions` grant mints a short-lived token per agent session (below) so the session never holds a provider key or the caller's own key. The token expires, has its own budget within the caller's, may use only the models it was minted for, carries fixed labels, and is revoked when the session ends |
| Accountability | One JSON log line per call (product, agent, model, tokens, cache reads and writes, cost, request id, any `x-valxel-*` labels such as component or feature id; `x-vrixel-*` still works) and Prometheus metrics |
| Audit | Every request is audited before it reaches a provider (CloudEvents `valxel.audit.v1` on the product's Kafka audit topic): caller, model, provider, kinds of data redacted, and every refusal with its reason. Never content. If the record can't be published the request is refused (`GATEWAY_AUDIT_REQUIRED`, default true) |
| Tracing | OpenTelemetry (`OTEL_EXPORTER_OTLP_ENDPOINT`): each request joins the agent's trace, with the provider call as a child span; logs are JSON with `trace.id`. The request's span carries the call in OpenTelemetry's GenAI terms (`gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.usage.*`) with `valxel.cost_usd`, the caller and its `agent`, `variant`, `run` labels, never the prompt or the reply |

## Embeddings and reranking

Knowledge (RAG) components embed and rerank through the gateway under the same rules as
messages: their own key and model list, the data class's provider rule, redaction, the daily
budget, the audit record before anything is sent, and the usage ledger.

- `POST /v1/embeddings` takes the OpenAI shape (`model`, `input`, `dimensions`) and returns the
  provider's answer: OpenAI, Azure OpenAI, vLLM, Hugging Face TEI, Voyage.
- `POST /v1/rerank` takes the Cohere shape (`model`, `query`, `documents`, `top_n`) and always
  answers `{"results": [{"index", "relevance_score"}]}`, best first, whether Cohere, vLLM, TEI or
  Voyage served it. Documents are never echoed back.

Anthropic providers serve messages only. An `openai`-type provider serves messages and embeddings
by default; list `operations` to change that, and add `rerank` for servers that rerank. For PHI,
embed with a provider under a BAA or with a self-hosted model (`self_hosted: true`, often with no
`api_key_env`): the data never leaves the product. The choice is made at the architecture gate
(G4) and written in the HLD's Knowledge section. Price embeddings by `input` tokens; a rerank
priced per search adds `per_request`.

It passes Anthropic requests through unchanged, so prompt caching (`cache_control`), tools,
extended thinking and streaming behave exactly as with the API. OpenAI-compatible providers
(OpenAI, Azure OpenAI, vLLM) are translated for text and tool use, without streaming.

## Run it

```bash
uv sync
GATEWAY_POLICY=config/policy.example.yaml ANTHROPIC_API_KEY=... uv run llm-gateway
uv run pytest                 # tests, coverage at least 90%
uv run ruff check . && uv run mypy
```

Deploy it in each product's OpenChoreo project with:
- `GATEWAY_POLICY` pointing at the product's policy (a config map); `data_class` must match the
  product's data class in the platform registry.
- Provider keys from secret references (`ANTHROPIC_API_KEY`, ...). Never in the policy or the repo.
- One replica, or move the budget ledger to Redis or Postgres first (it is in memory).
- No ingress from outside the project: only the product's agents call it.

## Session tokens

An orchestrator that runs agent sessions, such as the SDLC tool running Claude Code, gets a
`sessions` grant in the policy and mints one token per session:

```yaml
callers:
  sdlc-tool:
    key_sha256: "..."
    budget_usd_per_day: 200
    models: ["claude-*"]
    max_tokens: 64000
    sessions: { max_budget_usd: 20, max_ttl_s: 7200, max_open: 50 }
```

The session points its Anthropic client at the gateway (`ANTHROPIC_BASE_URL`) with the token as
its key. A gateway serves sessions of its own data class, or general ones: a PHI session can't be
minted on a gateway set up for general data. A session's spend counts towards its own budget and
its caller's daily one; a streamed call that crosses the session's budget finishes, and the next
is refused. A session may use only models the policy prices (a self-hosted one at 0, say), so its budget
always holds: that is the live dollar cap for agents with none of their own, such as Codex. Tokens live in memory, like the ledger: run one replica, and a restart ends every open
session. Minting and revoking are audited (`llm.session.minted`, `llm.session.revoked`).

## One gateway for several data classes

A gateway serves its deployment's data class. An orchestrator that works for products of several
classes, such as the SDLC tool, can use one gateway for all of them: its caller entry lists the
classes it may declare, and each request declares its class with `x-valxel-data-class` (a session
token is held to the class it was minted for, whatever it sends). The declared class's rule
applies: which providers may serve it (a BAA for PHI), and its redaction. A caller declaring a class
it may not is refused, and a policy can't list a class with no rule. The audit record names the
class each request was held to.

```yaml
data_class: general
callers:
  sdlc-tool:
    data_classes: [payments, phi]
```

## Audit settings

| Variable | Default | |
| --- | --- | --- |
| `GATEWAY_AUDIT_SINK` | `log` | `kafka` in deployed environments |
| `GATEWAY_AUDIT_KAFKA_BOOTSTRAP`, `GATEWAY_AUDIT_TOPIC` | `audit.v1` | The product's Kafka and audit topic |
| `GATEWAY_AUDIT_KAFKA_CONFIG` | `{}` | JSON of extra client settings (`security.protocol`, `sasl.*`) |
| `GATEWAY_AUDIT_REQUIRED` | `true` | Refuse requests whose audit record isn't acknowledged |
| `GATEWAY_ENVIRONMENT` | | Stamped on audit records |

## Endpoints

| Path | Purpose |
| --- | --- |
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/messages/count_tokens` | Anthropic token counting |
| `POST /v1/sessions` | Mint a session token: `{budget_usd, ttl_s, data_class, models?, labels?}` → `{id, token, expires_at, ...}`. The caller's own key only, and only with a `sessions` grant |
| `GET /v1/sessions/{id}`, `DELETE /v1/sessions/{id}` | Read a session's spend, or revoke it (returns what it spent) |
| `POST /v1/responses` | OpenAI's Responses API, streamed or not, passed through to an OpenAI-type provider (Codex). The same rules as messages: models, the data class's provider rule, redaction of instructions and input, `max_output_tokens` under the caller's cap, budgets, audit, ledger |
| `POST /v1/embeddings` | OpenAI-style embeddings |
| `POST /v1/rerank` | Cohere-style reranking |
| `GET /healthz`, `GET /readyz` | Liveness; readiness checks the policy and provider keys (self-hosted models without a key are fine) |
| `GET /metrics` | `llm_gateway_tokens_total`, `llm_gateway_cost_usd_total`, `llm_gateway_requests_total`, `llm_gateway_redactions_total` |
