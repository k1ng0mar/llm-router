# llm-router

A single-binary, self-hosted LLM router. It exposes an OpenAI-compatible endpoint, routes each request to the right model, falls back when an upstream fails, and gives you a dashboard to watch it all happen.

It's plain Go, no framework, one binary. SQLite keeps the request log.

## What it does

- **One endpoint, many models.** Point your client at `/v1/chat/completions`. The router picks a pool, tries the models in order, and moves to the next one if an upstream errors or times out.
- **Pools.** Group models by job. `chat`, `code`, `vision`, whatever you want. Requests pick a pool by header or by model name.
- **Fallback chain.** If the first model in a pool is down or slow, the router tries the next. It stops when one answers or the chain runs out.
- **Capability gate.** It knows what each model can do (context window, vision, tools) and won't send a request a model can't handle.
- **Non-OpenAI upstreams.** Most providers speak OpenAI's format. For Anthropic and Gemini, set `api_mode` on the provider and the router translates the request and response for you.
- **Dashboard.** A read/write web UI: watch requests live, see where they routed, edit config, add providers, build pools.
- **Request log.** Every request and attempt lands in SQLite, so you can go back and see exactly what happened and what it cost.

## Quick start

```bash
cp router.example.yaml router.yaml
# edit router.yaml: set router_key, add your provider keys
go build -o llm-router .
./llm-router serve --config router.yaml
```

Then send it a request:

```bash
curl http://127.0.0.1:8015/v1/chat/completions \
  -H "Authorization: Bearer YOUR_ROUTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"chat","messages":[{"role":"user","content":"hello"}]}'
```

Open the dashboard at `http://127.0.0.1:8015/`.

## Configuration

Everything lives in `router.yaml`. Copy `router.example.yaml` for a starting point.

- **`router_key`** — the Bearer token every API and dashboard request must carry. Set a strong random value in production.
- **`pools`** — named lists of `provider:model` entries. Requests pick a pool by header (`X-Route-Pool`) or by sending a model name that matches a pool.
- **`providers`** — upstream endpoints and keys. `openrouter` and `ollama` are built in; anything else under `providers.custom` is yours. Each provider can set:
  - `base_url` — the API root.
  - `keys` — one or more keys. The router rotates through them.
  - `api_mode` — `openai` (default), `anthropic`, or `gemini`. Picks the wire format.
  - `account_id` — fills an `{account_id}` placeholder in `base_url` (used by Cloudflare Workers AI).
- **`chains`** / **`tiers`** — advanced routing rules. Leave empty unless you need them.

## Endpoints

| Path | Purpose |
|------|---------|
| `POST /v1/chat/completions` | The chat endpoint. OpenAI-compatible. |
| `GET /healthz` | Health check. |
| `GET /` | Dashboard. |
| `GET /api/config` | Read config (providers, pools). |
| `POST /api/config/providers` | Add, update, or delete a provider. |
| `PUT /api/config/providers/{name}` | Update a provider's model limits and disabled models. |

## Running it properly

It's a long-lived process. Run it under systemd or similar so it restarts if it dies. It handles `SIGHUP` to reload `router.yaml` in place, so you can change config without a restart.

## Development

```bash
go build ./...
go vet ./...
go test ./...
```

The test suite covers the router logic end to end: fallback behavior, pool selection, config, the provider translation layer, and the HTTP server.

## Layout

```
main.go                  entry point, flags, config load
internal/config/         config model, load/save, provider state
internal/catalog/        model catalog (names, context, capabilities)
internal/classify/       request routing (pool selection)
internal/provider/       upstream HTTP calls, non-OpenAI translation
internal/route/          the request lifecycle: gate, route, fallback, log
internal/server/         HTTP server, dashboard, config API
internal/store/          SQLite request log
```