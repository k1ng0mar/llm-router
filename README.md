# llm-router

A single-binary, self-hosted LLM router. It exposes an OpenAI-compatible endpoint, routes each request to the right model, falls back when an upstream fails, and gives you a dashboard to watch it all happen.

It's plain Go, no framework, one binary. SQLite keeps the request log.

## What it does

- **One endpoint, many models.** Point your client at `/v1/chat/completions`. The router picks a pool, tries the models in order, and moves to the next one if an upstream errors or times out.
- **Pools.** Group models by job. `chat`, `code`, `media`, whatever you want. Requests pick a pool by header or by model name.
- **Fallback chain.** If the first model in a pool is down or slow, the router tries the next — each key, then each provider. It stops when one answers or the chain runs out. One bounded wait per key, and no overall deadline that could cut a long response short.
- **Media pool.** Images, audio, and video all route to one `media` pool. The gate picks the entry that can handle whichever form arrived, so a pool can mix single-modality models freely.
- **Capability gate.** It knows what each model can do (context window, image/audio/video, tools) and won't send a request a model can't handle. You can override it per model when the catalog is wrong.
- **Non-OpenAI upstreams.** Most providers speak OpenAI's format. For Anthropic and Gemini, set `api_mode` on the provider and the router translates the request and response for you.
- **Dashboard.** A read/write web UI: watch requests live, see where they routed, edit config, add providers, build pools. Unlocks with your `router_key`, so it's safe to reach over a private network.
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
- **`pools`** — named lists of `provider:model` entries. A pool named `media` gets requests carrying images, audio, or video (see below). Requests pick a pool in one of these ways:
  - the `X-Route-Pool` header (wins over everything);
  - a `model` field naming a pool — `"model": "code"` uses the `code` pool, logged as rule `model-pool`. This is how clients that can only set `model` steer routing;
  - `"model": "router"`, `"auto"`, or anything unrecognized — the classifier picks;
  - `"model": "provider:id"`, `"a:x,b:y"`, or `"chain:name"` — bypass pools entirely and use exactly those candidates, in order.
- **`providers`** — upstream endpoints and keys. `openrouter` and `ollama` are built in; anything else under `providers.custom` is yours. Each provider can set:
  - `base_url` — the API root.
  - `keys` — one or more keys. The router rotates through them.
  - `api_mode` — `openai` (default), `anthropic`, or `gemini`. Picks the wire format.
  - `account_id` — fills an `{account_id}` placeholder in `base_url` (used by Cloudflare Workers AI).
  - `model_limits` — per-model `max_tokens` cap, applied as an upper bound.
  - `strip_params` — request fields to drop before forwarding (e.g. groq rejects `reasoning_effort`).
  - `media_policies` — per-model media overrides. See below.
- **`chains`** / **`tiers`** — advanced routing rules. Leave empty unless you need them.

## Media routing

Anything that isn't text — images, audio, video — is "media", and it all takes one path.

**Where it goes.** Pool selection runs in a fixed order: an `X-Route-Pool` header wins, then keyword heuristics on the last user turn, then media, then the default pool. So a request with an attachment and no other signal lands in the `media` pool. Media ranks below heuristics on purpose: "explain this code" plus a screenshot still belongs in `code`, where the describe-first hop turns the pixels into text a code model can read.

If you have no `media` pool, media-carrying requests fall to the default pool, exactly as they did before. A pool named `vision` is honored as a fallback so older configs keep working.

**Which model takes it.** Detection spans the whole conversation, not just the last turn — an image three turns back still has to be decodable by whoever answers now. The gate then excludes every candidate that can't handle the forms present, so you can put image-only, audio-only, and video-only models in one `media` pool and let each request find its own match. If nothing in the pool can handle it, you get a 503 with the exclusions recorded, not a request forwarded to a model that would choke on it.

### The describe hop

When an image lands in a pool whose models can't read pixels, the router doesn't give up — it sends the image to an image-capable model from the `media` pool, then answers from the original pool with that description folded in. So "explain this code" plus a screenshot gets you a code model's answer about the screenshot.

The code model receives the whole conversation, not just the description:

```
system    : You write Go.
user      : import this design and build the page

              [Image description from vision pass]
              a login form, email + password, blue submit button
```

The description is attached to the final user turn rather than appended as a message of its own — some upstreams reject two consecutive user messages.

A few details worth knowing:

- **It applies to any pool**, not a fixed list. An image in `creative` gets the hop just like one in `code`.
- **The `media` pool is exempt.** That's where the describers live, so a request routed there gets the pixels directly.
- **If the target pool's first model can see images and `allow_direct_vision: true`, the hop is skipped** and the pixels go straight there — no point paying for a description a model doesn't need.
- **A failed direct attempt falls back to the hop.** If every pixel-capable candidate errors out, the router describes the image and retries the same pool as a text request. The request log shows both passes in one trail, with the rule marked `…+described`.
- **With no image-capable model configured, nothing changes.** Pixels go out and the capability gate decides, rather than the router inventing an error.

One limitation to know about: API keys are per *provider*, not per model. If a model 5xxes across all of its provider's keys, the router cools those keys for 15s — which makes every other model on that same provider unreachable, including one the describe retry wanted to use. Spread the models in a pool across different providers and it doesn't come up.

**When the catalog is wrong.** Capability data comes from the model catalog, which is frequently out of date on newer omni models. Override it per model with `media_policies`:

```yaml
providers:
  custom:
    my-provider:
      media_policies:
        some-omni-model:
          image: allow   # force it through — you know it has native support
          audio: auto    # follow the catalog (the default)
          video: deny    # never send video here, whatever the catalog claims
```

`allow` and `deny` both outrank the catalog. `auto` (or omitting the model entirely) defers to it, and unknown models fail open. The dashboard exposes all three as dropdowns per model under **Providers → View Details → Models & Routing**.

## Timeouts and key health

**One deadline, per key.** `fallback.timeout_s` bounds the wait for response headers — time to first byte — on a single key. When it expires the router moves to the next key on that provider; when the keys run out it moves to the next provider. A timeout never fails the request on its own, it only advances the chain.

```
seq1 slow:hang  status=0    upstream timed out after 2s (attempt deadline)
seq2 slow:hang  status=0    upstream timed out after 2s (attempt deadline)
seq3 fast:good  status=200
```

**There is deliberately no overall request deadline.** Once headers arrive the body streams for as long as it takes, so a long generation is never cut off mid-stream. The only bound is on waiting for an upstream that has gone quiet.

**Key state.** Keys belong to a provider and are shared by every model that provider serves, so the router is conservative about taking one out of rotation:

| Upstream says | What happens to the key |
|---|---|
| `401` | Marked dead. Needs manual rotation — it won't come back on its own. |
| `429` | Cooled for `Retry-After` (capped at 5 min), else `key_cooldown_s`. |
| `5xx`, other `4xx`, timeout, transport error | Nothing. Still usable immediately. |

Only a rate limit or an auth failure tells you something about the *key*. A 5xx or a timeout tells you about the upstream, or the single model behind it — and since keys are per provider, parking one on that basis took every other model on that provider offline for the cooldown window. So it doesn't.

The chain still terminates: each candidate tries each of its keys at most once per request, tracked per request rather than inferred from the cooldown clock.

**Client hangs up.** If the caller disconnects (or its deadline passes) mid-chain, the router stops instead of dialing the rest of the pool for a response nobody can receive. Those requests log as `499` so they're distinguishable from a genuine `503` exhaustion.

**Reload caveat.** `SIGHUP` picks up most changes in place, including `key_cooldown_s`. `timeout_s` is applied to the HTTP transport at startup, so changing it needs a restart.

## Reaching the dashboard remotely

The router binds `127.0.0.1` and should stay that way. To use the dashboard from another machine, put a private network in front of it rather than changing `listen`. With [Tailscale](https://tailscale.com):

```bash
tailscale serve --bg --https=8443 http://127.0.0.1:8015
# -> https://<your-machine>.<tailnet>.ts.net:8443  (tailnet only)
```

Pick a port that isn't already serving something (`tailscale serve status` shows what is). The dashboard's absolute `/api/...` paths mean it wants the root of a port, not a sub-path mount.

The page itself is a static shell served without auth; every `/api` call carries the `router_key` as a Bearer token, which you enter once in the unlock screen and which is kept in that browser's `localStorage`. The server never embeds the key in the page, so serving the HTML gives nothing away, and **Lock** in the header clears it from the device.

Do not put this behind `tailscale funnel`. Funnel publishes to the open internet, and this dashboard can read and write provider config — anyone who obtains the key controls your upstream spend. Keep it tailnet-only.

## Endpoints

| Path | Purpose |
|------|---------|
| `POST /v1/chat/completions` | The chat endpoint. OpenAI-compatible. |
| `GET /v1/models` | Lists the routable model names: `router`, `auto`, and every pool. OpenAI-compatible, so a client's model picker offers real routing choices. |
| `GET /healthz` | Health check. |
| `GET /` | Dashboard. |
| `GET /api/config` | Read config (providers, pools). |
| `POST /api/config/providers` | Add, update, or delete a provider. |
| `PUT /api/config/providers/{name}` | Update a provider's model limits, disabled models, and media policies. |

## Running it properly

It's a long-lived process, so run it under a supervisor. `contrib/llm-router.service` is a ready systemd unit — a *user* unit, so it needs no root:

```bash
mkdir -p ~/.config/systemd/user
cp contrib/llm-router.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llm-router
loginctl enable-linger "$USER"   # so it starts at boot without you logged in
```

```bash
systemctl --user status llm-router
journalctl --user -u llm-router -f
```

`systemctl --user reload` sends `SIGHUP`, which re-reads `router.yaml` in place — same PID, no dropped requests — so pool, provider and key edits need no downtime. The one exception is `timeout_s`: it is applied to the HTTP transport at startup, so changing it needs a real `restart` (see [Timeouts and key health](#timeouts-and-key-health)). The unit allows 45s to stop, which gives the router room for its own 30s drain so pending log writes survive a restart.

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
internal/classify/       request routing (pool selection, media detection)
internal/provider/       upstream HTTP calls, non-OpenAI translation
internal/route/          the request lifecycle: gate, route, fallback, log
internal/server/         HTTP server, dashboard, config API
internal/store/          SQLite request log
contrib/                 systemd unit
```