// Package provider talks to OpenAI-compatible upstreams and rotates keys
// with per-status state tracking: 401 = dead, 429 = cooldown, 5xx = short cooldown.
package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Client performs upstream calls.
type Client struct {
	HTTP *http.Client
}

func (c *Client) httpClient() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return http.DefaultClient
}

// NewClientWithTTFB creates a Client whose HTTP transport times out if
// the upstream doesn't return response headers within ttfb. The body
// streams freely after headers arrive — only the initial wait is bounded.
func NewClientWithTTFB(ttfb time.Duration) *Client {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.ResponseHeaderTimeout = ttfb
	return &Client{HTTP: &http.Client{Transport: t}}
}

// Upstream is one provider endpoint.
type Upstream struct {
	Name    string
	BaseURL string
	Keys    []string
	// APIMode selects the upstream wire format: "openai" (default),
	// "anthropic", or "gemini". Non-OpenAI modes translate the request and
	// response at this boundary so the rest of the router stays OpenAI-shaped.
	APIMode string
}

// Do POSTs payload to the upstream's chat endpoint. When key is "" no
// Authorization header is sent (keyless local providers). The caller owns the
// returned response body and must close it.
//
// For OpenAI-compatible upstreams the payload and response pass through
// unchanged. For Anthropic/Gemini upstreams the request is built in the
// upstream's native format and the 2xx response is translated back to OpenAI
// shape, so the caller (and the client) never sees the foreign wire format.
func (c *Client) Do(ctx context.Context, up *Upstream, key string, payload map[string]any) (*http.Response, error) {
	mode := normalizeMode(up.APIMode)
	if mode == ModeOpenAI {
		return c.doOpenAI(ctx, up, key, payload)
	}
	return c.doNative(ctx, up, key, payload, mode)
}

// doOpenAI is the original OpenAI-compatible path: POST the payload verbatim
// to {base}/v1/chat/completions with a Bearer key.
func (c *Client) doOpenAI(ctx context.Context, up *Upstream, key string, payload map[string]any) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	base := strings.TrimSuffix(up.BaseURL, "/")
	if !strings.HasSuffix(base, "/v1") {
		base += "/v1"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	return c.httpClient().Do(req)
}

// doNative builds the upstream's native request, POSTs it, and on 2xx
// translates the response back to OpenAI shape. Non-2xx responses pass through
// untouched so the router's status handling (retry/terminal/fallback) applies.
func (c *Client) doNative(ctx context.Context, up *Upstream, key string, payload map[string]any, mode string) (*http.Response, error) {
	body, model, err := buildNativeRequest(mode, payload)
	if err != nil {
		return nil, err
	}
	base := strings.TrimSuffix(up.BaseURL, "/")
	url := base + nativeEndpoint(mode)
	if mode == ModeGemini {
		// Gemini puts the model in the URL: /v1beta/models/{model}:generateContent
		if model == "" {
			return nil, fmt.Errorf("gemini upstream requires a model")
		}
		url = base + "/v1beta/models/" + urlPathEscape(model) + ":generateContent"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	switch mode {
	case ModeAnthropic:
		req.Header.Set("x-api-key", key)
		req.Header.Set("anthropic-version", "2023-06-01")
	case ModeGemini:
		if key != "" {
			// Gemini accepts the key as a query param or x-goog-api-key header.
			req.Header.Set("x-goog-api-key", key)
		}
	default:
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
	}

	resp, err := c.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	// Only translate 2xx bodies. Error bodies (4xx/5xx) are read by the router's
	// status handling as raw text, so leave them untouched.
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		raw, rerr := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		resp.Body.Close()
		if rerr != nil {
			return nil, rerr
		}
		translated, terr := translateNativeResponse(mode, raw, payload)
		if terr != nil {
			return nil, terr
		}
		streaming, _ := payload["stream"].(bool)
		return syntheticResponse(translated, streaming), nil
	}
	return resp, nil
}

func urlPathEscape(s string) string {
	// Model ids like "models/gemini-pro" contain "/" which must be escaped in
	// the URL path. Use a minimal escape for the common characters.
	r := strings.NewReplacer("/", "%2F", " ", "%20", ":", "%3A")
	return r.Replace(s)
}

// Retryable reports codes that are fallback triggers: 429 or any 5xx.
func Retryable(code int) bool {
	return code == 429 || code >= 500
}

// Terminal reports client-error codes that must NOT trigger fallback.
// 429 is excluded (it's retryable). Context-window-400 is handled separately
// by IsContextWindowError.
func Terminal(code int) bool {
	return code >= 400 && code < 500 && code != 429
}

// IsContextWindowError reports whether a 4xx body indicates the request was
// too long for the model and should be retried against a larger model.
// This is the single exception to "4xx never fallback".
func IsContextWindowError(code int, body string) bool {
	if code < 400 || code >= 500 {
		return false
	}
	body = strings.ToLower(body)
	return strings.Contains(body, "context") &&
		(strings.Contains(body, "length") || strings.Contains(body, "window") ||
			strings.Contains(body, "too long") || strings.Contains(body, "exceed"))
}

// IsProviderError reports whether a 4xx body indicates an upstream
// infrastructure failure (type:"provider_error") rather than a client-side
// rejection. Providers like General Compute return a generic HTTP 400 with
// type:"provider_error" when their streaming pipeline fails mid-flight; the
// request itself is valid, so the router should fall back to the next
// candidate instead of terminating the chain.
func IsProviderError(code int, body string) bool {
	if code < 400 || code >= 500 {
		return false
	}
	return strings.Contains(strings.ToLower(body), `"provider_error"`) ||
		strings.Contains(strings.ToLower(body), `"type":"provider_error"`)
}

// KeyState is the per-key failure state.
type KeyState int

const (
	KeyOK KeyState = iota
	KeyCooldown
	KeyDead
)

// KeyPicker rotates keys with a strategy and tracks per-key state:
//   - 401 → dead (manual rotation required)
//   - 429 → cooldown (Retry-After honored, capped, else default)
//   - anything else (5xx, other 4xx, timeouts, transport errors) → no cooldown
//   - success → clears transient state
//
// Only a rate limit or an auth failure says anything about the *key*. A 5xx or a
// timeout describes the upstream, or the one model behind it — and keys are
// shared by every model on a provider, so parking a key for those reasons took
// that provider's whole model list offline because a single model was having a
// bad minute. It also silently made the key stack the thing that ended a
// fallback loop; that job now belongs to the caller (see NextExcluding).
type KeyPicker struct {
	mu            sync.Mutex
	strategy      string
	keys          []string
	usage         []int
	failUntil     []time.Time
	dead          []bool
	rr            int
	defaultCD     time.Duration
	retryAfterCap time.Duration
}

// NewKeyPicker builds a picker for keys. Strategy: round_robin | least_used.
func NewKeyPicker(strategy string, keys []string) *KeyPicker {
	return &KeyPicker{
		strategy:      strategy,
		keys:          keys,
		usage:         make([]int, len(keys)),
		failUntil:     make([]time.Time, len(keys)),
		dead:          make([]bool, len(keys)),
		defaultCD:     60 * time.Second,
		retryAfterCap: 300 * time.Second,
	}
}

// SetCooldown overrides the default 429 cooldown duration.
func (p *KeyPicker) SetCooldown(d time.Duration) { p.defaultCD = d }

// SetRetryAfterCap caps how long a Retry-After value can keep a key down.
func (p *KeyPicker) SetRetryAfterCap(d time.Duration) { p.retryAfterCap = d }

// Next returns the next eligible key index. ok=false when all keys are dead/cooling.
func (p *KeyPicker) Next(now time.Time) (int, string, bool) {
	return p.NextExcluding(now, nil)
}

// NextExcluding is Next with a set of key indices to skip. ok=false when every
// key is dead, cooling, or excluded.
//
// A caller driving a fallback loop should pass the keys it has already tried for
// the current candidate. That bounds the loop at one attempt per key no matter
// what the cooldown policy does, so termination never depends on a failure
// having parked the key — which is what it used to depend on.
func (p *KeyPicker) NextExcluding(now time.Time, skip map[int]bool) (int, string, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.keys) == 0 {
		return 0, "", false
	}
	usable := func(i int) bool {
		if p.dead[i] || skip[i] {
			return false
		}
		return now.After(p.failUntil[i]) || p.failUntil[i].IsZero()
	}

	switch p.strategy {
	case "least_used":
		best := -1
		for i := range p.keys {
			if !usable(i) {
				continue
			}
			if best == -1 || p.usage[i] < p.usage[best] {
				best = i
			}
		}
		if best == -1 {
			return 0, "", false
		}
		p.rr = (best + 1) % len(p.keys)
		return best, p.keys[best], true
	default: // round_robin
		for n := 0; n < len(p.keys); n++ {
			i := (p.rr + n) % len(p.keys)
			if usable(i) {
				p.rr = (i + 1) % len(p.keys)
				p.usage[i]++
				return i, p.keys[i], true
			}
		}
		return 0, "", false
	}
}

// MarkFailure records a failed attempt on index. Only two statuses affect the
// key's availability:
//   - 401 → key is marked dead (manual reset required)
//   - 429 → cooldown = retryAfter (if > 0, capped at retryAfterCap), else defaultCD
//
// Every other outcome — 5xx, other 4xx, a TTFB timeout, a transport error — is
// recorded for usage accounting and nothing more. Those describe the upstream or
// the model, not the key, and keys are per provider: cooling one would strand
// every other model that provider serves. See the KeyPicker type comment.
//
// retryAfter is ignored for non-429 statuses.
func (p *KeyPicker) MarkFailure(idx int, now time.Time, retryAfter time.Duration, status int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if idx < 0 || idx >= len(p.keys) {
		return
	}
	p.usage[idx]++

	switch status {
	case 401:
		p.dead[idx] = true
		p.failUntil[idx] = time.Time{} // dead, not just cooling
	case 429:
		cd := p.defaultCD
		if retryAfter > 0 {
			cd = retryAfter
			if cd > p.retryAfterCap {
				cd = p.retryAfterCap
			}
		}
		p.failUntil[idx] = now.Add(cd)
	}
}

// MarkSuccess records a successful use (counts toward least_used balance).
func (p *KeyPicker) MarkSuccess(idx int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if idx >= 0 && idx < len(p.keys) {
		p.usage[idx]++
		p.failUntil[idx] = time.Time{} // clear any transient cooldown
	}
}
