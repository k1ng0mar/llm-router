// Package route drives candidate selection and the fallback chain.
// The contract: a 429 or 5xx from an upstream is a fallback trigger, never the
// caller's error; a 4xx is a client error and stops the chain immediately.
// The single exception: a detectable context-window-exceeded 400 falls through
// to the next candidate (the pre-call gate should prevent it, but if a gate
// slips the request recovers instead of erroring).
package route

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

// ErrExhausted means every candidate failed. Callers map it to a clean 503.
var ErrExhausted = errors.New("all candidates failed; fallback exhausted")

// Attempt is one upstream call (or a structural exclusion) in the trail.
type Attempt struct {
	Seq         int
	Provider    string
	Model       string
	KeyID       string
	Status      int // 0 = not attempted (excluded/cooldown), HTTP code otherwise
	LatencyMs   int
	Cost        float64 // always 0 for now — pending a pricing table
	Err         string
	ErrorOrigin string // "router" = router refused (gate/config), "upstream" = provider returned it
}

// Result carries the full decision trail plus the winning upstream response.
type Result struct {
	Pool     string
	Rule     string
	Attempts []Attempt
	Status   int // final HTTP status observed/decided
	Resp     *http.Response
}

// Router picks candidates and drives fallback for one request.
type Router struct {
	cfg     *config.Config
	gate    *catalog.Gate
	client  *provider.Client
	pMu     sync.Mutex
	pickers map[string]*provider.KeyPicker
}

// NewRouter wires the router. client may be nil (defaults to http.DefaultClient).
func NewRouter(cfg *config.Config, gate *catalog.Gate, client *provider.Client) *Router {
	if client == nil {
		client = &provider.Client{HTTP: http.DefaultClient}
	}
	return &Router{cfg: cfg, gate: gate, client: client, pickers: map[string]*provider.KeyPicker{}}
}

// pickerFor returns the cached KeyPicker for a provider, building one from the
// current key list if absent. The cache is invalidated (see InvalidatePicker)
// whenever an admin action changes a provider's keys, so rotation actually
// takes effect on the next request.
func (r *Router) pickerFor(name string, keys []string) *provider.KeyPicker {
	r.pMu.Lock()
	defer r.pMu.Unlock()
	if p, ok := r.pickers[name]; ok {
		return p
	}
	p := provider.NewKeyPicker(r.cfg.GetFallback().Strategy, keys)
	p.SetCooldown(time.Duration(r.cfg.GetFallback().KeyCooldownS) * time.Second)
	r.pickers[name] = p
	return p
}

// InvalidatePicker drops the cached picker for a provider so it is rebuilt with
// fresh keys on the next request.
func (r *Router) InvalidatePicker(name string) {
	r.pMu.Lock()
	delete(r.pickers, name)
	r.pMu.Unlock()
}

// InvalidateAllPickers drops every cached picker (used on config reload).
func (r *Router) InvalidateAllPickers() {
	r.pMu.Lock()
	r.pickers = map[string]*provider.KeyPicker{}
	r.pMu.Unlock()
}

// resolveEntries determines which pool entries to try, based on the model field:
//   - "auto", "", or omitted → pool classifier already picked the pool; use
//     cfg.TierSortedEntries(pool) so tiers reorder entries cheapest-first.
//   - "chain:<name>" → cfg.Chains[name] (explicit order, never tier-reordered)
//   - "provider:model" (single) → just that one entry
//   - "a:x,b:y" (comma-separated) → inline chain (explicit order)
//   - "provider:model" that exists in cfg.Chains → named chain (precedence)
//
// Any fallback that resolves back to the pool uses TierSortedEntries so tier
// ordering is honored consistently.
func (r *Router) resolveEntries(pool string, payload map[string]any) []string {
	model, _ := payload["model"].(string)
	model = strings.TrimSpace(model)

	// auto or empty → use pool entries, tier-ordered if configured
	if model == "" || model == "auto" {
		return r.cfg.TierSortedEntries(pool)
	}

	// chain:<name> → named chain (explicit order; NOT tier-reordered)
	if strings.HasPrefix(model, "chain:") {
		name := strings.TrimPrefix(model, "chain:")
		if chain, ok := r.cfg.GetChain(name); ok {
			return chain
		}
		// unknown chain → fall back to pool (still tier-ordered)
		return r.cfg.TierSortedEntries(pool)
	}

	// comma-separated → inline chain (explicit order)
	if strings.Contains(model, ",") {
		parts := strings.Split(model, ",")
		var entries []string
		for _, p := range parts {
			p = strings.TrimSpace(p)
			if p != "" {
				entries = append(entries, p)
			}
		}
		return entries
	}

	// single "provider:model" → just that entry
	// (verify it resolves; if not, fall back to pool)
	if _, _, err := r.cfg.Resolve(model); err == nil {
		return []string{model}
	}

	// unknown model string → fall back to pool (still tier-ordered)
	return r.cfg.TierSortedEntries(pool)
}

// Route runs the candidates through the capability gate, then attempts
// each in order with per-key rotation; 429/5xx/context-window-400/timeout/network
// rotate onward. 401/403/other-4xx stop the chain immediately.
// On success, Result.Resp is a live upstream response the caller must close.
//
// The model field in payload controls routing mode:
//   - "auto"/empty → pool classifier entries (tier-ordered if configured)
//   - "chain:fast" → named chain from config
//   - "a:x,b:y" → inline chain
//   - "a:x" → single model, skip classifier
func (r *Router) Route(ctx context.Context, pool string, payload map[string]any, hasImage, hasAudio, hasVideo bool, minContext int) (*Result, error) {
	res := &Result{Pool: pool}
	entries := r.resolveEntries(pool, payload)
	if len(entries) == 0 {
		res.Status = 503
		res.Attempts = append(res.Attempts, Attempt{Seq: 1, Err: fmt.Sprintf("pool %q has no candidates", pool), ErrorOrigin: "router"})
		return res, ErrExhausted
	}

	attemptTimeout := time.Duration(r.cfg.GetFallback().TimeoutS) * time.Second
	ap := func(providerName, model, keyID string, origin string) *Attempt {
		res.Attempts = append(res.Attempts, Attempt{})
		att := &res.Attempts[len(res.Attempts)-1]
		att.Seq = len(res.Attempts)
		att.Provider, att.Model, att.KeyID, att.ErrorOrigin = providerName, model, keyID, origin
		return att
	}

	for _, ref := range entries {
		pName, model, err := r.cfg.Resolve(ref)
		if err != nil {
			att := ap(ref, "", "", "router")
			att.Err = err.Error()
			continue
		}

		// structural capability gate: hard exclusion, fail-open for unknowns
		needsTools := 0
		if toolsFromPayload(payload) {
			needsTools = 1
		}
		if ok, reason := r.gate.Check(ref, hasImage, hasAudio, hasVideo, minContext, needsTools); !ok {
			att := ap(pName, model, "", "router")
			att.Status = 0
			att.Err = "excluded: " + reason
			continue
		}

		baseURL, keys, enabled, modelDisabled, modelLimit, hasLimit, ok, accountID, apiMode := r.cfg.GetProviderRouting(pName, model)
		if !ok || baseURL == "" {
			att := ap(pName, model, "", "router")
			att.Err = "provider has no base_url"
			continue
		}
		if !enabled {
			att := ap(pName, model, "", "router")
			att.Err = "provider disabled"
			continue
		}
		if modelDisabled {
			att := ap(pName, model, "", "router")
			att.Err = "model disabled"
			continue
		}
		picker := r.pickerFor(pName, keys)

		keyLoop:
		for {
			keyIdx, key, okk := picker.Next(time.Now())
			if !okk {
				if len(keys) == 0 {
					att := ap(pName, model, "", "router")
					att.Err = "no API key configured for provider"
				}
				break keyLoop
			}
		att := ap(pName, model, key, "upstream")

			// rewrite the model field so the upstream receives the raw model id
			// without the router's "provider:" namespace prefix
			upstreamPayload := payload
			payloadCopied := false
			if _, hasModel := payload["model"]; hasModel {
				upstreamPayload = make(map[string]any, len(payload)+1)
				for k, v := range payload {
					upstreamPayload[k] = v
				}
				upstreamPayload["model"] = model
				payloadCopied = true
			}
			// apply per-model max_tokens cap if configured
			if hasLimit {
				if !payloadCopied {
					upstreamPayload = make(map[string]any, len(payload)+1)
					for k, v := range payload {
						upstreamPayload[k] = v
					}
					payloadCopied = true
				}
				if mt, exists := upstreamPayload["max_tokens"]; exists {
					reqMT, _ := mt.(float64)
					if int(reqMT) > modelLimit {
						upstreamPayload["max_tokens"] = modelLimit
					}
				} else {
					upstreamPayload["max_tokens"] = modelLimit
				}
			}

			attemptCtx := ctx
			var cancel context.CancelFunc
			if attemptTimeout > 0 {
				attemptCtx, cancel = context.WithTimeout(ctx, attemptTimeout)
			}
			start := time.Now()
		// Substitute per-provider templating in base_url. The supported
		// placeholder is "{account_id}", replaced by the provider's
		// AccountID field. A provider with no AccountID keeps "{account_id}"
		// literal (it will 404 upstream, surfacing a config gap loudly
		// rather than silently hitting the wrong account).
		upBaseURL := baseURL
		if accountID != "" && strings.Contains(upBaseURL, "{account_id}") {
			upBaseURL = strings.ReplaceAll(upBaseURL, "{account_id}", accountID)
		}
		resp, doErr := r.client.Do(attemptCtx, &provider.Upstream{Name: pName, BaseURL: upBaseURL, Keys: keys, APIMode: apiMode}, key, upstreamPayload)
		att.LatencyMs = int(time.Since(start).Milliseconds())

		if doErr != nil {
			if cancel != nil {
				cancel()
			}
			picker.MarkFailure(keyIdx, time.Now(), 0, 0)
			// Make the failure legible instead of dumping a raw
			// `Post "...": context canceled`. Differentiate:
			//   - attempt deadline hit (upstream too slow)
			//   - parent request canceled (client left / edge timeout)
			//   - genuine network/transport error
			att.Err = describeAttemptError(doErr, attemptTimeout, ctx)
			continue
		}

		att.Status = resp.StatusCode
		var body string
		switch {
		case provider.Retryable(att.Status):
			body = readErrBody(resp)
			// capture the upstream body so operators can see *why* the attempt
			// failed (e.g. the rate-limit / 5xx message) in the request trail.
			att.Err = body
			ra := parseRetryAfter(resp.Header)
			picker.MarkFailure(keyIdx, time.Now(), ra, att.Status)
			resp.Body.Close()
			if cancel != nil {
				cancel()
			}
			continue
		case att.Status >= 400:
			body = readErrBody(resp)
			if provider.IsContextWindowError(att.Status, body) {
				att.Err = "context window exceeded (fallback-eligible 4xx)"
				picker.MarkFailure(keyIdx, time.Now(), 0, att.Status)
				resp.Body.Close()
				if cancel != nil {
					cancel()
				}
				continue
			}
			// Upstream infrastructure 400s (e.g. General Compute's generic
			// "Streaming request failed" marked type:"provider_error") are not
			// client errors — treat them like 5xx and rotate to the next
			// candidate instead of killing the chain.
			if provider.IsProviderError(att.Status, body) {
				att.Err = body
				picker.MarkFailure(keyIdx, time.Now(), 0, att.Status)
				resp.Body.Close()
				if cancel != nil {
					cancel()
				}
				continue
			}
			picker.MarkFailure(keyIdx, time.Now(), 0, att.Status)
			att.Err = body
			res.Status = att.Status
			resp.Body.Close()
			if cancel != nil {
				cancel()
			}
			return res, fmt.Errorf("upstream %s rejected request (HTTP %d): %s", pName, att.Status, body)
		default: // 2xx — success
			picker.MarkSuccess(keyIdx)
			res.Status = att.Status
			res.Resp = resp
			// Wrap the body so the per-attempt timeout context is canceled
			// once the caller finishes reading/closing it. Previously cancel
			// was intentionally not called on the success path (so the caller
			// could stream the body), which leaked the timer until it fired and
			// tripped `go vet`. cancelOnClose fixes both.
			if cancel != nil {
				res.Resp.Body = &cancelOnClose{rc: resp.Body, cancel: cancel}
			}
			return res, nil
		}
		}
	}

	res.Status = 503
	return res, ErrExhausted
}

// describeAttemptError converts a transport/context error into a legible
// failure string so operators can tell *why* an attempt died without digging
// into raw Go error text.
func describeAttemptError(doErr error, attemptTimeout time.Duration, parentCtx context.Context) string {
	raw := doErr.Error()
	// The per-attempt deadline fired — upstream took too long.
	if errors.Is(doErr, context.DeadlineExceeded) {
		return fmt.Sprintf("upstream timed out after %s (attempt deadline)", attemptTimeout)
	}
	// Parent request context canceled — the client went away (or an edge/proxy
	// timeout killed the connection). Nothing the router can do; say so.
	if parentCtx != nil && parentCtx.Err() != nil {
		if errors.Is(parentCtx.Err(), context.Canceled) {
			return fmt.Sprintf("client disconnected / request canceled before upstream responded: %s", raw)
		}
		if errors.Is(parentCtx.Err(), context.DeadlineExceeded) {
			return fmt.Sprintf("request deadline exceeded before upstream responded: %s", raw)
		}
	}
	return raw
}

func readErrBody(resp *http.Response) string {
	buf := make([]byte, 4096)
	n, _ := resp.Body.Read(buf)
	s := string(buf[:n])
	// Drain the remainder so the upstream connection can be reused (keep-alive)
	// instead of being closed prematurely.
	io.Copy(io.Discard, resp.Body)
	if len(s) > 2000 {
		s = s[:2000]
	}
	return s
}

// toolsFromPayload reports whether the request payload includes a tools array.
func toolsFromPayload(p map[string]any) bool {
	t, ok := p["tools"]
	if !ok {
		return false
	}
	arr, ok := t.([]any)
	return ok && len(arr) > 0
}

// parseRetryAfter extracts a duration from the Retry-After header
// (supports both delta-seconds and HTTP-date forms).
func parseRetryAfter(h http.Header) time.Duration {
	ra := h.Get("Retry-After")
	if ra == "" {
		return 0
	}
	if secs, err := strconv.Atoi(strings.TrimSpace(ra)); err == nil {
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(ra); err == nil {
		return time.Until(t)
	}
	return 0
}

// cancelOnClose wraps an io.ReadCloser so that the provided cancel func is
// called exactly once when Close() is invoked. It is used by Route to defer
// cancellation of the per-attempt timeout context until the caller has
// finished streaming the successful upstream response body — keeping the
// connection alive for streaming while still releasing the timer promptly.
type cancelOnClose struct {
	rc     io.ReadCloser
	cancel context.CancelFunc
	once   bool
}

func (c *cancelOnClose) Read(p []byte) (int, error) { return c.rc.Read(p) }
func (c *cancelOnClose) Close() error {
	err := c.rc.Close()
	if !c.once {
		c.once = true
		if c.cancel != nil {
			c.cancel()
		}
	}
	return err
}
