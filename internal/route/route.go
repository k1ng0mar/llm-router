// Package route drives candidate selection and the fallback chain.
//
// The contract: any non-200 from an upstream is a fallback trigger, never the
// caller's error. A 429, a 5xx, a 4xx and a 3xx all record the attempt and move
// to the next candidate; the chain ends only when a candidate answers 200 or the
// candidate list runs out (ErrExhausted, surfaced as a 503). Transport failures
// and TTFB timeouts rotate onward the same way.
//
// This is deliberately blunt. Providers in this space disagree wildly about
// which status a rejected request deserves — an unsupported parameter, an
// over-long context, a model that quietly disappeared and an expired key arrive
// as 400, 404, 413 and 422 more or less interchangeably — so treating any 4xx as
// terminal stranded requests that the very next candidate would have served.
// The cost of the blunt rule is that a genuinely malformed request is retried
// down the whole chain before failing; the decision trail in Result.Attempts
// records every status, so that case stays diagnosable.
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

// AttemptedRefs returns the "provider:model" refs this result actually called
// upstream, ready to hand to RouteExcluding. Structural exclusions (gate refusals,
// disabled providers) are left out on purpose: those candidates were never tried,
// and for a described-image retry they are exactly the ones worth reaching.
func (res *Result) AttemptedRefs() map[string]bool {
	out := map[string]bool{}
	for _, att := range res.Attempts {
		if att.ErrorOrigin == "upstream" && att.Provider != "" && att.Model != "" {
			out[att.Provider+":"+att.Model] = true
		}
	}
	return out
}

// Router picks candidates and drives fallback for one request.
type Router struct {
	cfg     *config.Config
	gate    *catalog.Gate
	client  *provider.Client
	pMu     sync.Mutex
	pickers map[string]*provider.KeyPicker
}

// NewRouter wires the router. client may be nil — in that case a
// TTFB-bounded client is built from the config's timeout_s setting,
// so per-attempt timeouts cover only the wait for response headers
// (first token), not the full body stream.
func NewRouter(cfg *config.Config, gate *catalog.Gate, client *provider.Client) *Router {
	if client == nil {
		ttfb := time.Duration(cfg.GetFallback().TimeoutS) * time.Second
		client = provider.NewClientWithTTFB(ttfb)
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

// Route runs the candidates through the capability gate, then attempts each in
// order with per-key rotation. Every non-200 status, plus transport errors and
// TTFB timeouts, rotates onward to the next key and then the next candidate; no
// status terminates the chain early. On success, Result.Resp is a live upstream
// response the caller must close.
//
// The model field in payload controls routing mode:
//   - "auto"/empty → pool classifier entries (tier-ordered if configured)
//   - "chain:fast" → named chain from config
//   - "a:x,b:y" → inline chain
//   - "a:x" → single model, skip classifier
func (r *Router) Route(ctx context.Context, pool string, payload map[string]any, hasImage, hasAudio, hasVideo bool, minContext int) (*Result, error) {
	return r.RouteExcluding(ctx, pool, payload, hasImage, hasAudio, hasVideo, minContext, nil)
}

// RouteExcluding is Route with a set of "provider:model" refs to leave out of the
// candidate list entirely. Skipped refs produce no attempt record, since a caller
// only excludes what it has already tried and logged.
//
// The describe-hop retry uses this: the candidates whose pixel attempts just
// failed are not worth re-treading as text, and doing so would add a full
// per-attempt timeout per key to an error path that is already slow.
func (r *Router) RouteExcluding(ctx context.Context, pool string, payload map[string]any, hasImage, hasAudio, hasVideo bool, minContext int, skip map[string]bool) (*Result, error) {
	res := &Result{Pool: pool}
	entries := r.resolveEntries(pool, payload)
	if len(skip) > 0 {
		kept := make([]string, 0, len(entries))
		for _, ref := range entries {
			if !skip[strings.TrimSpace(ref)] {
				kept = append(kept, ref)
			}
		}
		entries = kept
	}
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

		pr := r.cfg.GetProviderRouting(pName, model)
		if !pr.OK || pr.BaseURL == "" {
			att := ap(pName, model, "", "router")
			att.Err = "provider has no base_url"
			continue
		}
		if !pr.Enabled {
			att := ap(pName, model, "", "router")
			att.Err = "provider disabled"
			continue
		}
		if pr.ModelDisabled {
			att := ap(pName, model, "", "router")
			att.Err = "model disabled"
			continue
		}

		// Per-model media policy runs *before* the catalog gate and outranks it.
		// An explicitly denied modality excludes the candidate outright; an
		// explicitly allowed one is masked out of the flags handed to the gate,
		// so the catalog cannot veto a model the operator knows has native
		// support. Modalities left on "auto" pass through untouched.
		gateImage, gateAudio, gateVideo, denied := pr.MediaPolicy.Decide(hasImage, hasAudio, hasVideo)
		if denied != "" {
			att := ap(pName, model, "", "router")
			att.Status = 0
			att.Err = "excluded: " + denied
			continue
		}

		// structural capability gate: hard exclusion, fail-open for unknowns
		needsTools := 0
		if toolsFromPayload(payload) {
			needsTools = 1
		}
		if ok, reason := r.gate.Check(ref, gateImage, gateAudio, gateVideo, minContext, needsTools); !ok {
			att := ap(pName, model, "", "router")
			att.Status = 0
			att.Err = "excluded: " + reason
			continue
		}

		keys := pr.Keys
		picker := r.pickerFor(pName, keys)

		// One attempt per key, then on to the next candidate. This bound lives
		// here rather than in the cooldown clock: a 5xx or a timeout no longer
		// parks a key (neither says anything about the key), so the loop has to
		// know for itself when it has run out of keys to try.
		tried := make(map[int]bool, len(keys))

		keyLoop:
		for {
			keyIdx, key, okk := picker.NextExcluding(time.Now(), tried)
			if !okk {
				if len(keys) == 0 {
					att := ap(pName, model, "", "router")
					att.Err = "no API key configured for provider"
				}
				break keyLoop
			}
			tried[keyIdx] = true
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
			if pr.HasLimit {
				if !payloadCopied {
					upstreamPayload = make(map[string]any, len(payload)+1)
					for k, v := range payload {
						upstreamPayload[k] = v
					}
					payloadCopied = true
				}
				if mt, exists := upstreamPayload["max_tokens"]; exists {
					reqMT, _ := mt.(float64)
					if int(reqMT) > pr.ModelLimit {
						upstreamPayload["max_tokens"] = pr.ModelLimit
					}
				} else {
					upstreamPayload["max_tokens"] = pr.ModelLimit
				}
			}
			// strip params this provider doesn't support (e.g. groq rejects
			// reasoning_effort). Drops them before forwarding upstream.
			if len(pr.StripParams) > 0 {
				if !payloadCopied {
					upstreamPayload = make(map[string]any, len(payload))
					for k, v := range payload {
						upstreamPayload[k] = v
					}
					payloadCopied = true
				}
				for _, sp := range pr.StripParams {
					delete(upstreamPayload, sp)
				}
			}

			// The per-attempt TTFB timeout is enforced at the transport
			// level (ResponseHeaderTimeout), not via context.WithTimeout.
			// This means once response headers arrive, the body can stream
			// freely under the parent context — the client's own timeout
			// governs overall request lifetime.
			start := time.Now()
		// Substitute per-provider templating in base_url. The supported
		// placeholder is "{account_id}", replaced by the provider's
		// AccountID field. A provider with no AccountID keeps "{account_id}"
		// literal (it will 404 upstream, surfacing a config gap loudly
		// rather than silently hitting the wrong account).
		upBaseURL := pr.BaseURL
		if pr.AccountID != "" && strings.Contains(upBaseURL, "{account_id}") {
			upBaseURL = strings.ReplaceAll(upBaseURL, "{account_id}", pr.AccountID)
		}
		resp, doErr := r.client.Do(ctx, &provider.Upstream{Name: pName, BaseURL: upBaseURL, Keys: keys, APIMode: pr.APIMode}, key, upstreamPayload)
		att.LatencyMs = int(time.Since(start).Milliseconds())

		if doErr != nil {
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
		case att.Status == 200:
			picker.MarkSuccess(keyIdx)
			res.Status = att.Status
			res.Resp = resp
			return res, nil
		default: // any non-200 (4xx, 5xx, 3xx) — record and fall back
			body = readErrBody(resp)
			att.Err = body
			ra := parseRetryAfter(resp.Header)
			picker.MarkFailure(keyIdx, time.Now(), ra, att.Status)
			resp.Body.Close()
			continue
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
