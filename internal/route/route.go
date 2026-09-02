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
	"crypto/sha256"
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

// statusClientGone marks a chain abandoned because the caller disconnected or
// its deadline passed, rather than because upstreams failed. It borrows nginx's
// 499 so the event log can tell the two apart at a glance — no response is
// delivered in this case either way.
const statusClientGone = 499

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

// maxBreakerEntries bounds the breaker map. Entries outlive their cooldown so
// failure streaks can accumulate, and a caller may name arbitrary model strings
// against a configured provider, so the cold ones are dropped past this size.
const maxBreakerEntries = 4096

// breakerState is one candidate's health: when it may next be dialed, and how
// many consecutive hard failures it has taken. The streak deliberately outlives
// `until` — that is what lets repeated failures escalate from the short
// cooldown to the long lockout — and only a 200 clears it.
type breakerState struct {
	until time.Time
	fails int
}

// Router picks candidates and drives fallback for one request.
type Router struct {
	cfg     *config.Config
	gate    *catalog.Gate
	client  *provider.Client
	pMu     sync.Mutex
	pickers map[string]*provider.KeyPicker
	// breakers is the provider-level circuit-breaker state: ref ("provider:model")
	// → the earliest time that candidate may be dialed again. A candidate that
	// fails with a transport error or 5xx is pulled out of rotation for
	// fallback.provider_cooldown_s instead of being re-dialed at the front of a
	// pool on every request, which is how one dead upstream used to add its
	// full per-attempt timeout to each request's path.
	bMu      sync.Mutex
	breakers map[string]*breakerState
	// nowFn is injectable for tests; nil means time.Now.
	nowFn func() time.Time
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
	return &Router{cfg: cfg, gate: gate, client: client, pickers: map[string]*provider.KeyPicker{}, breakers: map[string]*breakerState{}}
}

// now is the router's clock. Tests inject nowFn to drive breaker expiry without
// sleeping; production uses time.Now.
func (r *Router) now() time.Time {
	if r.nowFn != nil {
		return r.nowFn()
	}
	return time.Now()
}

// Status is a redacted snapshot of live key/breaker health for the dashboard.
func (r *Router) Status() map[string]any {
	now := r.now()
	r.pMu.Lock()
	cooling, dead, total := 0, 0, 0
	for _, p := range r.pickers {
		if p == nil {
			continue
		}
		total += p.KeyCount()
		cooling += p.CoolingCount(now)
		dead += p.DeadCount()
	}
	r.pMu.Unlock()
	r.bMu.Lock()
	breakers := 0
	for _, st := range r.breakers {
		if st != nil && now.Before(st.until) {
			breakers++
		}
	}
	r.bMu.Unlock()
	return map[string]any{
		"keysCooling":      cooling,
		"keysDead":         dead,
		"keyTotal":         total,
		"providersCooling": breakers,
	}
}

// breakerFor reports a candidate's cooldown state: when it may be dialed again
// and how long its current failure streak is. Expired entries are kept rather
// than deleted so the streak survives to escalate; clearBreaker and eviction
// are what remove them.
func (r *Router) breakerFor(ref string) (breakerState, bool) {
	r.bMu.Lock()
	defer r.bMu.Unlock()
	st, ok := r.breakers[ref]
	if !ok || !r.now().Before(st.until) {
		return breakerState{}, false
	}
	return *st, true
}

// tripBreaker records one hard failure against ref and takes it out of
// rotation. Early failures cost the short cooldown; once the streak reaches
// provider_failure_threshold the candidate is locked out for provider_lockout_s
// instead, so an upstream that is genuinely down stops being re-probed every
// cooldown window. Both windows at 0 disables the breaker.
//
// Called at most once per candidate per request: a provider with six keys that
// fails on all six has failed once, not six times.
func (r *Router) tripBreaker(ref string) {
	fb := r.cfg.GetFallback()
	if fb.ProviderCooldownS <= 0 && fb.ProviderLockoutS <= 0 {
		return
	}
	r.bMu.Lock()
	defer r.bMu.Unlock()
	if r.breakers == nil {
		r.breakers = map[string]*breakerState{}
	}
	st, ok := r.breakers[ref]
	if !ok {
		st = &breakerState{}
		r.breakers[ref] = st
		r.evictColdNoLock()
	}
	st.fails++
	window := fb.ProviderCooldownS
	if fb.ProviderFailureThreshold > 0 && fb.ProviderLockoutS > 0 && st.fails >= fb.ProviderFailureThreshold {
		window = fb.ProviderLockoutS
	}
	if window <= 0 {
		// The streak still counts, so escalation can fire even when the short
		// cooldown is switched off.
		return
	}
	st.until = r.now().Add(time.Duration(window) * time.Second)
}

// evictColdNoLock bounds the breaker map, dropping entries that are not
// currently cooling down. Caller holds bMu.
func (r *Router) evictColdNoLock() {
	if len(r.breakers) <= maxBreakerEntries {
		return
	}
	now := r.now()
	for k, st := range r.breakers {
		if !now.Before(st.until) {
			delete(r.breakers, k)
		}
	}
}

// breakerKeyName computes the circuit-breaker key for a candidate. For
// single-endpoint providers this is just "provider:model". For deployment-links
// providers (where each URL serves one model), the key incorporates a short
// hash of the base URL so each link gets its own cooldown state — otherwise a
// failure on one Modal deployment would cool every other deployment under the
// same provider.
func breakerKeyName(ref, baseURL string) string {
	if baseURL == "" {
		return ref
	}
	h := sha256.Sum256([]byte(baseURL))
	return ref + "@" + fmt.Sprintf("%x", h[:4])
}

// clearBreaker puts ref back in rotation after it answers 200.
func (r *Router) clearBreaker(ref string) {
	r.bMu.Lock()
	defer r.bMu.Unlock()
	delete(r.breakers, ref)
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
		return r.quotaOrdered(r.cfg.TierSortedEntries(pool))
	}

	// chain:<name> → named chain (explicit order; NOT tier-reordered)
	if strings.HasPrefix(model, "chain:") {
		name := strings.TrimPrefix(model, "chain:")
		if chain, ok := r.cfg.GetChain(name); ok {
			return chain
		}
		// unknown chain → fall back to pool (still tier-ordered)
		return r.quotaOrdered(r.cfg.TierSortedEntries(pool))
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
	return r.quotaOrdered(r.cfg.TierSortedEntries(pool))
}

// quotaOrdered reorders pool entries by their provider's remaining quota
// (see quota.go). Explicit chains and single-model refs bypass this — only
// pool-driven resolution consults the quota file, and only when one is set.
func (r *Router) quotaOrdered(entries []string) []string {
	data := quotaState()
	if data == nil {
		return entries
	}
	return QuotaSortedEntries(entries, data, r.now())
}

// poolContextFloor returns the advertised context ceiling for a pool, or 0 when
// the pool has no override. The advertised ceiling is what the /v1/models
// endpoint exposes (pool_context in config); when set, candidates whose real
// context is below it are skipped in RouteExcluding so a pool that promises
// 512k/1m never silently routes to a 128k model.
func (r *Router) poolContextFloor(pool string) int {
	return r.cfg.GetPoolContext()[pool]
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

// RouteRaw is Route with the protective gates disabled: circuit breaker,
// capability gate, media policy, and context floor are all bypassed so a
// probe reaches the exact candidate even if the router considers it dead,
// exhausted, or too small. Intended for the dashboard playground's raw
// mode — a one-off diagnostic dial — never for production traffic.
func (r *Router) RouteRaw(ctx context.Context, pool string, payload map[string]any, hasImage, hasAudio, hasVideo bool, minContext int) (*Result, error) {
	return r.RouteExcluding(ctx, pool, payload, hasImage, hasAudio, hasVideo, minContext, nil, true)
}

// RouteExcluding is Route with a set of "provider:model" refs to leave out of the
// candidate list entirely. Skipped refs produce no attempt record, since a caller
// only excludes what it has already tried and logged.
//
// The describe-hop retry uses this: the candidates whose pixel attempts just
// failed are not worth re-treading as text, and doing so would add a full
// per-attempt timeout per key to an error path that is already slow.
//
// raw disables the router's protective exclusions (circuit breaker, capability
// gate, media policy, context floor) so every remaining candidate is dialed
// even when the router believes it is dead, disabled, or too small. Playground
// diagnostics only; production callers pass false.
func (r *Router) RouteExcluding(ctx context.Context, pool string, payload map[string]any, hasImage, hasAudio, hasVideo bool, minContext int, skip map[string]bool, raw ...bool) (*Result, error) {
	rawMode := len(raw) > 0 && raw[0]
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

	// Provider circuit breaker. A candidate that recently failed with a
	// transport error or a 5xx is skipped instead of being re-dialed at the
	// front of the pool, so one dead upstream stops charging its full TTFB
	// timeout to every request that passes through. Skipped candidates still
	// get an "excluded" attempt record, so the decision trail stays complete.
	//
	// Fail open: if every candidate is cooling down, honor none of them. A pool
	// in which everything tripped must still be tried, or one bad minute turns
	// into a hard 503 for the length of the cooldown.
	// The breaker key incorporates the base URL hash so deployment-links
	// providers get per-link cooldown (one dead Modal deployment doesn't cool
	// the others). We resolve each entry's base URL up front. Raw mode skips
	// the breaker entirely — it dials even cooling candidates.
	type coolingEntry struct {
		key   string
		state breakerState
	}
	cooling := map[string]coolingEntry{}
	if !rawMode {
		for _, ref := range entries {
			trimmed := strings.TrimSpace(ref)
			// Resolve base URL for this entry to compute the composite breaker key.
			cName, cModel, _ := r.cfg.Resolve(trimmed)
			var baseURL string
			if cName != "" {
				baseURL = r.cfg.GetProviderRouting(cName, cModel).BaseURL
			}
			bk := breakerKeyName(trimmed, baseURL)
			if st, isOpen := r.breakerFor(bk); isOpen {
				cooling[trimmed] = coolingEntry{key: bk, state: st}
			}
		}
		if len(cooling) == len(entries) {
			cooling = map[string]coolingEntry{}
		}
	}

	attemptTimeout := time.Duration(r.cfg.GetFallback().TimeoutS) * time.Second
	ap := func(providerName, model, keyID string, origin string) *Attempt {
		res.Attempts = append(res.Attempts, Attempt{})
		att := &res.Attempts[len(res.Attempts)-1]
		att.Seq = len(res.Attempts)
		att.Provider, att.Model, att.KeyID, att.ErrorOrigin = providerName, model, keyID, origin
		return att
	}

candidateLoop:
	for _, ref := range entries {
		// Nothing to answer to: the caller hung up (or its deadline passed)
		// before we got here. Every remaining candidate would be dialed for a
		// response that can never be delivered.
		if ctx.Err() != nil {
			res.Status = statusClientGone
			return res, ErrExhausted
		}
		refKey := strings.TrimSpace(ref)
		if ce, isCooling := cooling[refKey]; isCooling {
			cName, cModel, cErr := r.cfg.Resolve(ref)
			if cErr != nil {
				cName, cModel = ref, ""
			}
			att := ap(cName, cModel, "", "router")
			left := ce.state.until.Sub(r.now()).Round(time.Second)
			if th := r.cfg.GetFallback().ProviderFailureThreshold; th > 0 && ce.state.fails >= th {
				att.Err = fmt.Sprintf("excluded: %d consecutive failures, locked out for another %s", ce.state.fails, left)
			} else {
				att.Err = fmt.Sprintf("excluded: cooling down after a recent failure, retryable in %s", left)
			}
			continue
		}
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
		if denied != "" && !rawMode {
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
		if ok, reason := r.gate.Check(ref, gateImage, gateAudio, gateVideo, minContext, needsTools); !ok && !rawMode {
			att := ap(pName, model, "", "router")
			att.Status = 0
			att.Err = "excluded: " + reason
			continue
		}

		// Context-floor: if the pool advertises a context ceiling (pool_context
		// override) and this candidate's real context is below it, skip it so a
		// small-context model never silently serves a pool that promises more.
		// This is what makes a 512k/1m advertised pool actually route only to
		// models that can hold that much — the rest fall through to the next
		// candidate. Unknown models (real ctx 0) are kept: they fail open.
		if floor := r.poolContextFloor(pool); floor > 0 && !rawMode {
			if realCtx := r.gate.ContextWindow(ref); realCtx > 0 && realCtx < floor {
				att := ap(pName, model, "", "router")
				att.Status = 0
				att.Err = fmt.Sprintf("excluded: context %d below pool floor %d", realCtx, floor)
				continue
			}
		}

		keys := pr.Keys
		picker := r.pickerFor(pName, keys)

		// One attempt per key, then on to the next candidate. This bound lives
		// here rather than in the cooldown clock: a 5xx or a timeout no longer
		// parks a key (neither says anything about the key), so the loop has to
		// know for itself when it has run out of keys to try.
		tried := make(map[int]bool, len(keys))
		// breakerKey is set inside the keyLoop for each attempt; we track
		// the last one so tripBreaker can use it after the key rotation is done.
		var lastBreakerKey string
		// Set by any hard failure below. One trip is recorded for the candidate
		// once its keys are exhausted, so a six-key provider that fails on all
		// six counts as one failure in the streak, not six.
		candidateFailed := false

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
				// Deployment-as-model: pr.UpstreamModel carries the real
				// upstream id when the pool-facing model slot named a
				// deployment ("Modal:glm53" -> "glm-5.3-flash").
				if pr.UpstreamModel != "" {
					upstreamPayload["model"] = pr.UpstreamModel
				} else {
					upstreamPayload["model"] = model
				}
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
			// Some DeepSeek-family gateways reject an assistant tool-call turn
			// that arrives without reasoning_content. Put a placeholder back so
			// multi-turn tool conversations survive the second hop. This returns
			// a fresh payload rather than editing in place — the map is shared
			// with every other candidate in the chain.
			if pr.RepairReasoningContent {
				if repaired, changed := repairReasoningContent(upstreamPayload); changed {
					upstreamPayload = repaired
					payloadCopied = true
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
			// Composite breaker key: ref + base URL hash. For deployment-links
			// providers this gives each link its own cooldown state.
			lastBreakerKey = breakerKeyName(refKey, upBaseURL)
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
				// Distinguish "the client left" from "this upstream failed". Only
				// the latter is worth falling back over; the former used to walk
				// the entire chain, dialing every provider in the pool on behalf
				// of a request that had already gone away.
				if ctx.Err() != nil {
					res.Status = statusClientGone
					break candidateLoop
				}
				// A dial error or TTFB timeout is the upstream being unhealthy, not
				// this key or this request — cool the candidate off once the whole
				// key rotation is done.
				candidateFailed = true
				continue
			}

			att.Status = resp.StatusCode
			var body string
			switch {
			case att.Status == 200:
				picker.MarkSuccess(keyIdx)
				// Success ends the streak outright, not just the current window.
				r.clearBreaker(lastBreakerKey)
				res.Status = att.Status
				res.Resp = resp
				return res, nil
			case isTransientStatus(att.Status) && !streamRequested(payload):
				// A single 502/503/504 is often a blip (LB hiccup, brief deploy),
				// not a verdict on the candidate. Retry the SAME key in place —
				// bounded, jittered, never for streamed requests (the body may
				// have started arriving) and never when upstream sent an explicit
				// Retry-After. Only after the in-place retries are spent does the
				// attempt fall through to normal fallback handling.
				retried := false
				for i := 0; i < RetryTransientMax(r.cfg.GetFallback()); i++ {
					if ra := parseRetryAfter(resp.Header); ra > 0 {
						break // upstream told us when to come back; respect it
					}
					if ctx.Err() != nil {
						break // client gone mid-retry
					}
					time.Sleep(transientBackoff(i))
					resp2, err2 := r.client.Do(ctx, &provider.Upstream{Name: pName, BaseURL: upBaseURL, Keys: keys, APIMode: pr.APIMode}, key, upstreamPayload)
					resp.Body.Close()
					if err2 != nil {
						att.Err = describeAttemptError(err2, attemptTimeout, ctx)
						break // treat as a plain failure; key loop continues below
					}
					att.LatencyMs = int(time.Since(start).Milliseconds())
					att.Status = resp2.StatusCode
					resp = resp2
					retried = true
					if resp.StatusCode == 200 || !isTransientStatus(resp.StatusCode) {
						break
					}
				}
				if retried && att.Status == 200 {
					picker.MarkSuccess(keyIdx)
					r.clearBreaker(lastBreakerKey)
					res.Status = att.Status
					res.Resp = resp
					return res, nil
				}
				fallthrough
			default: // any non-200 (4xx, 5xx, 3xx) — record and fall back
				body = readErrBody(resp)
				att.Err = body
				ra := parseRetryAfter(resp.Header)
				picker.MarkFailure(keyIdx, time.Now(), ra, att.Status)
				resp.Body.Close()
				// 5xx is the upstream failing; 4xx is usually about this request or
				// this key, and the next request may well succeed. Only the former
				// is worth pulling the candidate out of rotation for.
				if att.Status >= 500 {
					candidateFailed = true
				}
				continue
			}
		}
		// One trip per candidate per request, recorded after its keys are spent,
		// so the failure streak counts requests rather than key rotations.
		if candidateFailed {
			r.tripBreaker(lastBreakerKey)
		}
	}

	if res.Status != statusClientGone {
		res.Status = 503
	}
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
