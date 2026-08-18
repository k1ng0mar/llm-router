// Package server exposes the OpenAI-compatible endpoint, the admin API, and
// the dashboard. The endpoint's contract: a provider 429/5xx is a fallback
// trigger and never becomes the caller's error.
package server

import (
	"bytes"
	"context"
	"crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"llm-router/internal/catalog"
	"llm-router/internal/classify"
	"llm-router/internal/config"
	"llm-router/internal/provider"
	"llm-router/internal/route"
	"llm-router/internal/store"
)

//go:embed web/index.html
var dashboardHTML string

//go:embed web/logos/*
var logoFS embed.FS

//go:embed web/icon-192.png web/icon-512.png
var iconFS embed.FS

// vendorFS embeds the vendored dashboard assets (Tailwind, the web fonts, and
// the Material Symbols icon font) so the dashboard is fully self-contained and
// works offline — no CDN dependency, which is the point of a single-binary
// self-hosted tool.
//
//go:embed web/vendor
var vendorFS embed.FS

// maxStoredBody caps how many bytes of a request/response body we persist to
// the SQLite event log per request. Media parts are also redacted (see
// redactPayloadForLog), but this is a hard backstop against unbounded growth.
const maxStoredBody = 256 * 1024

// vendorSubFS is the vendored tree rooted at web/vendor, served at /vendor/*.
var vendorSubFS fs.FS

func init() {
	v, err := fs.Sub(vendorFS, "web/vendor")
	if err != nil {
		panic(fmt.Sprintf("vendor sub: %v", err))
	}
	vendorSubFS = v
}

// Server wires handlers to the router, store, config, and catalog gate.
type Server struct {
	cfg    *config.Config
	store  *store.Store
	router *route.Router
	client *provider.Client
	gate   *catalog.Gate
}

// New builds a server. client and gate may be nil.
func New(cfg *config.Config, st *store.Store, r *route.Router, g *catalog.Gate) *Server {
	if g == nil {
		g = catalog.NewGate(nil)
	}
	// The describe hop gets the same per-attempt TTFB bound as the main path;
	// http.DefaultClient has no timeout at all.
	ttfb := time.Duration(cfg.GetFallback().TimeoutS) * time.Second
	return &Server{cfg: cfg, store: st, router: r, client: provider.NewClientWithTTFB(ttfb), gate: g}
}

// Handler returns the multiplexer for the whole HTTP surface.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	mux.HandleFunc("GET /v1/models", s.handleListModels)
	mux.HandleFunc("GET /api/requests", s.handleRequests)
	mux.HandleFunc("GET /api/requests/{id}", s.handleRequestDetail)
	mux.HandleFunc("GET /api/stats", s.handleStats)
	mux.HandleFunc("GET /api/usage/models", s.handleModelUsage)
	mux.HandleFunc("GET /api/config", s.handleGetConfig)
	mux.HandleFunc("POST /api/config/default", s.handleSetDefault)
	mux.HandleFunc("POST /api/config/pools", s.handleSetPool)
	mux.HandleFunc("POST /api/config/providers", s.handleSetProvider)
	mux.HandleFunc("PUT /api/config/providers/{name}", s.handleUpdateProvider)
	mux.HandleFunc("POST /api/config/keys", s.handleSetKeys)
	mux.HandleFunc("POST /api/config/toggle", s.handleToggleProvider)
	mux.HandleFunc("POST /api/config/models/toggle", s.handleToggleModel)
	mux.HandleFunc("POST /api/config/chains", s.handleSetChain)
	mux.HandleFunc("DELETE /api/config/chains", s.handleRemoveChain)
	mux.HandleFunc("POST /api/config/tiers", s.handleSetTier)
	mux.HandleFunc("GET /logos/", s.handleLogo)
	mux.Handle("GET /vendor/", http.StripPrefix("/vendor/", http.FileServer(http.FS(vendorSubFS))))
	mux.HandleFunc("GET /api/providers/{name}/models", s.handleProviderModels)
	mux.HandleFunc("GET /api/providers/{name}/cached-models", s.handleGetCachedModels)
	mux.HandleFunc("POST /api/providers/{name}/custom-model", s.handleAddCustomModel)
	mux.HandleFunc("DELETE /api/providers/{name}/custom-model", s.handleRemoveCustomModel)
	mux.HandleFunc("GET /api/providers/{name}/test", s.handleProviderTest)
	// Catch-alls for the API namespaces. Without these, an unmatched request
	// under /v1/ or /api/ falls through to the `GET /` dashboard catch-all
	// below and an OpenAI-compatible client's probe receives 200 text/html
	// instead of a JSON error — GET /v1/chat/completions was exactly that.
	// Registered routes are more specific and always win. A known path hit with
	// the wrong method (e.g. GET on the POST-only chat endpoint) lands on the
	// /v1/ catch-all, which turns it into a 405 (handleOpenAPIUnknown).
	mux.HandleFunc("GET /v1/", s.handleOpenAPIUnknown)
	mux.HandleFunc("POST /v1/", s.handleOpenAPIUnknown)
	mux.HandleFunc("GET /api/", s.handleAdminUnknown)
	mux.HandleFunc("POST /api/", s.handleAdminUnknown)
	mux.HandleFunc("PUT /api/", s.handleAdminUnknown)
	mux.HandleFunc("DELETE /api/", s.handleAdminUnknown)
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		io.WriteString(w, dashboardHTML)
	})
	mux.HandleFunc("GET /icon-192.png", serveIcon("icon-192.png"))
	mux.HandleFunc("GET /icon-512.png", serveIcon("icon-512.png"))
	return mux
}

func serveIcon(path string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		data, err := iconFS.ReadFile("web/" + path)
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "image/png")
		w.Write(data)
	}
}

// handleListModels answers the OpenAI /v1/models call with the router's pools,
// plus "auto" and "router" for "you pick". These are exactly the values a client
// may put in the model field, so an OpenAI-compatible client's model picker
// offers real routing choices instead of a list of upstream model ids the router
// would ignore.
//
// Without this route the mux's catch-all GET / served the dashboard HTML here —
// 142KB of text/html under a 200, which a client's model discovery would either
// fail to parse or, worse, misparse.
func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	out := []model{
		{ID: "router", Object: "model", OwnedBy: "llm-router"},
		{ID: "auto", Object: "model", OwnedBy: "llm-router"},
	}
	pools := make([]string, 0, len(s.cfg.GetPools()))
	for name := range s.cfg.GetPools() {
		pools = append(pools, name)
	}
	sort.Strings(pools) // stable ordering so a client's cache doesn't churn
	for _, name := range pools {
		out = append(out, model{ID: name, Object: "model", OwnedBy: "llm-router"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": out})
}

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	io.WriteString(w, "ok")
}

func (s *Server) authorized(r *http.Request) bool {
	if s.cfg.RouterKey != "" {
		h := strings.TrimSpace(r.Header.Get("Authorization"))
		// Case-insensitive scheme, exact comparison of the secret.
		h = strings.TrimPrefix(strings.ToLower(h), "bearer ")
		return h == s.cfg.RouterKey
	}
	// No router key configured: a request is only permitted when the operator
	// has explicitly opted in. An empty key must NEVER silently open everything.
	return s.cfg.InsecureNoAuth
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("writeJSON: %v", err)
	}
}

// v1EndpointMethods maps the known OpenAI endpoints to the method they accept.
// Because the catch-alls are registered with method prefixes (a method-less
// /v1/ would conflict with the GET / dashboard catch-all), the mux can't answer
// a wrong-method call on a known endpoint by itself — a GET /v1/chat/completions
// lands here matching GET /v1/. This table turns that into a 405.
var v1EndpointMethods = map[string]string{
	"/v1/chat/completions": http.MethodPost,
	"/v1/models":           http.MethodGet,
}

// handleOpenAPIUnknown answers any /v1/* path that is not a registered route:
// 405 for a known endpoint hit with the wrong method (e.g. GET on the POST-only
// chat endpoint), 404 otherwise. Either way the caller gets JSON, not the
// dashboard HTML the `GET /` catch-all would otherwise serve.
func (s *Server) handleOpenAPIUnknown(w http.ResponseWriter, r *http.Request) {
	if m, ok := v1EndpointMethods[r.URL.Path]; ok && r.Method != m {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": map[string]string{"message": "method not allowed, expected " + m}})
		return
	}
	writeJSON(w, http.StatusNotFound, map[string]any{"error": map[string]string{"message": "not found"}})
}

// handleAdminUnknown answers unknown /api/* paths with a JSON 404 instead of
// the dashboard HTML (see handleOpenAPIUnknown).
func (s *Server) handleAdminUnknown(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotFound, map[string]any{"error": map[string]string{"message": "not found"}})
}

// handleChat is the OpenAI-compatible entry point.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}

	var payload map[string]any
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<20)).Decode(&payload); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{"message": "invalid json: " + err.Error()}})
		return
	}
	// capture the original request body (JSON sent to /v1/chat/completions) for
	// the request detail view in the dashboard. Media parts are redacted so we
	// don't persist megabytes of base64 in the event log.
	redacted := redactPayloadForLog(payload)
	reqBodyJSON, _ := json.Marshal(redacted)
	if len(reqBodyJSON) > maxStoredBody {
		reqBodyJSON = reqBodyJSON[:maxStoredBody]
	}

	messages, _ := payload["messages"].([]any)
	hint := r.Header.Get("X-Route-Pool")
	r.Header.Del("X-Route-Pool") // must not leak upstream (spec: header hygiene)
	if hint != "" {
		if _, ok := s.cfg.GetPools()[hint]; !ok {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{"message": fmt.Sprintf("unknown pool %q in X-Route-Pool", hint)}})
			return
		}
	}
	// One pass decides the pool and reports every modality present. Requests
	// carrying media and matching no keyword heuristic land in the media pool
	// (images, audio and video alike) rather than the plain default pool.
	// A bare pool name in the model field selects that pool, exactly like the
	// X-Route-Pool header (the header wins when both are set). This is what the
	// docs have always promised, and without it the model field silently fell
	// through to the classifier's choice — so any client that can only set
	// `model`, which is most of them, could never steer routing at all.
	//
	// Only a name that actually matches a pool is treated this way. Anything
	// else — "auto", a chain, a provider:model ref, a bare model id — is left
	// for resolveEntries to interpret as before.
	modelPool := false
	if hint == "" {
		if m, _ := payload["model"].(string); m != "" {
			if _, ok := s.cfg.GetPools()[strings.TrimSpace(m)]; ok {
				hint = strings.TrimSpace(m)
				modelPool = true
			}
		}
	}
	pool, rule, _, media := classify.PoolForMedia(s.cfg.GetClassifierHeuristics(), messages, s.cfg.GetDefault(), hint, s.cfg.GetMediaPool())
	if modelPool && rule == "hint" {
		rule = "model-pool" // distinguish it from a header hint in the log
	}
	hasImage, hasAudio, hasVideo := media.Image, media.Audio, media.Video
	streaming, _ := payload["stream"].(bool)
	// Vision chain. When an image lands in a pool whose models can't read
	// pixels, an image-capable model describes it first and the original pool
	// answers with that description folded into the question. Applies to any
	// pool, not a hardcoded pair: an image in the creative pool used to 503
	// without the hop ever being tried.
	//
	// Pixels are kept as-is when any of these hold:
	//   - the target pool IS the media pool (that's where describers live)
	//   - allow_direct_vision is on and the pool's first model can see images
	//   - nothing image-capable is configured to describe with, in which case
	//     the gate should have the final say rather than a synthetic 400
	//
	// directPath records that we sent pixels onward, so an exhausted chain can
	// still retry via the describe hop below.
	directPath := false
	if hasImage {
		canDescribe := len(s.visionRefs()) > 0
		directOK := s.cfg.GetAllowDirectVision() && s.poolFirstEntryIsVision(pool, hasImage)
		if pool == s.cfg.GetMediaPool() || directOK || !canDescribe {
			directPath = canDescribe
			if _, ok := payload["messages"]; !ok {
				payload["messages"] = messages
			}
		} else {
			desc, err := s.describeImages(r.Context(), payload, messages)
			if err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{"message": err.Error()}})
				return
			}
			payload["messages"] = describedMessages(messages, desc)
			hasImage = false // pixels consumed; the pool sees text only
		}
	}
	if streaming {
		payload["stream"] = true
	}

	id := newRequestID()
	start := time.Now()
	minCtx := estimateTokens(payload)
	res, err := s.router.Route(r.Context(), pool, payload, hasImage, hasAudio, hasVideo, minCtx)

	// Second chance for the direct-pixel path: every pixel-capable candidate
	// failed, but the pool may hold text-only models the gate had to exclude.
	// Describe the image and retry the same pool as a text request rather than
	// returning a 503 the describe hop would have avoided. The first pass's
	// attempts are kept ahead of the retry's, so the trail shows both.
	if err != nil && directPath && hasImage {
		if desc, dErr := s.describeImages(r.Context(), payload, messages); dErr == nil {
			payload["messages"] = describedMessages(messages, desc)
			hasImage = false
			firstPass := res.Attempts
			// Skip the candidates whose pixel attempts just failed: as a text
			// request they would fail the same way, and each re-tread costs a
			// full per-attempt timeout per key. The gate-excluded candidates are
			// deliberately NOT skipped — those text-only models are the whole
			// point of the retry.
			res2, err2 := s.router.RouteExcluding(r.Context(), pool, payload, false, hasAudio, hasVideo,
				estimateTokens(payload), res.AttemptedRefs())
			if res2 != nil {
				// Keep both passes in the trail whether or not the retry
				// succeeded, so a 503 still shows why the pixels failed first.
				// The retry numbers its attempts from 1 again, so continue the
				// sequence — otherwise the dashboard sorts the two passes into
				// each other and the order reads as nonsense.
				for i := range res2.Attempts {
					res2.Attempts[i].Seq += len(firstPass)
				}
				res2.Attempts = append(firstPass, res2.Attempts...)
				res, err = res2, err2
				rule += "+described"
			}
		}
	}

	logReq := func(status int, prov, model string, attempts []route.Attempt, respBody string, promptTok, completionTok int) {
		// error_origin at the request level mirrors the last attempt's origin
		// so dashboard rows show where a failure actually came from ("router"
		// gate/config refusal vs "upstream" provider rejection), instead of
		// always defaulting to "router".
		origin := "router"
		if n := len(attempts); n > 0 && attempts[n-1].ErrorOrigin != "" {
			origin = attempts[n-1].ErrorOrigin
		}
		row := &store.RequestRow{
			ID:            id,
			TS:            time.Now().UTC().Format(time.RFC3339),
			TsUnix:        time.Now().Unix(),
			Pool:          pool,
			Rule:          rule,
			FinalStatus:   status,
			FinalProvider: prov,
			FinalModel:    model,
			TotalMs:       int(time.Since(start).Milliseconds()),
			PromptTokens:  promptTok,
			CompletionTok: completionTok,
			ErrorOrigin:   origin,
			RequestBody:   string(reqBodyJSON),
			ResponseBody:  respBody,
		}
		if err := s.store.AddRequest(row); err != nil {
			log.Printf("store.AddRequest: %v", err)
		}
		for _, a := range attempts {
			if err := s.store.AddAttempt(&store.AttemptRow{
				RequestID: id, Seq: a.Seq, Provider: a.Provider, Model: a.Model, KeyID: a.KeyID,
				Status: a.Status, LatencyMs: a.LatencyMs, Err: a.Err, ErrorOrigin: a.ErrorOrigin,
			}); err != nil {
				log.Printf("store.AddAttempt: %v", err)
			}
		}
	}

	if err != nil {
		if res != nil && res.Status >= 400 && res.Status < 500 {
			// terminal client error — never fallback past it
			last := lastAttempt(res)
			// capture the upstream response body for debugging
			respBody := ""
			if res.Resp != nil {
				b, _ := io.ReadAll(io.LimitReader(res.Resp.Body, 65536))
				res.Resp.Body.Close()
				respBody = string(b)
			} else if last.Err != "" {
				// response body was captured in the attempt's Err field
				respBody = last.Err
			}
			logReq(res.Status, last.Provider, last.Model, res.Attempts, respBody, 0, 0)
			writeJSON(w, res.Status, map[string]any{"error": map[string]string{"message": err.Error(), "request_id": id}})
			return
		}
		// exhausted — all attempts failed
		var capturedBody string
		if res != nil && res.Resp != nil {
			b, _ := io.ReadAll(io.LimitReader(res.Resp.Body, 65536))
			res.Resp.Body.Close()
			capturedBody = string(b)
		} else if last := lastAttempt(res); last.Err != "" {
			// all providers failed without a shared response — capture the last error
			capturedBody = last.Err
		}
		logReq(503, "", "", res.Attempts, capturedBody, 0, 0)
		w.Header().Set("X-Router-Fallback-Exhausted", "true")
		w.Header().Set("X-Router-Request-Id", id)
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{
			"error": map[string]string{"message": err.Error(), "request_id": id},
		})
		return
	}

	defer res.Resp.Body.Close()
	last := lastAttempt(res)

	// copy upstream headers, stripping hop-by-hop and transport-level headers
	// that Go's http.Server manages itself or that cause passthrough issues
	skipHeaders := map[string]bool{
		"Content-Length":      true,
		"Transfer-Encoding":   true,
		"Connection":          true,
		"Server":              true,
		"Date":                true,
		"Set-Cookie":          true, // don't leak upstream cookies to router clients
		"Content-Encoding":    true, // Go transport already decoded unless we re-encode
	}
	for k := range res.Resp.Header {
		if skipHeaders[strings.ToLower(k)] || skipHeaders[k] {
			continue
		}
		w.Header().Set(k, res.Resp.Header.Get(k))
	}
	w.Header().Set("X-Router-Request-Id", id)
	w.Header().Set("X-Router-Pool", pool)
	w.Header().Set("X-Router-Rule", rule)
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	if streaming {
		// SSE passthrough: forward each chunk in real time while capturing
		// the body for the dashboard log. The TeeReader lets us read once
		// — the bytes flow to the client and accumulate in the buffer
		// simultaneously.
		var captured bytes.Buffer
		tee := io.TeeReader(io.LimitReader(res.Resp.Body, maxStoredBody), &captured)
		buf := make([]byte, 4096)
		for {
			n, err := tee.Read(buf)
			if n > 0 {
				w.Write(buf[:n])
				if flusher != nil {
					flusher.Flush()
				}
			}
			if err != nil {
				break
			}
		}
		spt, sct := sseUsageTokens(captured.String())
		logReq(res.Status, last.Provider, last.Model, res.Attempts, captured.String(), spt, sct)
	} else {
		// Non-streaming: buffer the body to log token usage and response.
		respBodyBytes, _ := io.ReadAll(io.LimitReader(res.Resp.Body, maxStoredBody))
		var pt, ct int
		var respJSON struct {
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal(respBodyBytes, &respJSON) == nil {
			pt = respJSON.Usage.PromptTokens
			ct = respJSON.Usage.CompletionTokens
		}
		logReq(res.Status, last.Provider, last.Model, res.Attempts, string(respBodyBytes), pt, ct)
		io.Copy(w, bytes.NewReader(respBodyBytes))
		if flusher != nil {
			flusher.Flush()
		}
	}
}

func lastAttempt(res *route.Result) route.Attempt {
	if len(res.Attempts) == 0 {
		return route.Attempt{}
	}
	return res.Attempts[len(res.Attempts)-1]
}

// sseUsageTokens extracts prompt/completion token counts from a captured SSE
// body. Streaming responses carry the final usage chunk as
// `data: {"usage":{"prompt_tokens":N,"completion_tokens":M}}` (OpenAI-style
// stream_options.include_usage, or the last delta before [DONE]). We scan the
// last few lines for the richest usage object — typically the final chunk —
// rather than parsing every SSE frame. Returns (0,0) if nothing is found.
func sseUsageTokens(body string) (int, int) {
	if body == "" {
		return 0, 0
	}
	var pt, ct int
	// Walk lines in reverse: the last usage-bearing frame wins.
	lines := strings.Split(body, "\n")
	for i := len(lines) - 1; i >= 0 && i >= len(lines)-8; i-- {
		line := strings.TrimSpace(lines[i])
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var frame struct {
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal([]byte(payload), &frame) == nil && frame.Usage != nil {
			pt = frame.Usage.PromptTokens
			ct = frame.Usage.CompletionTokens
			break
		}
	}
	return pt, ct
}

// refSeesImages reports whether one "provider:model" ref can accept pixels.
// The per-model media policy outranks the catalog: an explicit allow says the
// operator knows the model has native image support the catalog under-reports,
// an explicit deny keeps pixels away from it regardless of what the catalog
// claims, and "auto" defers to the gate (which fails open for unknown models).
func (s *Server) refSeesImages(ref string) bool {
	prov, model, err := s.cfg.Resolve(ref)
	if err != nil {
		return false
	}
	switch s.cfg.GetProviderRouting(prov, model).MediaPolicy.Stance("image") {
	case config.PolicyAllow:
		return true
	case config.PolicyDeny:
		return false
	}
	ok, _ := s.gate.Check(prov+":"+model, true, false, false, 0, 0)
	return ok
}

// poolFirstEntryIsVision checks if the first entry in the pool can take pixels.
// Used by the dual-path vision logic to decide whether to send pixels directly
// or go through the describe→code chain.
func (s *Server) poolFirstEntryIsVision(pool string, hasImage bool) bool {
	if !hasImage {
		return false
	}
	entries := s.cfg.GetPools()[pool]
	if len(entries) == 0 {
		return false
	}
	return s.refSeesImages(strings.TrimSpace(entries[0]))
}

// visionRefs returns the ordered list of image-capable model refs to use for the
// describe hop, preferring the `media` pool, then a legacy `vision` pool, then
// the top-level `vision:` list (kept for configs/tests that only populate it).
//
// The candidate list is filtered to refs that can actually see images. That
// filter is what makes a single mixed-modality `media` pool safe here: the pool
// may hold audio- or video-only models, and handing pixels to one of those would
// waste a hop on a guaranteed failure.
func (s *Server) visionRefs() []string {
	var candidates []string
	pools := s.cfg.GetPools()
	for _, name := range []string{"media", "vision"} {
		if v := pools[name]; len(v) > 0 {
			candidates = v
			break
		}
	}
	if len(candidates) == 0 {
		candidates = s.cfg.GetVision()
	}
	refs := make([]string, 0, len(candidates))
	for _, ref := range candidates {
		if ref = strings.TrimSpace(ref); ref != "" && s.refSeesImages(ref) {
			refs = append(refs, ref)
		}
	}
	return refs
}

// describeImages runs the fixed vision hop: extract a description of the
// attached images so the code pool never receives pixels. It tries each
// configured vision provider in turn (skipping disabled / keyless ones) and
// returns the first successful description.
func (s *Server) describeImages(ctx context.Context, payload map[string]any, messages []any) (string, error) {
	refs := s.visionRefs()
	if len(refs) == 0 {
		return "", errors.New("no image-capable model configured for image+code requests (add one to the `media` pool, or set image: allow on a model whose catalog entry under-reports vision)")
	}
	hopMessages := append(messages, map[string]any{
		"role":    "user",
		"content": "Describe the image(s) in this message in precise detail: subjects, layout, text, and anything relevant to writing code about them. Output only the description.",
	})
	var lastErr error
	for _, ref := range refs {
		pName, model, err := s.cfg.Resolve(ref)
		if err != nil {
			lastErr = err
			continue
		}
		up, ok := s.cfg.GetProvider(pName)
		if !ok || up.BaseURL == "" || !up.IsEnabled() {
			lastErr = fmt.Errorf("vision provider %q unavailable", pName)
			continue
		}
		key := ""
		if len(up.Keys) > 0 {
			key = up.Keys[0]
		}
		hop := map[string]any{
			"model":    model,
			"messages": hopMessages,
			"max_tokens": 2048,
			"stream":     false,
		}
		// Bound the vision hop even if the caller's context is open-ended.
		hopCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
		resp, err := s.client.Do(hopCtx, &provider.Upstream{Name: pName, BaseURL: up.BaseURL, Keys: up.Keys}, key, hop)
		cancel()
		if err != nil {
			lastErr = fmt.Errorf("vision hop to %s failed: %w", pName, err)
			continue
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("vision hop %s failed: HTTP %d: %s", pName, resp.StatusCode, truncate(string(body), 200))
			continue
		}
		var out struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &out); err != nil || len(out.Choices) == 0 || out.Choices[0].Message.Content == "" {
			lastErr = fmt.Errorf("vision hop %s returned no content", pName)
			continue
		}
		return out.Choices[0].Message.Content, nil
	}
	if lastErr == nil {
		lastErr = errors.New("vision hop failed on all configured vision providers")
	}
	return "", lastErr
}

// describedMessages folds a vision-pass description into the conversation: the
// pixels come out, every turn's text stays, and the description is attached to
// the final user turn so the model still sees the question it is meant to answer.
func describedMessages(messages []any, desc string) []any {
	return classify.AppendToLastUser(classify.StripMedia(messages),
		"\n\n[Image description from vision pass]\n"+desc)
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// redactPayloadForLog returns a copy of the request payload suitable for
// persistent logging: any image/audio/video content part is replaced with a
// placeholder so we never store megabytes of base64 media in the event log.
func redactPayloadForLog(p map[string]any) map[string]any {
	cp := make(map[string]any, len(p))
	for k, v := range p {
		if k == "messages" {
			if msgs, ok := v.([]any); ok {
				cp[k] = redactMessages(msgs)
				continue
			}
		}
		cp[k] = v
	}
	return cp
}

// redactMessages replaces media parts in a messages array with placeholders.
func redactMessages(msgs []any) []any {
	out := make([]any, len(msgs))
	for i, m := range msgs {
		mm, ok := m.(map[string]any)
		if !ok {
			out[i] = m
			continue
		}
		c, ok := mm["content"].([]any)
		if !ok {
			out[i] = m
			continue
		}
		parts := make([]any, len(c))
		for j, part := range c {
			pm, ok := part.(map[string]any)
			if !ok {
				parts[j] = part
				continue
			}
			t, _ := pm["type"].(string)
			if t == "image_url" || t == "input_audio" || t == "video_url" {
				parts[j] = map[string]any{"type": t, "redacted": "[media omitted from log]"}
				continue
			}
			parts[j] = part
		}
		nm := make(map[string]any, len(mm))
		for kk, vv := range mm {
			nm[kk] = vv
		}
		nm["content"] = parts
		out[i] = nm
	}
	return out
}

func estimateTokens(payload map[string]any) int {
	// Media-aware estimate: previously this marshaled the ENTIRE payload
	// (including base64 image/audio bytes) and divided by 4, which made a
	// single attached image inflate the estimate by tens of thousands of
	// "tokens" and structurally exclude every reasonable candidate at the
	// capability gate. Now we estimate from text content only and add a
	// small fixed allowance per media part. The gate is advisory — under-
	// estimating is safe (an over-long request falls back via the
	// context-window-400 path); over-estimating is what was unsafe.
	msgs, _ := payload["messages"].([]any)
	textLen := 0
	media := 0
	for _, m := range msgs {
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		switch c := mm["content"].(type) {
		case string:
			textLen += len(c)
		case []any:
			for _, part := range c {
				p, ok := part.(map[string]any)
				if !ok {
					continue
				}
				switch p["type"] {
				case "text":
					if s, ok := p["text"].(string); ok {
						textLen += len(s)
					}
				case "image_url", "input_audio", "video_url":
					media++
				}
			}
		}
	}
	// also count a system prompt if present as a string
	if sys, ok := payload["system"].(string); ok {
		textLen += len(sys)
	}
	return textLen/4 + media*256 + 1024 // text tokens + per-media allowance + slack
}

func newRequestID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return fmt.Sprintf("%d-%s", time.Now().UnixMilli(), hex.EncodeToString(b))
}

// --- admin API ---

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	days := 7
	if v := r.URL.Query().Get("range"); v != "" {
		switch v {
		case "24h":
			days = 1
		case "30d":
			days = 30
		case "all":
			days = 0
		}
	}
	st, err := s.store.Stats(days)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, st)
}

func (s *Server) handleModelUsage(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	days := 7
	if v := r.URL.Query().Get("range"); v != "" {
		switch v {
		case "24h":
			days = 1
		case "30d":
			days = 30
		case "all":
			days = 0
		}
	}
	rows, err := s.store.ModelUsage(days)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	if rows == nil {
		rows = []store.ModelUsageRow{}
	}
	writeJSON(w, http.StatusOK, rows)
}

// attDTO and reqDTO are the JSON shapes for the request list/detail endpoints.
// RequestBody/ResponseBody use omitempty so the list response (which strips
// them by default) stays small — the detail endpoint carries the full bodies.
type attDTO struct {
	Seq         int    `json:"seq"`
	Provider    string `json:"provider"`
	Model       string `json:"model"`
	KeyID       string `json:"key_id"`
	Status      int    `json:"status"`
	LatencyMs   int    `json:"latency_ms"`
	Err         string `json:"err"`
	ErrorOrigin string `json:"error_origin"`
}

type reqDTO struct {
	ID               string   `json:"id"`
	TS               string   `json:"ts"`
	Pool             string   `json:"pool"`
	Rule             string   `json:"rule"`
	FinalStatus      int      `json:"final_status"`
	FinalProvider    string   `json:"final_provider"`
	FinalModel       string   `json:"final_model"`
	PromptTokens     int      `json:"prompt_tokens"`
	CompletionTokens int      `json:"completion_tokens"`
	Cost             float64  `json:"cost"`
	TotalMs          int      `json:"total_ms"`
	ErrorOrigin      string   `json:"error_origin"`
	RequestBody      string   `json:"request_body,omitempty"`
	ResponseBody     string   `json:"response_body,omitempty"`
	Attempts         []attDTO `json:"attempts"`
}

func toRequestDTO(x store.RequestWithAttempts) reqDTO {
	d := reqDTO{ID: x.ID, TS: x.TS, Pool: x.Pool, Rule: x.Rule, FinalStatus: x.FinalStatus, FinalProvider: x.FinalProvider, FinalModel: x.FinalModel, PromptTokens: x.PromptTokens, CompletionTokens: x.CompletionTok, Cost: x.Cost, TotalMs: x.TotalMs, ErrorOrigin: x.ErrorOrigin, RequestBody: x.RequestBody, ResponseBody: x.ResponseBody}
	for _, a := range x.Attempts {
		d.Attempts = append(d.Attempts, attDTO{Seq: a.Seq, Provider: a.Provider, Model: a.Model, KeyID: a.KeyID, Status: a.Status, LatencyMs: a.LatencyMs, Err: a.Err, ErrorOrigin: a.ErrorOrigin})
	}
	return d
}

func (s *Server) handleRequests(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	q := r.URL.Query()
	f := store.Filter{Pool: q.Get("pool"), Limit: 50}
	if v := q.Get("status"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			f.Status = n
		}
	}
	if v := q.Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 500 {
			f.Limit = n
		}
	}
	if v := q.Get("offset"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			f.Offset = n
		}
	}
	// Bodies are stripped unless explicitly asked for. Each row can carry up to
	// maxStoredBody of request+response text, so the default list response for a
	// few hundred requests ran to ~76MB and made the dashboard's Recent Requests
	// and Requests pages unusable. The detail endpoint loads bodies on demand.
	includeBodies := q.Get("include_bodies") == "1" || q.Get("include_bodies") == "true"
	rows, err := s.store.ListRequests(f, boolInt(includeBodies))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	out := make([]reqDTO, 0, len(rows))
	for _, x := range rows {
		out = append(out, toRequestDTO(x))
	}
	total, _ := s.store.CountRequests(store.Filter{Pool: q.Get("pool"), Status: f.Status, RequestID: q.Get("request_id")})
	writeJSON(w, http.StatusOK, map[string]any{"data": out, "total": total})
}

// handleRequestDetail returns one request with its attempts and full bodies —
// the counterpart to the body-less list rows the dashboard pages over.
func (s *Server) handleRequestDetail(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	id := r.PathValue("id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{"message": "request id required"}})
		return
	}
	req, err := s.store.GetRequest(id)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	if req == nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"error": map[string]string{"message": "request not found"}})
		return
	}
	writeJSON(w, http.StatusOK, toRequestDTO(*req))
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func (s *Server) handleGetConfig(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	writeJSON(w, http.StatusOK, s.cfg.Redacted())
}

func (s *Server) handleSetDefault(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	var body struct {
		Pool string `json:"pool"`
	}
	json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body)
	if err := s.cfg.SetDefault(body.Pool); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"default": s.cfg.Default})
}

func (s *Server) handleSetPool(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	var body struct {
		Pool    string   `json:"pool"`
		Entries []string `json:"entries"`
	}
	json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body)
	if err := s.cfg.SetPool(body.Pool, body.Entries); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"pool": body.Pool, "entries": s.cfg.Pools[body.Pool]})
}

func (s *Server) handleSetProvider(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	var body struct {
		Name      string   `json:"name"`
		BaseURL   string   `json:"base_url"`
		AccountID string   `json:"account_id"`
		Keys      []string `json:"keys"`
		APIMode   string   `json:"api_mode"`
		Action    string   `json:"action"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if body.Action == "delete" {
		if err := s.cfg.DeleteProvider(body.Name); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		s.router.InvalidatePicker(body.Name)
		writeJSON(w, http.StatusOK, map[string]any{"deleted": body.Name})
		return
	}
	if err := s.cfg.SetProvider(body.Name, body.BaseURL, body.AccountID, body.Keys, body.APIMode); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	s.router.InvalidatePicker(body.Name)
	p, _ := s.cfg.Providers.Get(body.Name)
	writeJSON(w, http.StatusOK, map[string]any{"name": body.Name, "base_url": p.BaseURL, "account_id": p.AccountID, "key_count": len(p.Keys)})

	// async: fetch and cache models from the upstream so the dashboard has
	// them ready without the user manually clicking "Fetch Models"
	if s.store != nil && p.BaseURL != "" {
		go s.fetchAndCacheModels(body.Name, p)
	}
}

func (s *Server) handleUpdateProvider(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	name := r.PathValue("name")
	var body struct {
		ModelLimits    map[string]int                 `json:"model_limits"`
		DisabledModels []string                       `json:"disabled_models"`
		MediaPolicies  map[string]config.MediaPolicy  `json:"media_policies"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if body.ModelLimits == nil {
		body.ModelLimits = map[string]int{}
	}
	if body.DisabledModels == nil {
		body.DisabledModels = []string{}
	}
	if body.MediaPolicies == nil {
		body.MediaPolicies = map[string]config.MediaPolicy{}
	}
	if err := s.cfg.SetProviderModelSettings(name, body.ModelLimits, body.DisabledModels, body.MediaPolicies); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	// Re-read what was actually stored: zero-opinion policies are dropped
	// rather than persisted, so echoing the request body back would tell the
	// dashboard it saved rows that no longer exist.
	stored := map[string]config.MediaPolicy{}
	if p, ok := s.cfg.GetProvider(name); ok {
		stored = p.MediaPolicies
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "model_limits": body.ModelLimits, "disabled_models": body.DisabledModels, "media_policies": stored})
}

func (s *Server) handleSetKeys(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	var body struct {
		Name string   `json:"name"`
		Keys []string `json:"keys"`
		// KeyLabels are index-aligned nicknames for Keys. Optional: absent or
		// short means the remaining keys are unlabeled.
		KeyLabels []string `json:"key_labels"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if err := s.cfg.SetProviderKeys(body.Name, body.Keys, body.KeyLabels); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	s.router.InvalidatePicker(body.Name)
	p, _ := s.cfg.Providers.Get(body.Name)
	writeJSON(w, http.StatusOK, map[string]any{"name": body.Name, "key_count": len(p.Keys), "key_labels": p.KeyLabels})

	// async: fetch and cache models now that keys are set
	if s.store != nil && p.BaseURL != "" {
		go s.fetchAndCacheModels(body.Name, p)
	}
}

func (s *Server) handleToggleProvider(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	var body struct {
		Name    string `json:"name"`
		Enabled bool   `json:"enabled"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if err := s.cfg.ToggleProvider(body.Name, body.Enabled); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": body.Name, "enabled": body.Enabled})
}

func (s *Server) handleToggleModel(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	var body struct {
		Provider string `json:"provider"`
		Model    string `json:"model"`
		Disabled bool   `json:"disabled"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if body.Provider == "" || body.Model == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "provider and model are required"})
		return
	}
	if err := s.cfg.ToggleModel(body.Provider, body.Model, body.Disabled); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"provider": body.Provider, "model": body.Model, "disabled": body.Disabled})
}

func (s *Server) handleSetChain(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	var body struct {
		Name    string   `json:"name"`
		Entries []string `json:"entries"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if err := s.cfg.SetChain(body.Name, body.Entries); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"chain": body.Name, "entries": s.cfg.Chains[body.Name]})
}

func (s *Server) handleRemoveChain(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	var body struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if err := s.cfg.RemoveChain(body.Name); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": body.Name})
}

func (s *Server) handleSetTier(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	var body struct {
		Pool    string   `json:"pool"`
		Entries []string `json:"entries"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if err := s.cfg.SetTier(body.Pool, body.Entries); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"pool": body.Pool, "tiers": s.cfg.Tiers[body.Pool]})
}

// handleLogo serves embedded provider logo images from /logos/{name}.png.
// Supports PNG, ICO, and SVG files — content-type is detected from the file signature.
func (s *Server) handleLogo(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/logos/")
	name = strings.TrimSuffix(name, ".png")
	if name == "" {
		http.NotFound(w, r)
		return
	}
	// try .png, .ico, .svg in order
	for _, ext := range []string{".png", ".ico", ".svg"} {
		data, err := fs.ReadFile(logoFS, "web/logos/"+name+ext)
		if err != nil {
			continue
		}
		switch ext {
		case ".svg":
			w.Header().Set("Content-Type", "image/svg+xml")
		case ".ico":
			w.Header().Set("Content-Type", "image/x-icon")
		default:
			w.Header().Set("Content-Type", "image/png")
		}
		w.Header().Set("Cache-Control", "public, max-age=86400")
		w.Write(data)
		return
	}
	http.NotFound(w, r)
}

// resolveProviderURL builds the /v1/models URL and resolves the API key for a
// provider strictly from the configured values. It deliberately ignores any
// caller-supplied ?base_url=/?key= query params: allowing those would turn the
// admin API into an SSRF proxy to arbitrary internal/metadata endpoints.
func (s *Server) resolveProviderURL(r *http.Request) (modelsURL, key string, errStatus int, err error) {
	name := r.PathValue("name")
	p, ok := s.cfg.GetProvider(name)
	if !ok {
		return "", "", http.StatusNotFound, fmt.Errorf("provider not found")
	}
	baseURL := p.BaseURL
	accountID := p.AccountID
	key = ""
	if len(p.Keys) > 0 {
		key = p.Keys[0]
	}
	if accountID != "" {
		baseURL = strings.ReplaceAll(baseURL, "{account_id}", accountID)
	}
	u := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(u, "/v1") {
		u += "/models"
	} else if strings.HasSuffix(u, "/v1/chat/completions") {
		u = strings.TrimSuffix(u, "/chat/completions") + "/models"
	} else {
		u += "/v1/models"
	}
	return u, key, 0, nil
}

// handleProviderModels proxies a GET /v1/models request to the provider's
// upstream API, so the dashboard can fetch model lists without CORS issues.
// After a successful fetch, the models are cached in the store so the
// dashboard doesn't need to re-fetch every time the provider panel is opened.
// It uses the provider's configured base_url and key only (no caller overrides).
func (s *Server) handleProviderModels(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	name := r.PathValue("name")
	modelsURL, key, errStatus, err := s.resolveProviderURL(r)
	if err != nil {
		writeJSON(w, errStatus, map[string]any{"error": err.Error()})
		return
	}
	req, err := http.NewRequest("GET", modelsURL, nil)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]any{"error": "invalid base_url: " + err.Error()})
		return
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		// fall back to cached models if the fetch fails
		if s.store != nil {
			cached, cerr := s.store.GetProviderModels(name)
			if cerr == nil && len(cached) > 0 {
				models := make([]map[string]any, 0, len(cached))
				for _, m := range cached {
					models = append(models, map[string]any{"id": m.ModelID, "object": "model", "source": m.Source})
				}
				writeJSON(w, http.StatusOK, map[string]any{"data": models, "cached": true})
				return
			}
		}
		writeJSON(w, http.StatusBadGateway, map[string]any{"error": "connection failed: " + err.Error()})
		return
	}
	defer resp.Body.Close()

	// read and cache the models if the fetch was successful
	bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode == 200 && s.store != nil {
		var data struct {
			Data   []struct{ ID string `json:"id"` } `json:"data"`
			Models []struct{ ID string `json:"id"` } `json:"models"`
		}
		if json.Unmarshal(bodyBytes, &data) == nil {
			var ids []string
			for _, m := range data.Data {
				if m.ID != "" {
					ids = append(ids, m.ID)
				}
			}
			for _, m := range data.Models {
				if m.ID != "" {
					ids = append(ids, m.ID)
				}
			}
			if len(ids) > 0 {
				s.store.SetProviderModels(name, ids)
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	w.Write(bodyBytes)
}

// handleGetCachedModels returns the stored model cache for a provider without
// making an upstream call. The dashboard uses this when opening a provider
// panel so the user doesn't have to wait for or trigger a fetch.
func (s *Server) handleGetCachedModels(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	name := r.PathValue("name")
	if s.store == nil {
		writeJSON(w, http.StatusOK, map[string]any{"data": []any{}})
		return
	}
	cached, err := s.store.GetProviderModels(name)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	models := make([]map[string]any, 0, len(cached))
	for _, m := range cached {
		models = append(models, map[string]any{"id": m.ModelID, "object": "model", "source": m.Source})
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": models})
}

// handleAddCustomModel adds a manually-specified model to a provider's cache.
// This is for providers with broken or outdated /v1/models endpoints where
// the user knows the model ID but can't auto-fetch it.
func (s *Server) handleAddCustomModel(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	name := r.PathValue("name")
	var body struct {
		ModelID string `json:"model_id"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if body.ModelID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "model_id is required"})
		return
	}
	if len(body.ModelID) > 256 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "model_id too long (max 256 chars)"})
		return
	}
	if _, ok := s.cfg.Providers.Get(name); !ok {
		writeJSON(w, http.StatusNotFound, map[string]any{"error": "provider not found"})
		return
	}
	if s.store == nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "store unavailable"})
		return
	}
	if err := s.store.AddCustomModel(name, body.ModelID); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"provider": name, "model_id": body.ModelID, "source": "custom"})
}

// handleRemoveCustomModel removes a custom model from a provider's cache.
// Accepts model_id via JSON body (preferred) or query param for compatibility.
func (s *Server) handleRemoveCustomModel(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	name := r.PathValue("name")
	modelID := r.URL.Query().Get("model_id")
	if modelID == "" && r.Body != nil {
		var body struct {
			ModelID string `json:"model_id"`
		}
		if json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body) == nil {
			modelID = body.ModelID
		}
	}
	if modelID == "" {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": "model_id is required"})
		return
	}
	if s.store == nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "store unavailable"})
		return
	}
	if err := s.store.RemoveCustomModel(name, modelID); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"deleted": modelID, "provider": name})
}

// fetchAndCacheModels fetches /v1/models from the upstream and caches the
// result in the store. Designed to run as a goroutine — errors are silently
// ignored (the cache just won't be populated, and the user can manually fetch).
func (s *Server) fetchAndCacheModels(name string, p *config.Provider) {
	base := strings.TrimSuffix(p.BaseURL, "/")
	if !strings.HasSuffix(base, "/v1") {
		base += "/v1"
	}
	req, err := http.NewRequest("GET", base+"/models", nil)
	if err != nil {
		log.Printf("fetchAndCacheModels %s: %v", name, err)
		return
	}
	key := ""
	if len(p.Keys) > 0 {
		key = p.Keys[0]
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("fetchAndCacheModels %s: %v", name, err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		log.Printf("fetchAndCacheModels %s: HTTP %d", name, resp.StatusCode)
		return
	}
	bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var data struct {
		Data   []struct{ ID string `json:"id"` } `json:"data"`
		Models []struct{ ID string `json:"id"` } `json:"models"`
	}
	if json.Unmarshal(bodyBytes, &data) != nil {
		log.Printf("fetchAndCacheModels %s: invalid JSON from /models", name)
		return
	}
	var ids []string
	for _, m := range data.Data {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}
	for _, m := range data.Models {
		if m.ID != "" {
			ids = append(ids, m.ID)
		}
	}
	if len(ids) > 0 {
		if err := s.store.SetProviderModels(name, ids); err != nil {
			log.Printf("fetchAndCacheModels %s: SetProviderModels: %v", name, err)
		}
	}
}

// handleProviderTest is a lightweight connection test — same as models
// but returns a summary {ok, model_count} instead of the raw response.
// It uses the provider's configured base_url and key only.
func (s *Server) handleProviderTest(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "unauthorized"}})
		return
	}
	modelsURL, key, errStatus, err := s.resolveProviderURL(r)
	if err != nil {
		writeJSON(w, errStatus, map[string]any{"error": err.Error()})
		return
	}
	req, err := http.NewRequest("GET", modelsURL, nil)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]any{"error": "invalid base_url: " + err.Error()})
		return
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]any{"error": "connection failed: " + err.Error()})
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		writeJSON(w, http.StatusBadGateway, map[string]any{"error": "HTTP " + fmt.Sprintf("%d", resp.StatusCode)})
		return
	}
	var data struct {
		Data   []any `json:"data"`
		Models []any `json:"models"`
	}
	json.NewDecoder(resp.Body).Decode(&data)
	count := len(data.Data) + len(data.Models)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "model_count": count})
}
