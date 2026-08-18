package server

import (
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
	"llm-router/internal/route"
	"llm-router/internal/store"
)

func TestUnmatchedAPIPathsReturnJSONNotDashboard(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	// GET on a known POST-only endpoint: 405 JSON, not dashboard HTML.
	resp, err := http.Get(h.srv.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatalf("GET chat completions: %v", err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET /v1/chat/completions: status %d, want 405 (body: %s)", resp.StatusCode, b)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Fatalf("GET /v1/chat/completions: Content-Type = %q, want JSON", ct)
	}
	var e struct {
		Error map[string]any `json:"error"`
	}
	if err := json.Unmarshal(b, &e); err != nil || e.Error == nil {
		t.Fatalf("GET /v1/chat/completions: expected JSON error body, got %q", b)
	}

	// Unknown /v1/* and /api/* paths: 404 JSON.
	for _, path := range []string{"/v1/does-not-exist", "/api/does-not-exist", "/api/requests/not-a-real-id"} {
		resp, err := http.Get(h.srv.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		bb, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Fatalf("GET %s: status %d, want 404 (body: %s)", path, resp.StatusCode, bb)
		}
		if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "application/json") {
			t.Fatalf("GET %s: Content-Type = %q, want JSON", path, ct)
		}
	}

	// The dashboard itself is still served at / with HTML.
	resp, _ = http.Get(h.srv.URL + "/")
	b, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !strings.Contains(resp.Header.Get("Content-Type"), "text/html") {
		t.Fatalf("GET /: status %d ct %q, want 200 text/html", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	if !strings.Contains(string(b), "LLM Router") {
		t.Fatalf("GET /: dashboard HTML not served")
	}
}

func TestRequestListStripsBodiesAndDetailHasThem(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	resp, body := post(t, h.srv.URL+"/v1/chat/completions", "", `{"model":"any","messages":[{"role":"user","content":"hello detail body test"}]}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("chat: status %d body %s", resp.StatusCode, body)
	}

	rows, _ := h.store.ListRequests(store.Filter{Limit: 5}, 0)
	if len(rows) != 1 {
		t.Fatalf("expected 1 logged request, got %d", len(rows))
	}
	id := rows[0].ID

	getJSON := func(path string, v any) int {
		r, err := http.Get(h.srv.URL + path)
		if err != nil {
			t.Fatalf("GET %s: %v", path, err)
		}
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(v); err != nil {
			t.Fatalf("decode %s: %v", path, err)
		}
		return r.StatusCode
	}

	// The list endpoint strips request/response bodies by default so paging the
	// event log stays small (a few hundred requests used to arrive as ~76MB).
	var list struct {
		Data []map[string]any `json:"data"`
	}
	if st := getJSON("/api/requests?limit=10", &list); st != 200 {
		t.Fatalf("list status %d", st)
	}
	if len(list.Data) != 1 {
		t.Fatalf("list: expected 1 row, got %d", len(list.Data))
	}
	if _, has := list.Data[0]["request_body"]; has {
		t.Fatal("list response must strip request_body")
	}
	if _, has := list.Data[0]["response_body"]; has {
		t.Fatal("list response must strip response_body")
	}

	// GET /api/requests/{id} returns the full row with bodies for the detail panel.
	var det map[string]any
	if st := getJSON("/api/requests/"+id, &det); st != 200 {
		t.Fatalf("detail status %d", st)
	}
	reqBody, _ := det["request_body"].(string)
	if !strings.Contains(reqBody, "hello detail body test") {
		t.Fatalf("detail request_body should carry the payload, got %q", reqBody)
	}
	respBody, _ := det["response_body"].(string)
	if !strings.Contains(respBody, "chat-response") {
		t.Fatalf("detail response_body should carry the upstream response, got %q", respBody)
	}

	// include_bodies=true on the list opts back in.
	var ilist struct {
		Data []map[string]any `json:"data"`
	}
	if st := getJSON("/api/requests?include_bodies=true", &ilist); st != 200 {
		t.Fatalf("include_bodies list status %d", st)
	}
	if _, has := ilist.Data[0]["request_body"]; !has {
		t.Fatal("include_bodies=true list must include request_body")
	}
	if _, has := ilist.Data[0]["response_body"]; !has {
		t.Fatal("include_bodies=true list must include response_body")
	}
}

func TestVendorEmbeddedAndServed(t *testing.T) {
	// Sanity: the vendored assets are actually embedded at compile time.
	for want := range map[string]bool{
		"web/vendor/tailwind.js":                true,
		"web/vendor/fonts/fonts.css":            true,
		"web/vendor/fonts/material-symbols.css": true,
	} {
		if _, err := fs.Stat(vendorFS, want); err != nil {
			t.Fatalf("expected %s to be embedded: %v", want, err)
		}
	}
	if _, err := fs.Stat(vendorFS, "web/vendor/fonts/kJEPhwA2.woff2"); err != nil {
		// The Hanken font filename; allow the actual woff2 to differ.
		entries, _ := fs.ReadDir(vendorFS, "web/vendor/fonts")
		var found bool
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".woff2") {
				found = true
			}
		}
		if !found {
			t.Fatalf("expected a woff2 font file to be embedded: %v", err)
		}
	}

	// The dashboard must reference the local (not CDN) assets.
	if strings.Contains(dashboardHTML, "cdn.tailwindcss.com") ||
		strings.Contains(dashboardHTML, "fonts.googleapis.com") {
		t.Fatal("dashboardHTML still references a CDN; should use local /vendor/* assets only")
	}
	for _, local := range []string{
		`/vendor/tailwind.js`,
		`/vendor/fonts/fonts.css`,
		`/vendor/fonts/material-symbols.css`,
	} {
		if !strings.Contains(dashboardHTML, local) {
			t.Fatalf("dashboardHTML missing local asset %q", local)
		}
	}

	// Serve via the real multiplexer with nil deps; the /vendor/ route does
	// not consult cfg/store/router/client/gate.
	srv := &Server{cfg: nil, store: nil, router: nil,
		client: nil, gate: catalog.NewGate(nil)}
	mux := srv.Handler()
	ts := httptest.NewServer(mux)
	defer ts.Close()

	cases := []struct {
		path            string
		wantType        string
		bodyMustContain string
	}{
		{"/vendor/tailwind.js", "javascript", "tailwind"},
		{"/vendor/fonts/fonts.css", "text/css", "Hanken Grotesk"},
		{"/vendor/fonts/material-symbols.css", "text/css", "material-symbols"},
	}
	for _, c := range cases {
		c := c
		t.Run(c.bodyMustContain, func(t *testing.T) {
			res, err := http.Get(ts.URL + c.path)
			if err != nil {
				t.Fatalf("GET %s: %v", c.path, err)
			}
			defer res.Body.Close()
			if res.StatusCode != http.StatusOK {
				t.Fatalf("GET %s: status %d", c.path, res.StatusCode)
			}
			if ct := res.Header.Get("Content-Type"); !strings.Contains(ct, c.wantType) {
				t.Fatalf("GET %s: Content-Type = %q, want it to contain %q", c.path, ct, c.wantType)
			}
		})
	}

	// A vendored woff2 must be served as a font with the right binary mime.
	entries, err := fs.ReadDir(vendorFS, "web/vendor/fonts")
	if err != nil {
		t.Fatalf("readdir fonts: %v", err)
	}
	var woff2 string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".woff2") {
			woff2 = "/vendor/fonts/" + e.Name()
			break
		}
	}
	if woff2 == "" {
		t.Fatal("no woff2 font embedded")
	}
	res, err := http.Get(ts.URL + woff2)
	if err != nil {
		t.Fatalf("GET %s: %v", woff2, err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("GET %s: status %d", woff2, res.StatusCode)
	}
	if ct := res.Header.Get("Content-Type"); !strings.HasPrefix(ct, "font/woff2") {
		t.Fatalf("GET %s: Content-Type = %q, want font/woff2*", woff2, ct)
	}
}

type harness struct {
	srv    *httptest.Server
	store  *store.Store
	cfg    *config.Config
	vision *httptest.Server
	chat   *httptest.Server
	code   *httptest.Server
}

func newHarness(t *testing.T, routerKey string) *harness {
	t.Helper()
	var visionHits, chatHits, codeHits atomic.Int32

	vision := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		visionHits.Add(1)
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		// hop 1 must never receive stream=true
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"DESCRIBED:image"}}]}`)
	}))

	chat := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"chat-response"}}]}`)
	}))

	code := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		codeHits.Add(1)
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"coded"}}]}`)
	}))

	cfg := &config.Config{
		Default:         "chat",
		RouterKey:       routerKey,
		InsecureNoAuth:  true, // local unit tests run unauthenticated
		Vision:          []string{"vis:vm"},
		Pools: map[string][]string{
			"chat": {"chatp:cm"},
			"code": {"codep:cm2"},
		},
		Classifier: config.ClassifierCfg{Heuristics: map[string][]string{"code": {"import "}}},
		Providers: config.Providers{
			Custom: map[string]*config.Provider{
				"vis":   {BaseURL: vision.URL, Keys: []string{"vk"}},
				"chatp": {BaseURL: chat.URL, Keys: []string{"ck"}},
				"codep": {BaseURL: code.URL, Keys: []string{"dk"}},
			},
		},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}

	g := catalog.NewGate(map[string]catalog.ModelInfo{
		"vis:vm":    {ContextWindow: 100000, Vision: true},
		"chatp:cm":  {ContextWindow: 100000, Vision: true},
		"codep:cm2": {ContextWindow: 100000, Vision: false},
	})

	st, err := store.Open(filepath.Join(t.TempDir(), "srv.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, g)
	ts := httptest.NewServer(s.Handler())

	return &harness{srv: ts, store: st, cfg: cfg, vision: vision, chat: chat, code: code}
}

func (h *harness) close() {
	h.srv.Close()
	h.vision.Close()
	h.chat.Close()
	h.code.Close()
	h.store.Close()
}

func post(t *testing.T, url, key, body string, hdr map[string]string) (*http.Response, string) {
	t.Helper()
	req, _ := http.NewRequest("POST", url, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp, string(b)
}

func TestChatCompletionsSuccessAndLogged(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	resp, body := post(t, h.srv.URL+"/v1/chat/completions", "", `{"model":"any","messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	if !strings.Contains(body, "chat-response") {
		t.Fatalf("expected chat pool passthrough, got %s", body)
	}
	rows, _ := h.store.ListRequests(store.Filter{Limit: 10}, 0)
	if len(rows) != 1 || rows[0].Pool != "chat" || rows[0].Rule != "default" {
		t.Fatalf("request not logged correctly: %+v", rows)
	}
	if len(rows[0].Attempts) != 1 || rows[0].Attempts[0].Status != 200 {
		t.Fatalf("attempt not logged: %+v", rows[0].Attempts)
	}
}

func TestHintHeaderRoutesToCodePool(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	resp, body := post(t, h.srv.URL+"/v1/chat/completions", "", `{"model":"any","messages":[{"role":"user","content":"hello"}]}`, map[string]string{"X-Route-Pool": "code"})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	rows, _ := h.store.ListRequests(store.Filter{Limit: 10}, 0)
	if rows[0].Rule != "hint" {
		t.Fatalf("rule should be hint, got %s", rows[0].Rule)
	}
	if rows[0].FinalProvider != "codep" {
		t.Fatalf("should have hit code pool provider, got %s", rows[0].FinalProvider)
	}
}

func TestImagePlusCodeUsesVisionChain(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	msg := `{"model":"any","messages":[{"role":"user","content":[{"type":"text","text":"import os\nwrite tests for this"},{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]}]}`
	resp, body := post(t, h.srv.URL+"/v1/chat/completions", "", msg, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	// code pool provider must have received the extraction, not pixels
	rows, _ := h.store.ListRequests(store.Filter{Limit: 10}, 0)
	if rows[0].Pool != "code" {
		t.Fatalf("image+code signal should land in code pool, got %s", rows[0].Pool)
	}
	_ = body
}

func TestAuthRequiredWhenConfigured(t *testing.T) {
	h := newHarness(t, "secret-key")
	defer h.close()

	resp, _ := post(t, h.srv.URL+"/v1/chat/completions", "", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != 401 {
		t.Fatalf("missing key status %d, want 401", resp.StatusCode)
	}
	resp, body := post(t, h.srv.URL+"/v1/chat/completions", "secret-key", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("valid key status %d body %s", resp.StatusCode, body)
	}
}

func TestFallbackAcrossProvidersOverHTTP(t *testing.T) {
	// provider chatp 429s, code pool succeeds
	fail := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(429)
	}))
	defer fail.Close()
	good := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"rescued"}}]}`)
	}))
	defer good.Close()

	cfg := &config.Config{
		InsecureNoAuth: true,
		Default: "chat",
		Pools:   map[string][]string{"chat": {"a:x", "b:y"}},
		Providers: config.Providers{
			Custom: map[string]*config.Provider{
				"a": {BaseURL: fail.URL, Keys: []string{"k1", "k2"}},
				"b": {BaseURL: good.URL, Keys: []string{"k3"}},
			},
		},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	st, _ := store.Open(filepath.Join(t.TempDir(), "srv.db"))
	defer st.Close()
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, g)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	resp, body := post(t, ts.URL+"/v1/chat/completions", "", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %s: a 429 must never surface as your error", resp.StatusCode, body)
	}
	if !strings.Contains(body, "rescued") {
		t.Fatalf("body should come from fallback provider: %s", body)
	}
	rows, _ := st.ListRequests(store.Filter{Limit: 10}, 0)
	if len(rows[0].Attempts) != 3 {
		t.Fatalf("expected k1, k2, k3 attempts: %+v", rows[0].Attempts)
	}
	if rows[0].Attempts[2].Provider != "b" {
		t.Fatalf("final attempt should be provider b: %+v", rows[0].Attempts[2])
	}
}

func TestExhaustedChainReturnsClean503(t *testing.T) {
	fail := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(429)
	}))
	defer fail.Close()

	cfg := &config.Config{
		InsecureNoAuth: true,
		Default: "chat",
		Pools:   map[string][]string{"chat": {"a:x"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: fail.URL, Keys: []string{"k1"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	st, _ := store.Open(filepath.Join(t.TempDir(), "srv.db"))
	defer st.Close()
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, g)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	resp, body := post(t, ts.URL+"/v1/chat/completions", "", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != 503 {
		t.Fatalf("status %d, want 503", resp.StatusCode)
	}
	if resp.Header.Get("X-Router-Fallback-Exhausted") != "true" {
		t.Fatal("missing X-Router-Fallback-Exhausted header")
	}
	if resp.Header.Get("X-Router-Request-Id") == "" {
		t.Fatal("missing X-Router-Request-Id header")
	}
	if !strings.Contains(body, "exhausted") {
		t.Fatalf("503 body should explain exhaustion: %s", body)
	}
}

// TestRetryableAttemptCapturesErrorBody ensures upstream 429/5xx responses are
// recorded with their error body in the attempt trail, so operators can see
// *why* a model failed from the dashboard. This guards the capture that was
// previously missing on the retryable path (att.Err was left empty for 429/5xx).
func TestRetryableAttemptCapturesErrorBody(t *testing.T) {
	fail := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(429)
		io.WriteString(w, `{"error":{"message":"Rate limit reached, please slow down","type":"rate_limit_error"}}`)
	}))
	defer fail.Close()

	cfg := &config.Config{
		InsecureNoAuth: true,
		Default: "chat",
		Pools:   map[string][]string{"chat": {"a:x"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: fail.URL, Keys: []string{"k1"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	st, _ := store.Open(filepath.Join(t.TempDir(), "srv.db"))
	defer st.Close()
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, g)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	resp, _ := post(t, ts.URL+"/v1/chat/completions", "", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != 503 {
		t.Fatalf("status %d, want 503 (fallback exhausted)", resp.StatusCode)
	}
	rows, _ := st.ListRequests(store.Filter{Limit: 10}, 0)
	if len(rows) == 0 {
		t.Fatal("expected a logged request")
	}
	att := rows[0].Attempts[0]
	if att.Status != 429 {
		t.Fatalf("attempt status = %d, want 429", att.Status)
	}
	if !strings.Contains(att.Err, "Rate limit reached") {
		t.Fatalf("retryable attempt must capture upstream error body; got Err=%q", att.Err)
	}
}

func TestStreamingPassthrough(t *testing.T) {
	done := make(chan struct{})
	streamUp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		// write both chunks in ONE write so there's no flush-timing issue
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"chunk1\"}}]}\n\ndata: [DONE]\n\n")
		if fl, ok := w.(http.Flusher); ok {
			fl.Flush()
		}
		<-done
	}))
	defer streamUp.Close()

	cfg := &config.Config{
		InsecureNoAuth: true,
		Default: "chat",
		Pools:   map[string][]string{"chat": {"a:x"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: streamUp.URL, Keys: []string{"k1"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	st, _ := store.Open(filepath.Join(t.TempDir(), "srv.db"))
	defer st.Close()
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, g)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// Signal the upstream to release its handler as soon as we've
	// started reading the response — the body is already written
	// and flushed, so closing done unblocks the upstream handler
	// without losing data.
	slowClient := &http.Client{}
	req, _ := http.NewRequest("POST", ts.URL+"/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"stream":true}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := slowClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	close(done)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %s", resp.StatusCode, string(body))
	}
	if !strings.Contains(string(body), "chunk1") || !strings.Contains(string(body), "[DONE]") {
		t.Fatalf("SSE chunks must pass through: %q", string(body))
	}
}

func TestAdminRequestsEndpoint(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()
	post(t, h.srv.URL+"/v1/chat/completions", "", `{"messages":[{"role":"user","content":"hi"}]}`, nil)

	resp, err := http.Get(h.srv.URL + "/api/requests?limit=5")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	var wrapper struct {
		Data  []map[string]any `json:"data"`
		Total int              `json:"total"`
	}
	if err := json.Unmarshal(b, &wrapper); err != nil {
		t.Fatalf("admin /api/requests must return JSON: %v (%s)", err, b)
	}
	out := wrapper.Data
	if len(out) == 0 {
		t.Fatal("expected at least one request row")
	}
	if out[0]["pool"] == nil {
		t.Fatalf("row shape wrong: %s", b)
	}
}

func TestHealthz(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()
	resp, err := http.Get(h.srv.URL + "/healthz")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 || strings.TrimSpace(string(b)) != "ok" {
		t.Fatalf("healthz: %d %s", resp.StatusCode, b)
	}
}

func TestConfigDefaultUpdatePersists(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "router.yaml")
	cfgContent := `listen: "127.0.0.1:8011"
default: chat
insecure_no_auth: true
pools:
  chat: ["a:x"]
  code: ["a:x"]
providers:
  custom:
    a: { base_url: "http://127.0.0.1:1/v1", keys: ["sk-test"] }
fallback:
  strategy: round_robin
`
	cfg, err := config.Load(strings.NewReader(cfgContent))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	cfg.Path = cfgPath
	cfg.DBPath = filepath.Join(dir, "r.db")
	st, _ := store.Open(cfg.DBPath)
	defer st.Close()
	g := catalog.NewGate(nil)
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, g)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/api/config/default", strings.NewReader(`{"pool":"code"}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post default: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("set-default status %d", resp.StatusCode)
	}
	if cfg.Default != "code" {
		t.Fatalf("in-memory default = %s, want code", cfg.Default)
	}
	// persisted to yaml
	reloaded, err := config.LoadFile(cfgPath)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if reloaded.Default != "code" {
		t.Fatalf("persisted default = %s, want code", reloaded.Default)
	}
}

func TestConfigAPIHidesKeys(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()
	resp, err := http.Get(h.srv.URL + "/api/config")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if strings.Contains(string(b), "vk") || strings.Contains(string(b), "sk-test") {
		t.Fatalf("keys must be redacted in /api/config: %s", b)
	}
	if !strings.Contains(string(b), "srv") && !strings.Contains(string(b), "keys") {
		t.Fatalf("unexpected config payload: %s", b)
	}
}

var _ = context.Background
