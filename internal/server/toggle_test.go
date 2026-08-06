package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
	"llm-router/internal/route"
	"llm-router/internal/store"
)

func TestToggleProviderDisabledSkipped(t *testing.T) {
	// provider "a" returns 200, provider "b" returns 200.
	// Disable "a" → requests should skip it and hit "b" only.
	goodA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"from-a"}}]}`)
	}))
	defer goodA.Close()
	goodB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"from-b"}}]}`)
	}))
	defer goodB.Close()

	cfg := &config.Config{
		InsecureNoAuth: true,
		Default:  "chat",
		Pools:    map[string][]string{"chat": {"a:x", "b:y"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: goodA.URL, Keys: []string{"k1"}},
			"b": {BaseURL: goodB.URL, Keys: []string{"k2"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	st, _ := store.Open(filepath.Join(t.TempDir(), "srv.db"))
	defer st.Close()
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, nil)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// disable provider "a" via the API
	body, _ := json.Marshal(map[string]any{"name": "a", "enabled": false})
	resp, err := http.Post(ts.URL+"/api/config/toggle", "application/json", strings.NewReader(string(body)))
	if err != nil {
		t.Fatalf("toggle: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("toggle status %d", resp.StatusCode)
	}
	resp.Body.Close()

	// now send a request — should hit "b" only (skip disabled "a")
	resp2, respBody := post(t, ts.URL+"/v1/chat/completions", "", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp2.StatusCode != 200 {
		t.Fatalf("status %d body %s", resp2.StatusCode, respBody)
	}
	if !strings.Contains(respBody, "from-b") {
		t.Fatalf("should have hit provider b (a is disabled), got: %s", respBody)
	}
	rows, _ := st.ListRequests(store.Filter{Limit: 10}, 0)
	if len(rows[0].Attempts) != 2 {
		t.Fatalf("should have 2 attempts (1 excluded + 1 from b), got %d", len(rows[0].Attempts))
	}
	// first attempt is logged as router-origin exclusion
	if rows[0].Attempts[0].Err != "provider disabled" {
		t.Fatalf("first attempt should be 'provider disabled', got %q", rows[0].Attempts[0].Err)
	}
	// second attempt is from the enabled provider
	if rows[0].Attempts[1].Provider != "b" {
		t.Fatalf("second attempt should be from b, got %s", rows[0].Attempts[1].Provider)
	}
}

func TestToggleProviderReEnabled(t *testing.T) {
	goodA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		io.WriteString(w, `{"choices":[{"message":{"content":"from-a"}}]}`)
	}))
	defer goodA.Close()

	cfg := &config.Config{
		InsecureNoAuth: true,
		Default:  "chat",
		Pools:    map[string][]string{"chat": {"a:x"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: goodA.URL, Keys: []string{"k1"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	st, _ := store.Open(filepath.Join(t.TempDir(), "srv.db"))
	defer st.Close()
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, nil)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// disable
 body, _ := json.Marshal(map[string]any{"name": "a", "enabled": false})
	resp, _ := http.Post(ts.URL+"/api/config/toggle", "application/json", strings.NewReader(string(body)))
	resp.Body.Close()

	// request should fail (exhausted, no enabled providers)
	resp2, respBody := post(t, ts.URL+"/v1/chat/completions", "", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp2.StatusCode != 503 {
		t.Fatalf("disabled-only-provider should give 503, got %d body %s", resp2.StatusCode, respBody)
	}

	// re-enable
 body2, _ := json.Marshal(map[string]any{"name": "a", "enabled": true})
	resp3, _ := http.Post(ts.URL+"/api/config/toggle", "application/json", strings.NewReader(string(body2)))
	resp3.Body.Close()

	// request should succeed
	resp4, respBody4 := post(t, ts.URL+"/v1/chat/completions", "", `{"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp4.StatusCode != 200 {
		t.Fatalf("re-enabled provider should work, got %d body %s", resp4.StatusCode, respBody4)
	}
}

func TestToggleShowsInConfig(t *testing.T) {
	cfg := &config.Config{
		InsecureNoAuth: true,
		Default:  "chat",
		Pools:    map[string][]string{"chat": {"a:x"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: "http://localhost:1/v1", Keys: []string{"k1"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	st, _ := store.Open(filepath.Join(t.TempDir(), "srv.db"))
	defer st.Close()
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, nil)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// disable
 body, _ := json.Marshal(map[string]any{"name": "a", "enabled": false})
	resp, _ := http.Post(ts.URL+"/api/config/toggle", "application/json", strings.NewReader(string(body)))
	resp.Body.Close()

	// /api/config should show enabled: false
	resp2, err := http.Get(ts.URL + "/api/config")
	if err != nil {
		t.Fatalf("get config: %v", err)
	}
	defer resp2.Body.Close()
	b, _ := io.ReadAll(resp2.Body)
	if !strings.Contains(string(b), `"enabled":false`) {
		t.Fatalf("config should show provider a as disabled: %s", b)
	}
}
