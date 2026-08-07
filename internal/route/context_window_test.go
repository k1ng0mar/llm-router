package route

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

func TestContextWindowExceededFallsThrough(t *testing.T) {
	// provider "a" returns 400 with a context-window error body;
	// provider "b" (larger context) succeeds. Should fall through, not fail loud.
	tooLong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		w.Write([]byte(`{"error":{"message":"This model's maximum context length is 8192 tokens"}}`))
	}))
	defer tooLong.Close()

	good := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer good.Close()

	cfg := &config.Config{
		Default:  "chat",
		Pools:    map[string][]string{"chat": {"a:x", "b:y"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: tooLong.URL, Keys: []string{"k1"}},
			"b": {BaseURL: good.URL, Keys: []string{"k2"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(map[string]catalog.ModelInfo{
		"a:x": {ContextWindow: 200000},
		"b:y": {ContextWindow: 200000},
	})
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "x"}, false, false, false, 5000)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200 (should have fallen through)", res.Status)
	}
	if len(res.Attempts) != 2 {
		t.Fatalf("expected 2 attempts (a failed, b ok): %d", len(res.Attempts))
	}
	if res.Attempts[0].Provider != "a" || res.Attempts[0].Status != 400 {
		t.Fatalf("first attempt should be from a with 400: %+v", res.Attempts[0])
	}
	if res.Attempts[0].Err == "" {
		t.Fatal("context-window attempt should carry the fallback-eligible marker")
	}
}

func TestPure4xxFallsBack(t *testing.T) {
	// 400 with an unrelated message → falls back to next provider (non-200 = fallback)
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		w.Write([]byte(`{"error":{"message":"bad request: missing field"}}`))
	}))
	defer bad.Close()

	good := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer good.Close()

	cfg := &config.Config{
		Default:  "chat",
		Pools:    map[string][]string{"chat": {"a:x", "b:y"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: bad.URL, Keys: []string{"k1"}},
			"b": {BaseURL: good.URL, Keys: []string{"k2"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "x"}, false, false, false, 500)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200 (should have fallen through)", res.Status)
	}
	if len(res.Attempts) != 2 {
		t.Fatalf("expected 2 attempts (a failed, b ok): %d", len(res.Attempts))
	}
	if res.Attempts[0].Provider != "a" || res.Attempts[0].Status != 400 {
		t.Fatalf("first attempt should be from a with 400: %+v", res.Attempts[0])
	}
	if res.Attempts[1].Provider != "b" || res.Attempts[1].Status != 200 {
		t.Fatalf("second attempt should be from b with 200: %+v", res.Attempts[1])
	}
}
