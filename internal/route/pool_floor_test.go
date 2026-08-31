package route

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

func TestPoolContextFloorSkipsSmallModels(t *testing.T) {
	// Pool "chat" advertises 512k. a:x has 200k (below floor) -> skipped.
	// b:y has 600k (above floor) -> serves the request.
	good := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer good.Close()

	cfg := &config.Config{
		Default:     "chat",
		Pools:       map[string][]string{"chat": {"a:x", "b:y"}},
		PoolContext: map[string]int{"chat": 512000},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: good.URL, Keys: []string{"k1"}},
			"b": {BaseURL: good.URL, Keys: []string{"k2"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(map[string]catalog.ModelInfo{
		"a:x": {ContextWindow: 200000},
		"b:y": {ContextWindow: 600000},
	})
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "auto"}, false, false, false, 1000)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200", res.Status)
	}
	if len(res.Attempts) != 2 {
		t.Fatalf("expected 2 attempts (a excluded, b ok): %d", len(res.Attempts))
	}
	a := res.Attempts[0]
	if a.Provider != "a" || a.Status != 0 {
		t.Fatalf("first attempt should be excluded a (status 0): %+v", a)
	}
	if !strings.Contains(a.Err, "below pool floor") {
		t.Fatalf("first attempt err should mention pool floor: %q", a.Err)
	}
	if res.Attempts[1].Provider != "b" || res.Attempts[1].Status != 200 {
		t.Fatalf("second attempt should be b with 200: %+v", res.Attempts[1])
	}
}
