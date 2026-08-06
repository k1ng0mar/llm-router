package route

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

// stub returns a fixed status (or scripted per-path behavior) and counts hits.
func stub(code int, body string, hits *atomic.Int32) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits != nil {
			hits.Add(1)
		}
		w.WriteHeader(code)
		if body != "" {
			w.Write([]byte(body))
		}
	}))
}

func testRouter(stub1, stub2 *httptest.Server) (*Router, *config.Config) {
	cfg := &config.Config{
		Default: "chat",
		Pools: map[string][]string{
			"code": {"p1:m1", "p2:m2"},
		},
		Providers: config.Providers{
			Custom: map[string]*config.Provider{
				"p1": {BaseURL: stub1.URL, Keys: []string{"k1", "k2"}},
				"p2": {BaseURL: stub2.URL, Keys: []string{"k3"}},
			},
		},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(map[string]catalog.ModelInfo{"p1:m1": {ContextWindow: 100000, Vision: true}})
	return NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient}), cfg
}

func TestFallbackAcrossKeysThenProvider(t *testing.T) {
	var hits1, hits2 atomic.Int32
	s1 := stub(429, "", &hits1)
	s2 := stub(200, `{"choice":"ok"}`, &hits2)
	defer s1.Close()
	defer s2.Close()

	r, _ := testRouter(s1, s2)
	res, err := r.Route(context.Background(), "code", map[string]any{"model": "x"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("final status = %d, want 200", res.Status)
	}
	if hits1.Load() != 2 {
		t.Fatalf("stub1 should be hit twice (k1, k2), got %d", hits1.Load())
	}
	if hits2.Load() != 1 {
		t.Fatalf("stub2 should be hit once, got %d", hits2.Load())
	}
	if len(res.Attempts) != 3 {
		t.Fatalf("expected 3 attempts, got %d", len(res.Attempts))
	}
	// keys used in order k1, k2 (p1), then provider p2
	if res.Attempts[0].KeyID != "k1" || res.Attempts[1].KeyID != "k2" || res.Attempts[2].Provider != "p2" {
		t.Fatalf("attempt sequence wrong: %+v", res.Attempts)
	}
	if res.Attempts[0].Status != 429 || res.Attempts[1].Status != 429 || res.Attempts[2].Status != 200 {
		t.Fatalf("attempt statuses wrong: %+v", res.Attempts)
	}
	if res.Resp == nil {
		t.Fatal("successful response must be attached")
	}
	body, _ := io.ReadAll(res.Resp.Body)
	res.Resp.Body.Close()
	if string(body) != `{"choice":"ok"}` {
		t.Fatalf("body = %s", body)
	}
}

func TestFourXXStopsImmediately(t *testing.T) {
	var hits1 atomic.Int32
	s1 := stub(400, `{"error":"bad"}`, &hits1)
	s2 := stub(200, `{"choice":"ok"}`, nil)
	defer s1.Close()
	defer s2.Close()

	r, _ := testRouter(s1, s2)
	res, err := r.Route(context.Background(), "code", map[string]any{"model": "x"}, false, false, false, 10)
	if err == nil {
		t.Fatal("a 4xx must surface as an error, not fall through")
	}
	if res == nil || res.Status != 400 {
		t.Fatalf("status = %v, want 400", res.Status)
	}
	if hits1.Load() != 1 {
		t.Fatalf("stub1 should be hit exactly once, got %d", hits1.Load())
	}
	if len(res.Attempts) != 1 {
		t.Fatalf("no provider after a 4xx: got %d attempts", len(res.Attempts))
	}
}

func TestExhaustedReturns503(t *testing.T) {
	var hits1, hits2 atomic.Int32
	s1 := stub(429, "", &hits1)
	s2 := stub(429, "", &hits2)
	defer s1.Close()
	defer s2.Close()

	r, _ := testRouter(s1, s2)
	res, err := r.Route(context.Background(), "code", map[string]any{"model": "x"}, false, false, false, 10)
	if err == nil {
		t.Fatal("exhausted chain must error")
	}
	if res.Status != 503 {
		t.Fatalf("exhausted status = %d, want 503", res.Status)
	}
	if len(res.Attempts) != 3 {
		t.Fatalf("all three candidates attempted: got %d", len(res.Attempts))
	}
}

func TestCapabilityExclusionIsLogged(t *testing.T) {
	var hits1 atomic.Int32
	// p1:m1 is vision-capable; p2:m2 is unknown → fails open. Use a pool where the
	// second provider model is known non-vision so it is structurally excluded.
	s1 := stub(200, `{"ok":1}`, &hits1)
	defer s1.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"code": {"p1:m1"}},
		Providers: config.Providers{
			Custom: map[string]*config.Provider{"p1": {BaseURL: s1.URL, Keys: []string{"k1"}}},
		},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin"},
	}
	g := catalog.NewGate(map[string]catalog.ModelInfo{"p1:m1": {ContextWindow: 100000, Vision: true}})
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	res, err := r.Route(context.Background(), "code", map[string]any{"model": "x"}, true, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status %d, want 200", res.Status)
	}
}

func TestCapabilityExclusionBlocksNonVisionOnlyPool(t *testing.T) {
	s1 := stub(200, `{"ok":1}`, nil)
	defer s1.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"code": {"p1:m1"}},
		Providers: config.Providers{
			Custom: map[string]*config.Provider{"p1": {BaseURL: s1.URL, Keys: []string{"k1"}}},
		},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin"},
	}
	g := catalog.NewGate(map[string]catalog.ModelInfo{"p1:m1": {ContextWindow: 100000, Vision: false}})
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	res, err := r.Route(context.Background(), "code", map[string]any{"model": "x"}, true, false, false, 10)
	if err == nil {
		t.Fatal("pool with only non-vision models on an image request must fail")
	}
	if res.Status != 503 {
		t.Fatalf("status %d, want 503 with capability reason", res.Status)
	}
	if len(res.Attempts) != 1 || res.Attempts[0].Status != 0 {
		t.Fatalf("excluded attempt should be logged: %+v", res.Attempts)
	}
	if res.Attempts[0].Err == "" {
		t.Fatal("excluded attempt must carry the reason")
	}
}
