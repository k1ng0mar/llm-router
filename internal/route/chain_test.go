package route

import (
	"context"
	"net/http"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

func TestNamedChainRoutesToConfiguredSequence(t *testing.T) {
	s1 := stub(429, ``, nil)
	defer s1.Close()
	s2 := stub(200, `{"choices":[{"message":{"content":"from-chain-2"}}]}`, nil)
	defer s2.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"a:x"}},
		Chains: map[string][]string{
			"fast": {"a:x", "b:y"},
		},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: s1.URL, Keys: []string{"k1"}},
			"b": {BaseURL: s2.URL, Keys: []string{"k2"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	// model field = "chain:fast" → use the named chain, skip pool classifier
	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "chain:fast"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200", res.Status)
	}
	if len(res.Attempts) != 2 {
		t.Fatalf("expected 2 attempts (chain a→b), got %d", len(res.Attempts))
	}
	if res.Attempts[0].Provider != "a" || res.Attempts[1].Provider != "b" {
		t.Fatalf("chain order wrong: %+v", res.Attempts)
	}
}

func TestInlineChainCommaSeparated(t *testing.T) {
	s1 := stub(429, ``, nil)
	defer s1.Close()
	s2 := stub(200, `{"choices":[{"message":{"content":"ok"}}]}`, nil)
	defer s2.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"a:x"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: s1.URL, Keys: []string{"k1"}},
			"b": {BaseURL: s2.URL, Keys: []string{"k2"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	// inline chain: model = "a:x,b:y"
	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "a:x,b:y"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d", res.Status)
	}
	if len(res.Attempts) != 2 {
		t.Fatalf("expected 2 inline chain attempts, got %d", len(res.Attempts))
	}
}

func TestSingleProviderModelSkipsClassifier(t *testing.T) {
	good := stub(200, `{"choices":[{"message":{"content":"direct"}}]}`, nil)
	defer good.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"a:x"}, "code": {"a:x"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: good.URL, Keys: []string{"k1"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	// model = "a:x" → should use provider a directly, not pool classifier
	// even though "chat" pool has "a:x", the classifier should be bypassed
	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "a:x"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d", res.Status)
	}
	if len(res.Attempts) != 1 {
		t.Fatalf("single model should be 1 attempt, got %d", len(res.Attempts))
	}
}

func TestAutoModelUsesPoolPassedIn(t *testing.T) {
	good := stub(200, `{"choices":[{"message":{"content":"direct"}}]}`, nil)
	defer good.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"a:x"}, "code": {"a:x"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: good.URL, Keys: []string{"k1"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	// Route uses the pool parameter passed to it; classifier runs in server, not Route
	// model="auto" → Route resolves pool entries by the pool arg
	res, err := r.Route(context.Background(), "code", map[string]any{"model": "auto"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Pool != "code" {
		t.Fatalf("pool should be whatever was passed, got %q", res.Pool)
	}
}
