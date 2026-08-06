package route

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

// TestTierOrderingPutsCheapFirst verifies that when Tiers are configured for a
// pool, the router tries the cheapest tier first. With the positional
// convention (entry[i] maps to Tiers[pool][i]), "cheap" sits at tier index 0.
// The pool is declared following the documented convention (cheapest entry
// first). The test asserts the cheap model is tried first and succeeds, and the
// pricey model is never hit — proving tier ordering is wired into Route() and
// does not reverse into the expensive tier. It also asserts the Config-level
// TierSortedEntries helper returns the expected order.
func TestTierOrderingPutsCheapFirst(t *testing.T) {
	var cheapHits, priceyHits atomic.Int32
	cheap := stub(200, `{"choices":[{"message":{"content":"cheap"}}]}`, &cheapHits)
	defer cheap.Close()
	pricey := stub(200, `{"choices":[{"message":{"content":"pricey"}}]}`, &priceyHits)
	defer pricey.Close()

	cfg := &config.Config{
		Default: "chat",
		// Pool declared cheapest-first per the documented YAML convention:
		// entry[0] ("a:cheap-model") corresponds to tier "cheap" (index 0).
		Pools: map[string][]string{"chat": {"a:cheap-model", "b:pricey-model"}},
		Tiers: map[string][]string{"chat": {"cheap", "standard"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: cheap.URL, Keys: []string{"k1"}},
			"b": {BaseURL: pricey.URL, Keys: []string{"k2"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "auto"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200", res.Status)
	}
	if cheapHits.Load() != 1 {
		t.Fatalf("cheap model should be tried first: got %d hits", cheapHits.Load())
	}
	if priceyHits.Load() != 0 {
		t.Fatalf("pricey model should NOT be tried when cheap succeeds: got %d hits", priceyHits.Load())
	}
	if got := res.Attempts[0].Provider; got != "a" {
		t.Fatalf("first attempt provider = %q, want cheap-first \"a\"", got)
	}

	// TierSortedEntries must return entries in tier (cheapest-first) order.
	ordered := cfg.TierSortedEntries("chat")
	if len(ordered) != 2 || ordered[0] != "a:cheap-model" || ordered[1] != "b:pricey-model" {
		t.Fatalf("TierSortedEntries(chat) = %v, want [a:cheap-model b:pricey-model]", ordered)
	}
}

// TestNoTiersUsesPoolOrder verifies that with no Tiers configured for a pool,
// the router uses the pool's entries in the order they are declared (existing
// behavior preserved), and that TierSortedEntries returns the raw pool entries.
func TestNoTiersUsesPoolOrder(t *testing.T) {
	var firstHits, secondHits atomic.Int32
	first := stub(200, `{"ok":1}`, &firstHits)
	defer first.Close()
	second := stub(200, `{"ok":2}`, &secondHits)
	defer second.Close()

	cfg := &config.Config{
		Default: "chat",
		// No Tiers map at all — pool entries are used as-is, in declared order.
		Pools: map[string][]string{"chat": {"a:first", "b:second"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: first.URL, Keys: []string{"k1"}},
			"b": {BaseURL: second.URL, Keys: []string{"k2"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "auto"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200", res.Status)
	}
	if firstHits.Load() != 1 {
		t.Fatalf("first (declared first in pool) should be tried and succeed, got %d hits", firstHits.Load())
	}
	if secondHits.Load() != 0 {
		t.Fatalf("second should NOT be tried when first succeeds, got %d hits", secondHits.Load())
	}
	if got := res.Attempts[0].Provider; got != "a" {
		t.Fatalf("first attempt provider = %q, want a (declared first in pool)", got)
	}

	// TierSortedEntries with no tiers configured returns the raw pool entries.
	ordered := cfg.TierSortedEntries("chat")
	if len(ordered) != 2 || ordered[0] != "a:first" || ordered[1] != "b:second" {
		t.Fatalf("TierSortedEntries(chat) = %v, want [a:first b:second] (raw pool order)", ordered)
	}
}
