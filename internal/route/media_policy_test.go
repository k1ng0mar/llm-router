package route

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

// policyRouter builds a two-candidate pool where each provider carries the given
// per-model media policy, plus a catalog seed the policy is meant to override.
func policyRouter(t *testing.T, s *httptest.Server, seed map[string]catalog.ModelInfo, pol1, pol2 config.MediaPolicy) *Router {
	t.Helper()
	cfg := &config.Config{
		Default: "media",
		Pools:   map[string][]string{"media": {"p1:m1", "p2:m2"}},
		Providers: config.Providers{
			Custom: map[string]*config.Provider{
				"p1": {BaseURL: s.URL, Keys: []string{"k1"}, MediaPolicies: map[string]config.MediaPolicy{"m1": pol1}},
				"p2": {BaseURL: s.URL, Keys: []string{"k2"}, MediaPolicies: map[string]config.MediaPolicy{"m2": pol2}},
			},
		},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	return NewRouter(cfg, catalog.NewGate(seed), &provider.Client{HTTP: http.DefaultClient})
}

// TestMediaPolicyAllowOverridesCatalog covers the "model has native support the
// catalog doesn't know about" case: the catalog says m1 can't see images, the
// operator says it can, and the request must reach it.
func TestMediaPolicyAllowOverridesCatalog(t *testing.T) {
	var hits atomic.Int32
	s := stub(200, `{"choices":[{"message":{"content":"ok"}}]}`, &hits)
	defer s.Close()

	seed := map[string]catalog.ModelInfo{
		"p1:m1": {ContextWindow: 100000, Vision: false},
		"p2:m2": {ContextWindow: 100000, Vision: false},
	}
	r := policyRouter(t, s, seed, config.MediaPolicy{Image: config.PolicyAllow}, config.MediaPolicy{})

	res, err := r.Route(context.Background(), "media", map[string]any{"model": "auto"}, true, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200 — image:allow must beat the catalog's no-vision verdict", res.Status)
	}
	if hits.Load() != 1 {
		t.Fatalf("hits = %d, want 1 (only the allowed model attempted)", hits.Load())
	}
	if got := res.Attempts[0].Model; got != "m1" {
		t.Fatalf("first attempt model = %q, want m1", got)
	}
}

// TestMediaPolicyDenyOverridesCatalog covers the inverse: the catalog claims a
// modality the model does not really have, so the operator denies it and the
// candidate must be skipped without an upstream call.
func TestMediaPolicyDenyOverridesCatalog(t *testing.T) {
	var hits atomic.Int32
	s := stub(200, `{"choices":[{"message":{"content":"ok"}}]}`, &hits)
	defer s.Close()

	seed := map[string]catalog.ModelInfo{
		"p1:m1": {ContextWindow: 100000, Audio: true},
		"p2:m2": {ContextWindow: 100000, Audio: true},
	}
	// p1:m1 is denied audio despite the catalog claiming it.
	r := policyRouter(t, s, seed, config.MediaPolicy{Audio: config.PolicyDeny}, config.MediaPolicy{})

	res, err := r.Route(context.Background(), "media", map[string]any{"model": "auto"}, false, true, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200 (p2:m2 should answer)", res.Status)
	}
	if hits.Load() != 1 {
		t.Fatalf("hits = %d, want 1 — the denied model must not be called", hits.Load())
	}
	first := res.Attempts[0]
	if first.Model != "m1" || first.Status != 0 || first.ErrorOrigin != "router" {
		t.Fatalf("first attempt = %+v, want a router-origin exclusion of m1", first)
	}
	if !strings.Contains(first.Err, "audio denied by per-model media policy") {
		t.Fatalf("exclusion reason = %q, want it to name the audio policy", first.Err)
	}
}

// TestMediaPolicyOnlyAppliesToPresentModalities guards the obvious footgun:
// denying video must not affect a text-only or image-only request.
func TestMediaPolicyOnlyAppliesToPresentModalities(t *testing.T) {
	var hits atomic.Int32
	s := stub(200, `{"choices":[{"message":{"content":"ok"}}]}`, &hits)
	defer s.Close()

	seed := map[string]catalog.ModelInfo{"p1:m1": {ContextWindow: 100000, Vision: true}}
	r := policyRouter(t, s, seed, config.MediaPolicy{Video: config.PolicyDeny}, config.MediaPolicy{})

	// image-only request: the video deny is irrelevant
	res, err := r.Route(context.Background(), "media", map[string]any{"model": "p1:m1"}, true, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200 — a video deny must not block an image request", res.Status)
	}
}

// TestMediaPolicyAllowDoesNotBypassContextGate confirms the allow override is
// scoped to its own modality: it masks the media flag, not the whole gate.
func TestMediaPolicyAllowDoesNotBypassContextGate(t *testing.T) {
	var hits atomic.Int32
	s := stub(200, `{"choices":[{"message":{"content":"ok"}}]}`, &hits)
	defer s.Close()

	seed := map[string]catalog.ModelInfo{"p1:m1": {ContextWindow: 1000, Vision: false}}
	r := policyRouter(t, s, seed, config.MediaPolicy{Image: config.PolicyAllow}, config.MediaPolicy{})

	// minContext far exceeds the model's 1000-token window
	res, err := r.Route(context.Background(), "media", map[string]any{"model": "p1:m1"}, true, false, false, 500000)
	if err == nil {
		t.Fatal("expected exhaustion: the context gate still applies under image:allow")
	}
	if hits.Load() != 0 {
		t.Fatalf("hits = %d, want 0 — context exclusion must precede any upstream call", hits.Load())
	}
	if !strings.Contains(res.Attempts[0].Err, "context") {
		t.Fatalf("exclusion reason = %q, want a context-window exclusion", res.Attempts[0].Err)
	}
}

// TestMediaPolicyDecideMasksAllowedModality is the unit-level statement of the
// masking contract Route depends on.
func TestMediaPolicyDecideMasksAllowedModality(t *testing.T) {
	pol := config.MediaPolicy{Image: config.PolicyAllow, Audio: config.PolicyDeny}

	gi, ga, gv, deny := pol.Decide(true, false, false)
	if deny != "" || gi || ga || gv {
		t.Fatalf("image present + allow: got (%v,%v,%v,%q), want all false and no deny", gi, ga, gv, deny)
	}

	if _, _, _, deny = pol.Decide(false, true, false); deny == "" {
		t.Fatal("audio present + deny: expected a deny reason")
	}

	// video is on auto, so it passes through to the gate untouched
	if _, _, gv, deny = pol.Decide(false, false, true); deny != "" || !gv {
		t.Fatalf("video on auto: got (gv=%v, deny=%q), want gv=true and no deny", gv, deny)
	}
}
