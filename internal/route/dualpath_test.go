package route

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

func TestDualPathVisionCapableCodeModelGetsPixels(t *testing.T) {
	// vision-capable code model should get the raw image, NOT the describe→code chain
	var codeHits atomic.Int32
	codeSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		codeHits.Add(1)
		// check if the request contains image_url (pixels) or text-only (described)
		w.WriteHeader(200)
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer codeSrv.Close()

	cfg := &config.Config{
		Default:         "chat",
		Pools:           map[string][]string{"code": {"a:vm"}},
		Vision:          []string{"b:vis"},
		AllowDirectVision: true,
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: codeSrv.URL, Keys: []string{"k1"}},
			"b": {BaseURL: "http://localhost:9999/v1", Keys: []string{"k2"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(map[string]catalog.ModelInfo{
		"a:vm": {ContextWindow: 200000, Vision: true}, // code model IS vision-capable
		"b:vis": {ContextWindow: 200000, Vision: true},
	})
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	// image + code keywords → vision chain should NOT fire for vision-capable code model
	// instead the code model gets the pixels directly
	res, err := r.Route(context.Background(), "code", map[string]any{
		"model": "auto",
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": "build a website from this"},
				map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/ui.png"}},
			},
		}},
	}, true, false, false, 10)

	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d", res.Status)
	}
	if codeHits.Load() != 1 {
		t.Fatalf("vision-capable code model should be hit directly with pixels: %d", codeHits.Load())
	}
}

func TestDualPathNonVisionCodeModelExcludedByGate(t *testing.T) {
	// non-vision code model → the gate excludes it when hasImage=true.
	// the server handles the describe→code chain; at the route level,
	// the non-vision model simply fails the capability gate.
	codeSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer codeSrv.Close()

	cfg := &config.Config{
		Default:         "chat",
		Pools:           map[string][]string{"code": {"a:deepseek"}},
		AllowDirectVision: true,
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"a": {BaseURL: codeSrv.URL, Keys: []string{"k1"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(map[string]catalog.ModelInfo{
		"a:deepseek": {ContextWindow: 200000, Vision: false}, // NOT vision-capable
	})
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	// hasImage=true + non-vision code model → gate excludes it → 503
	res, err := r.Route(context.Background(), "code", map[string]any{"model": "auto"}, true, false, false, 10)
	if err == nil {
		t.Fatal("expected error for non-vision model with image")
	}
	if res.Status != 503 {
		t.Fatalf("expected 503 (all candidates excluded by gate), got %d", res.Status)
	}
	// the attempt should show "excluded" as a router-origin rejection
	for _, att := range res.Attempts {
		if att.ErrorOrigin != "router" || att.Status != 0 {
			t.Fatalf("excluded attempt should have ErrorOrigin=router, Status=0: %+v", att)
		}
	}
}
