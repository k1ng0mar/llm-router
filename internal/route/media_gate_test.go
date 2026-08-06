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

// TestAudioRequestExcludesNonAudioModel verifies that when an audio input is
// present, only audio-capable models pass the capability gate. A non-audio
// model in the pool must be structurally excluded, so the chain exhausts.
func TestAudioRequestExcludesNonAudioModel(t *testing.T) {
	var hits atomic.Int32
	s1 := stub(200, `{"choices":[{"message":{"content":"ok"}}]}`, &hits)
	defer s1.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"p1:m1", "p2:m2"}},
		Providers: config.Providers{
			Custom: map[string]*config.Provider{
				"p1": {BaseURL: s1.URL, Keys: []string{"k1"}},
				"p2": {BaseURL: s1.URL, Keys: []string{"k2"}},
			},
		},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	// p1:m1 has audio; p2:m2 does NOT.
	g := catalog.NewGate(map[string]catalog.ModelInfo{
		"p1:m1": {ContextWindow: 100000, Audio: true},
		"p2:m2": {ContextWindow: 100000, Audio: false},
	})
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "auto"}, false, true, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	// the audio-capable model should succeed (200) and the non-audio model
	// must NOT have been hit (gate excluded it).
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200 (audio-capable p1:m1 should succeed)", res.Status)
	}
	if hits.Load() != 1 {
		t.Fatalf("non-audio model p2:m2 must be excluded by gate; got %d hits to stub", hits.Load())
	}
}

// TestVideoRequestExcludesNonVideoModel verifies that when a video input is
// present, only video-capable models pass the capability gate.
func TestVideoRequestExcludesNonVideoModel(t *testing.T) {
	var hits atomic.Int32
	s1 := stub(200, `{"choices":[{"message":{"content":"ok"}}]}`, &hits)
	defer s1.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"p1:m1", "p2:m2"}},
		Providers: config.Providers{
			Custom: map[string]*config.Provider{
				"p1": {BaseURL: s1.URL, Keys: []string{"k1"}},
				"p2": {BaseURL: s1.URL, Keys: []string{"k2"}},
			},
		},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	// p1:m1 has video; p2:m2 does NOT.
	g := catalog.NewGate(map[string]catalog.ModelInfo{
		"p1:m1": {ContextWindow: 100000, Video: true},
		"p2:m2": {ContextWindow: 100000, Video: false},
	})
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "auto"}, false, false, true, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200 (video-capable p1:m1 should succeed)", res.Status)
	}
	if hits.Load() != 1 {
		t.Fatalf("non-video model p2:m2 must be excluded by gate; got %d hits to stub", hits.Load())
	}
}
