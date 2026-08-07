package route

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

func TestStripParamsDropsFields(t *testing.T) {
	var capturedBody string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		capturedBody = string(raw)
		w.WriteHeader(200)
		w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer upstream.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"groq:model"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"groq": {
				BaseURL:    upstream.URL,
				Keys:       []string{"k1"},
				StripParams: []string{"reasoning_effort", "reasoning"},
			},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	r := NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})

	payload := map[string]any{
		"model":            "model",
		"reasoning_effort": "high",
		"reasoning":        "high",
		"messages":         []map[string]any{{"role": "user", "content": "hi"}},
	}
	res, err := r.Route(context.Background(), "chat", payload, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200", res.Status)
	}
	if contains(capturedBody, "reasoning_effort") {
		t.Fatalf("reasoning_effort should be stripped, body: %s", capturedBody)
	}
	if contains(capturedBody, "reasoning") {
		t.Fatalf("reasoning should be stripped, body: %s", capturedBody)
	}
	if !contains(capturedBody, "messages") {
		t.Fatalf("messages should be preserved, body: %s", capturedBody)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || (len(s) > 0 && containsStr(s, sub)))
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
