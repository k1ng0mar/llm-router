package route

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
)

// hangingStub accepts the request and never sends response headers, so only a
// time-to-first-byte bound can end the attempt. release closes on cleanup so the
// handler goroutines don't outlive the test.
func hangingStub(t *testing.T, hits *atomic.Int32) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits != nil {
			hits.Add(1)
		}
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(func() {
		close(release)
		srv.Close()
	})
	return srv
}

// TestTTFBTimeoutRotatesKeyThenProvider is the timeout contract: each key gets its
// own bounded wait for first byte, a hit moves to the next key, and running out of
// keys moves to the next provider. A timeout never fails the request by itself.
func TestTTFBTimeoutRotatesKeyThenProvider(t *testing.T) {
	var hangHits, goodHits atomic.Int32
	hang := hangingStub(t, &hangHits)
	good := stub(200, `{"choices":[{"message":{"content":"ok"}}]}`, &goodHits)
	defer good.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"slow:m1", "fast:m2"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"slow": {BaseURL: hang.URL, Keys: []string{"k1", "k2", "k3"}},
			"fast": {BaseURL: good.URL, Keys: []string{"k4"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 1, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil)
	r := NewRouter(cfg, g, provider.NewClientWithTTFB(150*time.Millisecond))

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "auto"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200 — the next provider should answer", res.Status)
	}
	// every key on the slow provider is tried exactly once, then we move on
	if got := hangHits.Load(); got != 3 {
		t.Fatalf("hanging provider hits = %d, want 3 (one per key)", got)
	}
	if got := goodHits.Load(); got != 1 {
		t.Fatalf("fast provider hits = %d, want 1", got)
	}
	if len(res.Attempts) != 4 {
		t.Fatalf("attempts = %d, want 4 (3 timed-out keys + 1 success)", len(res.Attempts))
	}
	for _, att := range res.Attempts[:3] {
		if att.Provider != "slow" {
			t.Fatalf("attempt %d provider = %q, want slow", att.Seq, att.Provider)
		}
		if !strings.Contains(att.Err, "timed out") {
			t.Fatalf("attempt %d err = %q, want a legible timeout", att.Seq, att.Err)
		}
	}
	// keys must be distinct: one attempt per key, no re-treading
	seen := map[string]bool{}
	for _, att := range res.Attempts[:3] {
		if seen[att.KeyID] {
			t.Fatalf("key %q was tried twice for the same candidate", att.KeyID)
		}
		seen[att.KeyID] = true
	}
}

// TestSingleKeyTimeoutMovesToNextProvider is the "or next provider if no multiple
// keys" half of the contract.
func TestSingleKeyTimeoutMovesToNextProvider(t *testing.T) {
	var hangHits, goodHits atomic.Int32
	hang := hangingStub(t, &hangHits)
	good := stub(200, `{"choices":[{"message":{"content":"ok"}}]}`, &goodHits)
	defer good.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"slow:m1", "fast:m2"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"slow": {BaseURL: hang.URL, Keys: []string{"only"}},
			"fast": {BaseURL: good.URL, Keys: []string{"k"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 1, Strategy: "round_robin", KeyCooldownS: 60},
	}
	r := NewRouter(cfg, catalog.NewGate(nil), provider.NewClientWithTTFB(150*time.Millisecond))

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "auto"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 || goodHits.Load() != 1 {
		t.Fatalf("status = %d, fast hits = %d — a single-key timeout must fall to the next provider", res.Status, goodHits.Load())
	}
	if hangHits.Load() != 1 {
		t.Fatalf("slow hits = %d, want exactly 1 (its only key)", hangHits.Load())
	}
}

// TestNoOverallDeadlineOnSlowBody pins the "no overall timeout that nukes the
// request" rule: the bound covers the wait for headers only. A response whose body
// takes far longer than timeout_s to finish streaming must still complete.
func TestNoOverallDeadlineOnSlowBody(t *testing.T) {
	ttfb := 150 * time.Millisecond
	trickle := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		w.(http.Flusher).Flush() // headers land inside the TTFB window
		// now spend several TTFB windows delivering the body
		for i := 0; i < 5; i++ {
			time.Sleep(ttfb)
			io.WriteString(w, " ")
			w.(http.Flusher).Flush()
		}
		io.WriteString(w, `{"choices":[{"message":{"content":"slow but complete"}}]}`)
	}))
	defer trickle.Close()

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"p:m"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"p": {BaseURL: trickle.URL, Keys: []string{"k"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 1, Strategy: "round_robin", KeyCooldownS: 60},
	}
	r := NewRouter(cfg, catalog.NewGate(nil), provider.NewClientWithTTFB(ttfb))

	res, err := r.Route(context.Background(), "chat", map[string]any{"model": "auto"}, false, false, false, 10)
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200", res.Status)
	}
	defer res.Resp.Body.Close()
	body, err := io.ReadAll(res.Resp.Body)
	if err != nil {
		t.Fatalf("reading a body slower than timeout_s must not fail: %v", err)
	}
	if !strings.Contains(string(body), "slow but complete") {
		t.Fatalf("body was truncated by a deadline: %q", body)
	}
}

// TestTimeoutDoesNotParkTheKey: a timeout says nothing about the key, so the very
// next request must be free to use it again.
func TestTimeoutDoesNotParkTheKey(t *testing.T) {
	var hangHits atomic.Int32
	hang := hangingStub(t, &hangHits)

	cfg := &config.Config{
		Default: "chat",
		Pools:   map[string][]string{"chat": {"slow:m1"}},
		Providers: config.Providers{Custom: map[string]*config.Provider{
			"slow": {BaseURL: hang.URL, Keys: []string{"only"}},
		}},
		Fallback: config.FallbackCfg{TimeoutS: 1, Strategy: "round_robin", KeyCooldownS: 60},
	}
	r := NewRouter(cfg, catalog.NewGate(nil), provider.NewClientWithTTFB(120*time.Millisecond))

	for i := 1; i <= 2; i++ {
		res, err := r.Route(context.Background(), "chat", map[string]any{"model": "auto"}, false, false, false, 10)
		if err == nil {
			t.Fatalf("request %d: expected exhaustion", i)
		}
		if len(res.Attempts) != 1 {
			t.Fatalf("request %d: attempts = %d, want 1", i, len(res.Attempts))
		}
	}
	if got := hangHits.Load(); got != 2 {
		t.Fatalf("upstream hits = %d, want 2 — a timed-out key must stay usable for the next request", got)
	}
}
