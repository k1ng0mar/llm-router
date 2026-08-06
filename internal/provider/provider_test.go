package provider

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRoundRobinCycles(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b", "c"})
	want := []int{0, 1, 2, 0, 1}
	for i, w := range want {
		idx, _, ok := p.Next(time.Now())
		if !ok || idx != w {
			t.Fatalf("step %d: got idx=%d ok=%v, want %d", i, idx, ok, w)
		}
	}
}

func TestCooldownSkipsRecentlyFailedKey(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b"})
	now := time.Now()
	idx, _, _ := p.Next(now)
	if idx != 0 {
		t.Fatalf("first pick should be key 0, got %d", idx)
	}
	p.MarkFailure(0, now, 60*time.Second, 429)
	// even though round_robin would pick 1 next anyway, key 0 must be skipped while in cooldown
	idx, key, _ := p.Next(now.Add(5 * time.Second))
	_ = key
	if idx == 0 {
		t.Fatal("key in cooldown must not be picked")
	}
	// after cooldown expires key 0 is usable again (round-robin resumes)
	idx, key, _ = p.Next(now.Add(61 * time.Second))
	if idx != 0 || key != "a" {
		t.Fatalf("expected key 0 after cooldown expiry, got idx=%d key=%q", idx, key)
	}
}

func TestLeastUsedPicksMinUsage(t *testing.T) {
	p := NewKeyPicker("least_used", []string{"a", "b", "c"})
	p.MarkFailure(0, time.Now(), time.Second, 429) // usage: a=1
	// mark success usage on b twice
	p.MarkSuccess(1)
	p.MarkSuccess(1)
	idx, _, ok := p.Next(time.Now())
	if !ok || idx != 2 {
		t.Fatalf("least_used should pick c (usage 0), got %d ok=%v", idx, ok)
	}
}

func TestDoSendsBearerAndPath(t *testing.T) {
	var gotAuth, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		w.WriteHeader(200)
		w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	c := &Client{HTTP: http.DefaultClient}
	up := &Upstream{Name: "stub", BaseURL: srv.URL, Keys: []string{"sk-test-123"}}
	resp, err := c.Do(context.Background(), up, "sk-test-123", map[string]any{"model": "m1"})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if gotAuth != "Bearer sk-test-123" {
		t.Fatalf("auth header = %q", gotAuth)
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path = %q, want /v1/chat/completions", gotPath)
	}
	if string(body) != `{"ok":true}` {
		t.Fatalf("body passthrough broken: %s", body)
	}
}

func TestDoKeylessSendsNoAuth(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.WriteHeader(200)
	}))
	defer srv.Close()

	c := &Client{HTTP: http.DefaultClient}
	up := &Upstream{Name: "local", BaseURL: srv.URL, Keys: nil}
	_, err := c.Do(context.Background(), up, "", map[string]any{"model": "m1"})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if gotAuth != "" {
		t.Fatalf("keyless upstream must send no Authorization, got %q", gotAuth)
	}
}

func TestRetryableAndTerminal(t *testing.T) {
	for _, code := range []int{429, 500, 502, 503, 504} {
		if !Retryable(code) {
			t.Fatalf("%d should be retryable", code)
		}
	}
	for _, code := range []int{200, 400, 401, 403, 404, 422} {
		if Retryable(code) {
			t.Fatalf("%d must NOT be retryable", code)
		}
	}
	for _, code := range []int{400, 401, 403, 404, 422} {
		if !Terminal(code) {
			t.Fatalf("%d should be terminal", code)
		}
	}
	if Terminal(429) || Terminal(500) {
		t.Fatal("429/5xx are fallback triggers, not terminal")
	}
}

func TestConcurrentNextIsSafe(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b", "c", "d", "e"})
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, _, ok := p.Next(time.Now()); !ok {
				t.Error("Next failed")
			}
		}()
	}
	wg.Wait()
}

func TestAllKeysInCooldownReportedUnavailable(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b"})
	now := time.Now()
	p.MarkFailure(0, now, time.Minute, 429)
	p.MarkFailure(1, now, time.Minute, 429)
	_, _, ok := p.Next(now)
	if ok {
		t.Fatal("all keys in cooldown must report unavailable")
	}
}

var _ = strings.TrimSpace
