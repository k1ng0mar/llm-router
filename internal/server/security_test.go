package server

// Coverage for the security middleware: response headers and auth backoff.

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"llm-router/internal/config"
)

func TestSecurityHeadersOnEveryResponse(t *testing.T) {
	cfg := &config.Config{RouterKey: "k", Listen: "127.0.0.1:0"}
	srv := New(cfg, nil, nil, nil)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	cases := map[string]string{
		"Content-Security-Policy": "",
		"X-Content-Type-Options":  "nosniff",
		"X-Frame-Options":         "DENY",
		"Referrer-Policy":         "no-referrer",
	}
	for h := range cases {
		if resp.Header.Get(h) == "" {
			t.Errorf("missing security header %s", h)
		}
	}
	if got := resp.Header.Get("X-Content-Type-Options"); got != "nosniff" {
		t.Errorf("X-Content-Type-Options = %q", got)
	}
	if got := resp.Header.Get("X-Frame-Options"); got != "DENY" {
		t.Errorf("X-Frame-Options = %q", got)
	}
}

func TestAuthBackoffBlocksAfterRepeatedFailures(t *testing.T) {
	tr := newAuthBackoffTracker()
	now := time.Now()
	ip := "10.1.1.1"

	for i := 0; i < authBackoffMaxFailures-1; i++ {
		blocked, _ := tr.record(ip, now)
		if blocked {
			t.Fatalf("blocked after only %d failures", i+1)
		}
	}
	blocked, retryIn := tr.record(ip, now)
	if !blocked {
		t.Fatal("not blocked after threshold failures")
	}
	if retryIn <= 0 {
		t.Fatal("Retry-After duration should be positive")
	}

	// A different IP is unaffected.
	if blocked, _ := tr.record("10.1.1.2", now); blocked {
		t.Fatal("unrelated IP got blocked")
	}
}

func TestAuthBackoffWindowExpires(t *testing.T) {
	tr := newAuthBackoffTracker()
	now := time.Now()
	ip := "10.2.2.2"
	for i := 0; i < authBackoffMaxFailures; i++ {
		tr.record(ip, now)
	}
	if blocked, _ := tr.blocked(ip, now); !blocked {
		t.Fatal("expected active block")
	}
	later := now.Add(authBackoffWindow + time.Second)
	if blocked, _ := tr.blocked(ip, later); blocked {
		t.Fatal("block should expire after the window")
	}
	// And a new failure streak starts fresh.
	if blocked, _ := tr.record(ip, later); blocked {
		t.Fatal("post-expiry failure should start a new streak, not block")
	}
}

func TestAuthBackoffClearOnSuccess(t *testing.T) {
	tr := newAuthBackoffTracker()
	now := time.Now()
	ip := "10.3.3.3"
	for i := 0; i < authBackoffMaxFailures-1; i++ {
		tr.record(ip, now)
	}
	tr.clear(ip)
	if blocked, _ := tr.record(ip, now); blocked {
		t.Fatal("cleared IP should not block on the next single failure")
	}
}

func TestClientIPPrefersForwardedFor(t *testing.T) {
	r := httptest.NewRequest("POST", "/", nil)
	r.RemoteAddr = "192.168.0.9:55555"
	if got := clientIP(r); got != "192.168.0.9" {
		t.Errorf("RemoteAddr fallback = %q", got)
	}
	r.Header.Set("X-Forwarded-For", "203.0.113.5, 10.0.0.1")
	if got := clientIP(r); got != "203.0.113.5" {
		t.Errorf("XFF extraction = %q", got)
	}
}

func TestEndToEndBackoffBlocksBruteForce(t *testing.T) {
	cfg := &config.Config{RouterKey: "sekret", Listen: "127.0.0.1:0"}
	srv := New(cfg, nil, nil, nil)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	url := ts.URL + "/v1/models"
	var resp *http.Response
	for i := 0; i < authBackoffMaxFailures; i++ {
		req, _ := http.NewRequest("GET", url, nil)
		req.Header.Set("Authorization", "Bearer wrong-key")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}
	// The next bad attempt is now blocked with 429 + Retry-After.
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Authorization", "Bearer wrong-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected 429 after %d failures, got %d", authBackoffMaxFailures, resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Fatal("missing Retry-After on blocked response")
	}
	// And even the CORRECT key is blocked while the IP is locked out — that is
	// the point of brute-force protection.
	req2, _ := http.NewRequest("GET", url, nil)
	req2.Header.Set("Authorization", "Bearer sekret")
	resp2, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("blocked IP should stay blocked even with the right key, got %d", resp2.StatusCode)
	}
}
