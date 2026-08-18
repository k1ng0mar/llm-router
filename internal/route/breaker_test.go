package route

import (
	"context"
	"encoding/json"
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

// breakerRouter builds a router over the given pool with the circuit breaker
// window set to cooldownS (0 disables it), and a frozen clock the caller
// advances by hand.
func breakerRouter(cooldownS int, pool []string, provs map[string]*config.Provider) (*Router, *time.Time) {
	// Threshold 0 = no escalation, so the single-tier tests stay single-tier.
	return escalatingRouter(cooldownS, 0, 0, pool, provs)
}

// escalatingRouter is breakerRouter with the two-tier knobs exposed.
func escalatingRouter(cooldownS, threshold, lockoutS int, pool []string, provs map[string]*config.Provider) (*Router, *time.Time) {
	cfg := &config.Config{
		Default:   "chat",
		Pools:     map[string][]string{"code": pool},
		Providers: config.Providers{Custom: provs},
		Fallback: config.FallbackCfg{
			TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60,
			ProviderCooldownS:        cooldownS,
			ProviderFailureThreshold: threshold,
			ProviderLockoutS:         lockoutS,
		},
	}
	r := NewRouter(cfg, catalog.NewGate(map[string]catalog.ModelInfo{}), &provider.Client{HTTP: http.DefaultClient})
	clock := time.Now()
	r.nowFn = func() time.Time { return clock }
	return r, &clock
}

func route(t *testing.T, r *Router) *Result {
	t.Helper()
	res, _ := r.Route(context.Background(), "code", map[string]any{"model": "x"}, false, false, false, 10)
	if res != nil && res.Resp != nil {
		res.Resp.Body.Close()
	}
	return res
}

func TestBreakerSkipsCandidateAfter5xx(t *testing.T) {
	var dead, good atomic.Int32
	s1 := stub(500, "", &dead)
	s2 := stub(200, `{"ok":1}`, &good)
	defer s1.Close()
	defer s2.Close()

	r, clock := breakerRouter(60, []string{"p1:m1", "p2:m2"}, map[string]*config.Provider{
		"p1": {BaseURL: s1.URL, Keys: []string{"k1"}},
		"p2": {BaseURL: s2.URL, Keys: []string{"k2"}},
	})

	if res := route(t, r); res.Status != 200 || dead.Load() != 1 {
		t.Fatalf("first request: status=%d p1 hits=%d, want 200 and 1", res.Status, dead.Load())
	}

	// Second request: p1 is cooling down, so it must not be dialed at all —
	// that wasted dial is the whole point of the breaker.
	res := route(t, r)
	if dead.Load() != 1 {
		t.Fatalf("p1 was dialed while cooling down (hits=%d)", dead.Load())
	}
	if res.Status != 200 {
		t.Fatalf("status = %d, want 200 from the healthy candidate", res.Status)
	}
	if len(res.Attempts) != 2 {
		t.Fatalf("want an excluded record plus the real attempt, got %+v", res.Attempts)
	}
	skipped := res.Attempts[0]
	if skipped.Provider != "p1" || skipped.Status != 0 || !strings.HasPrefix(skipped.Err, "excluded") {
		t.Fatalf("first attempt should be an excluded p1 record, got %+v", skipped)
	}

	// Once the window passes, the candidate returns to rotation.
	*clock = clock.Add(61 * time.Second)
	route(t, r)
	if dead.Load() != 2 {
		t.Fatalf("p1 should be retried after the cooldown expired (hits=%d)", dead.Load())
	}
}

// A pool in which every candidate has tripped must still be attempted;
// otherwise one bad minute becomes a hard 503 for the whole window.
func TestBreakerFailsOpenWhenEveryCandidateIsCooling(t *testing.T) {
	var hits atomic.Int32
	s := stub(500, "", &hits)
	defer s.Close()

	r, _ := breakerRouter(60, []string{"p1:m1"}, map[string]*config.Provider{
		"p1": {BaseURL: s.URL, Keys: []string{"k1"}},
	})
	route(t, r)
	if hits.Load() != 1 {
		t.Fatalf("setup: want 1 hit, got %d", hits.Load())
	}
	route(t, r)
	if hits.Load() != 2 {
		t.Fatalf("the only candidate must still be tried when everything is cooling (hits=%d)", hits.Load())
	}
}

// 4xx is about the request or the key, not upstream health, so it must leave
// the candidate in rotation. A 400 isolates the breaker: unlike 401/429 it does
// not park the key either, so a second dial proves the candidate is still live.
func TestBreakerIgnores4xx(t *testing.T) {
	var rejected, good atomic.Int32
	s1 := stub(400, "", &rejected)
	s2 := stub(200, `{"ok":1}`, &good)
	defer s1.Close()
	defer s2.Close()

	r, _ := breakerRouter(60, []string{"p1:m1", "p2:m2"}, map[string]*config.Provider{
		"p1": {BaseURL: s1.URL, Keys: []string{"k1"}},
		"p2": {BaseURL: s2.URL, Keys: []string{"k2"}},
	})
	route(t, r)
	route(t, r)
	if rejected.Load() != 2 {
		t.Fatalf("a 400 must not trip the breaker (p1 hits=%d, want 2)", rejected.Load())
	}
}

func TestBreakerClearedOnSuccess(t *testing.T) {
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if n.Add(1) == 1 {
			w.WriteHeader(500)
			return
		}
		w.Write([]byte(`{"ok":1}`))
	}))
	defer srv.Close()

	r, _ := breakerRouter(60, []string{"p1:m1"}, map[string]*config.Provider{
		"p1": {BaseURL: srv.URL, Keys: []string{"k1"}},
	})
	route(t, r)
	if _, open := r.breakerFor("p1:m1"); !open {
		t.Fatal("500 should have tripped the breaker")
	}
	route(t, r) // fails open, and this time the upstream answers 200
	if _, open := r.breakerFor("p1:m1"); open {
		t.Fatal("a 200 must put the candidate back in rotation")
	}
}

// A cooldown of 0 is the documented off switch.
func TestBreakerDisabledAtZeroCooldown(t *testing.T) {
	var hits atomic.Int32
	s1 := stub(500, "", &hits)
	s2 := stub(200, `{"ok":1}`, nil)
	defer s1.Close()
	defer s2.Close()

	r, _ := breakerRouter(0, []string{"p1:m1", "p2:m2"}, map[string]*config.Provider{
		"p1": {BaseURL: s1.URL, Keys: []string{"k1"}},
		"p2": {BaseURL: s2.URL, Keys: []string{"k2"}},
	})
	route(t, r)
	route(t, r)
	if hits.Load() != 2 {
		t.Fatalf("breaker must stay off at cooldown 0 (p1 hits=%d, want 2)", hits.Load())
	}
}

// End-to-end proof that repair_reasoning_content is actually wired into the
// request path rather than merely being a config field.
func TestRepairReasoningContentReachesUpstream(t *testing.T) {
	bodies := make(chan map[string]any, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var got map[string]any
		json.Unmarshal(raw, &got)
		bodies <- got
		w.Write([]byte(`{"ok":1}`))
	}))
	defer srv.Close()

	payload := func() map[string]any {
		return map[string]any{"model": "x", "messages": []any{assistantToolCall()}}
	}

	for _, tc := range []struct {
		name string
		on   bool
		want any
	}{
		{"enabled injects the placeholder", true, reasoningPlaceholder},
		{"disabled forwards untouched", false, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, _ := breakerRouter(0, []string{"p1:m1"}, map[string]*config.Provider{
				"p1": {BaseURL: srv.URL, Keys: []string{"k1"}, RepairReasoningContent: tc.on},
			})
			res, err := r.Route(context.Background(), "code", payload(), false, false, false, 10)
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			res.Resp.Body.Close()
			sent := <-bodies
			msg := sent["messages"].([]any)[0].(map[string]any)
			if got := msg["reasoning_content"]; got != tc.want {
				t.Fatalf("upstream received reasoning_content=%v, want %v", got, tc.want)
			}
		})
	}
}

// The xkiro case: an upstream that fails at the transport layer (dial refused,
// or a TTFB timeout) is exactly what the breaker exists to stop re-dialing,
// since each retry costs a full per-attempt timeout before falling back.
func TestBreakerTripsOnTransportError(t *testing.T) {
	dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	deadURL := dead.URL
	dead.Close() // nothing is listening now; dials are refused

	var good atomic.Int32
	s2 := stub(200, `{"ok":1}`, &good)
	defer s2.Close()

	r, _ := breakerRouter(60, []string{"p1:m1", "p2:m2"}, map[string]*config.Provider{
		"p1": {BaseURL: deadURL, Keys: []string{"k1"}},
		"p2": {BaseURL: s2.URL, Keys: []string{"k2"}},
	})

	if res := route(t, r); res.Status != 200 {
		t.Fatalf("first request should still fall back to p2, got %d", res.Status)
	}
	if _, open := r.breakerFor("p1:m1"); !open {
		t.Fatal("a transport failure should have tripped the breaker")
	}

	res := route(t, r)
	if res.Status != 200 {
		t.Fatalf("second request status = %d, want 200", res.Status)
	}
	if len(res.Attempts) != 2 || !strings.HasPrefix(res.Attempts[0].Err, "excluded") {
		t.Fatalf("dead candidate should be excluded, not re-dialed: %+v", res.Attempts)
	}
}

// Five consecutive failures escalate from the 60s cooldown to a 10m lockout.
func TestBreakerEscalatesAfterConsecutiveFailures(t *testing.T) {
	var dead atomic.Int32
	s1 := stub(500, "", &dead)
	s2 := stub(200, `{"ok":1}`, nil)
	defer s1.Close()
	defer s2.Close()

	r, clock := escalatingRouter(60, 5, 600, []string{"p1:m1", "p2:m2"}, map[string]*config.Provider{
		"p1": {BaseURL: s1.URL, Keys: []string{"k1"}},
		"p2": {BaseURL: s2.URL, Keys: []string{"k2"}},
	})

	// Each round: p1 is dialed, fails, and cools for 60s. Stepping the clock
	// past the short window is what lets the next failure land on the streak.
	for i := 1; i <= 5; i++ {
		route(t, r)
		if int(dead.Load()) != i {
			t.Fatalf("round %d: p1 dial count = %d, want %d", i, dead.Load(), i)
		}
		st, open := r.breakerFor("p1:m1")
		if !open {
			t.Fatalf("round %d: p1 should be cooling", i)
		}
		if st.fails != i {
			t.Fatalf("round %d: streak = %d, want %d", i, st.fails, i)
		}
		*clock = clock.Add(61 * time.Second)
	}

	// The fifth failure escalated: 61s later it is still locked out, where the
	// plain cooldown would have expired.
	st, open := r.breakerFor("p1:m1")
	if !open {
		t.Fatal("after 5 consecutive failures p1 must still be locked out past the short cooldown")
	}
	if left := st.until.Sub(r.now()); left < 8*time.Minute {
		t.Fatalf("remaining lockout = %s, want ~10m", left)
	}

	res := route(t, r)
	if int(dead.Load()) != 5 {
		t.Fatalf("p1 must not be dialed while locked out (dials=%d)", dead.Load())
	}
	if !strings.Contains(res.Attempts[0].Err, "consecutive failures") {
		t.Fatalf("excluded record should name the streak, got %q", res.Attempts[0].Err)
	}

	// And it does come back once the long window passes.
	*clock = clock.Add(11 * time.Minute)
	route(t, r)
	if int(dead.Load()) != 6 {
		t.Fatalf("p1 should be retried after the lockout expired (dials=%d)", dead.Load())
	}
}

// "Consecutive" means exactly that: one success resets the streak.
func TestBreakerStreakResetBySuccess(t *testing.T) {
	var n atomic.Int32
	flaky := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Fails on every dial except the 3rd.
		if n.Add(1) == 3 {
			w.Write([]byte(`{"ok":1}`))
			return
		}
		w.WriteHeader(500)
	}))
	defer flaky.Close()
	s2 := stub(200, `{"ok":1}`, nil)
	defer s2.Close()

	r, clock := escalatingRouter(60, 5, 600, []string{"p1:m1", "p2:m2"}, map[string]*config.Provider{
		"p1": {BaseURL: flaky.URL, Keys: []string{"k1"}},
		"p2": {BaseURL: s2.URL, Keys: []string{"k2"}},
	})

	// Advance before each round rather than after, so the last failure's
	// cooldown is still open when the streak is inspected.
	for i := 0; i < 6; i++ {
		if i > 0 {
			*clock = clock.Add(61 * time.Second)
		}
		route(t, r)
	}
	// Dials 1,2 failed; 3 succeeded (streak cleared); 4,5,6 failed → streak 3.
	st, _ := r.breakerFor("p1:m1")
	if st.fails != 3 {
		t.Fatalf("streak = %d, want 3 — a success must reset the count", st.fails)
	}
	if left := st.until.Sub(r.now()); left > 2*time.Minute {
		t.Fatalf("still on the short cooldown expected, got %s remaining", left)
	}
}

// A multi-key provider failing on every key is one failure, not one per key —
// otherwise a six-key provider would escalate on its first bad request.
func TestBreakerCountsOneFailurePerRequest(t *testing.T) {
	var dead atomic.Int32
	s1 := stub(500, "", &dead)
	s2 := stub(200, `{"ok":1}`, nil)
	defer s1.Close()
	defer s2.Close()

	r, _ := escalatingRouter(60, 5, 600, []string{"p1:m1", "p2:m2"}, map[string]*config.Provider{
		"p1": {BaseURL: s1.URL, Keys: []string{"k1", "k2", "k3", "k4", "k5", "k6"}},
		"p2": {BaseURL: s2.URL, Keys: []string{"k7"}},
	})
	route(t, r)
	if dead.Load() != 6 {
		t.Fatalf("all six keys should have been tried, got %d dials", dead.Load())
	}
	st, _ := r.breakerFor("p1:m1")
	if st.fails != 1 {
		t.Fatalf("streak = %d after one request, want 1", st.fails)
	}
}

// Threshold 0 keeps the single-tier behaviour: failures never escalate.
func TestBreakerEscalationDisabledAtZeroThreshold(t *testing.T) {
	var dead atomic.Int32
	s1 := stub(500, "", &dead)
	s2 := stub(200, `{"ok":1}`, nil)
	defer s1.Close()
	defer s2.Close()

	r, clock := escalatingRouter(60, 0, 600, []string{"p1:m1", "p2:m2"}, map[string]*config.Provider{
		"p1": {BaseURL: s1.URL, Keys: []string{"k1"}},
		"p2": {BaseURL: s2.URL, Keys: []string{"k2"}},
	})
	for i := 0; i < 6; i++ {
		route(t, r)
		*clock = clock.Add(61 * time.Second)
	}
	if dead.Load() != 6 {
		t.Fatalf("with escalation off every round should re-dial p1, got %d", dead.Load())
	}
}
