package provider

import (
	"testing"
	"time"
)

func TestMarkFailureCooldownWithRetryAfter(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b"})
	p.SetCooldown(time.Minute)
	p.SetRetryAfterCap(5 * time.Minute)
	now := time.Now()

	// 429 with Retry-After: 120 → cooldown should be 120s, not the default 60s
	idx, key, ok := p.Next(now)
	if !ok || idx != 0 || key != "a" {
		t.Fatalf("first pick: idx=%d key=%s ok=%v", idx, key, ok)
	}
	p.MarkFailure(idx, now, 120*time.Second, 429)

	// 5s later key 0 should still be in cooldown (429 > 60s would've ended by 65s)
	_, _, ok = p.Next(now.Add(5 * time.Second))
	// key 0 in cooldown, round_robin should pick key 1
	idx2, key2, ok2 := p.Next(now.Add(5 * time.Second))
	if !ok2 || key2 != "b" {
		// key 0 cooldown not expired, should pick key 1
		if idx2 == 0 {
			t.Fatal("key 0 should be in cooldown (Retry-After=120s) after only 5s")
		}
	}
	// 65s later key 0 should still be in cooldown (120 > 65)
	idx3, key3, ok3 := p.Next(now.Add(65 * time.Second))
	if idx3 == 0 && ok3 && key3 == "a" {
		t.Fatal("key 0 should still be in cooldown after 65s with Retry-After=120s")
	}
	// 121s later key 0 should be usable again
	idx4, key4, ok4 := p.Next(now.Add(121 * time.Second))
	if !ok4 || idx4 != 0 || key4 != "a" {
		t.Fatalf("key 0 should be usable after Retry-After cooldown expires: idx=%d key=%s ok=%v", idx4, key4, ok4)
	}
}

func TestMarkFailureDefaultCooldownNoRetryAfter(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b"})
	p.SetCooldown(time.Minute)
	now := time.Now()

	idx, key, _ := p.Next(now)
	if idx != 0 || key != "a" {
		t.Fatalf("first pick: idx=%d key=%s", idx, key)
	}
	// 429 without Retry-After → default cooldown
	p.MarkFailure(idx, now, 0, 429)

	// 30s later key 0 in cooldown, key 1 not
	p.Next(now.Add(30 * time.Second))
	idx2, key2, ok2 := p.Next(now.Add(30 * time.Second))
	if !ok2 || key2 == "a" || idx2 == 0 {
		// key 0 should be skipped (cooldown), pick key 1
		if ok2 && key2 == "a" {
			t.Fatal("key a should be in cooldown after 30s with default 60s cooldown")
		}
	}
	// 61s later key 0 usable again
	idx3, key3, ok3 := p.Next(now.Add(61 * time.Second))
	if !ok3 || idx3 != 0 || key3 != "a" {
		t.Fatalf("key 0 should be usable after default cooldown: idx=%d key=%s ok=%v", idx3, key3, ok3)
	}
}

func TestFourOFKeyMarkedDead(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b"})
	p.SetCooldown(time.Minute)
	now := time.Now()

	idx, key, ok := p.Next(now)
	if !ok || key != "a" {
		t.Fatalf("first pick: idx=%d key=%s ok=%v", idx, key, ok)
	}
	// 401 → key is DEAD, not just cooldown
	p.MarkFailure(idx, now, 0, 401)

	// 61s later (past default cooldown) key should STILL not be usable
	_, key2, ok2 := p.Next(now.Add(61 * time.Second))
	if ok2 && key2 == "a" {
		t.Fatal("401'd key must remain dead even past cooldown window")
	}
	// key b is still usable
	_, key3, ok3 := p.Next(now.Add(61 * time.Second))
	if !ok3 || key3 != "b" {
		t.Fatalf("key b should be usable: key=%s ok=%v", key3, ok3)
	}
}

// A 5xx describes the upstream or the model behind it, not the key. Since keys
// are shared by every model on a provider, cooling one for a 5xx stranded that
// provider's whole model list — so it no longer affects availability at all.
func TestFiveXXDoesNotCoolKey(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a"})
	p.SetCooldown(time.Minute)
	now := time.Now()

	idx, _, _ := p.Next(now)
	p.MarkFailure(idx, now, 0, 500)

	// immediately usable again — a single-key provider must not go dark
	idx2, key2, ok2 := p.Next(now)
	if !ok2 || idx2 != 0 || key2 != "a" {
		t.Fatalf("a 5xx must leave the key usable: idx=%d key=%s ok=%v", idx2, key2, ok2)
	}
}

// Same for the other non-429, non-401 outcomes the router sees.
func TestTransientFailuresDoNotCoolKey(t *testing.T) {
	for _, status := range []int{0 /* transport error / TTFB timeout */, 400, 403, 404, 422, 500, 502, 503} {
		p := NewKeyPicker("round_robin", []string{"a"})
		p.SetCooldown(time.Minute)
		now := time.Now()
		idx, _, _ := p.Next(now)
		p.MarkFailure(idx, now, 0, status)
		if _, _, ok := p.Next(now); !ok {
			t.Fatalf("status %d must not park the key", status)
		}
	}
}

// The cooldown no longer terminates a fallback loop, so the loop's bound has to
// come from the caller: NextExcluding lets it try each key exactly once.
func TestNextExcludingBoundsTheLoop(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b", "c"})
	now := time.Now()

	tried := map[int]bool{}
	var got []string
	for {
		idx, key, ok := p.NextExcluding(now, tried)
		if !ok {
			break
		}
		if tried[idx] {
			t.Fatalf("NextExcluding returned an excluded key: %d", idx)
		}
		tried[idx] = true
		got = append(got, key)
		p.MarkFailure(idx, now, 0, 500) // would previously have parked it
		if len(got) > 3 {
			t.Fatal("loop did not terminate after every key was tried")
		}
	}
	if len(got) != 3 {
		t.Fatalf("tried %v, want all three keys exactly once", got)
	}
}

// Exclusions stack with the real states: a dead key and a cooling key stay out.
func TestNextExcludingRespectsDeadAndCooling(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b", "c"})
	p.SetCooldown(time.Minute)
	now := time.Now()

	p.MarkFailure(0, now, 0, 401) // dead
	p.MarkFailure(1, now, 0, 429) // cooling
	idx, key, ok := p.NextExcluding(now, nil)
	if !ok || key != "c" || idx != 2 {
		t.Fatalf("got idx=%d key=%s ok=%v, want the one live key (c)", idx, key, ok)
	}
	// now exclude the only live key too
	if _, _, ok := p.NextExcluding(now, map[int]bool{2: true}); ok {
		t.Fatal("no key should be available once the live one is excluded")
	}
}

func TestRetryAfterCapped(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b"})
	p.SetCooldown(time.Minute)
	p.SetRetryAfterCap(10 * time.Second)
	now := time.Now()

	idx, _, _ := p.Next(now)
	p.MarkFailure(idx, now, 300*time.Second, 429) // Retry-After claims 300s, cap at 10s

	// 11s later key 0 should be usable again (9s over the capped 10s)
	// round-robin may pick it or rotate to key 1 first — just verify key 0 is available
	_, _, ok0 := p.Next(now.Add(11 * time.Second))
	// exhaust key 1 by failing it too
	p.MarkFailure(1, now.Add(11*time.Second), time.Minute, 429)
	idx2, key2, ok2 := p.Next(now.Add(11 * time.Second))
	if !ok2 || idx2 != 0 || key2 != "a" {
		t.Fatalf("Retry-After should be capped: idx=%d key=%s ok=%v", idx2, key2, ok2)
	}
	_ = ok0
}
