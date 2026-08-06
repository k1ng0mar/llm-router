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

func TestFiveXXShortCooldown(t *testing.T) {
	p := NewKeyPicker("round_robin", []string{"a", "b"})
	p.SetCooldown(time.Minute)
	p.SetShortCooldown(15 * time.Second)
	now := time.Now()

	idx, _, _ := p.Next(now)
	p.MarkFailure(idx, now, 0, 500) // 5xx → 15s cool

	// 5s later key 0 should still be in cooldown (15 > 5)
	idx2, _, _ := p.Next(now.Add(5 * time.Second))
	if idx2 == 0 {
		t.Fatal("5xx key should be in 15s cooldown after 5s")
	}
	// 16s later key 0 usable again (short cooldown expired)
	idx3, key3, ok3 := p.Next(now.Add(16 * time.Second))
	if !ok3 || idx3 != 0 || key3 != "a" {
		t.Fatalf("5xx key should be usable after 15s cool: idx=%d key=%s ok=%v", idx3, key3, ok3)
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
