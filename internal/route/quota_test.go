package route

// Coverage for transient in-place retry and quota-aware pool ordering.

import (
	"net/http"
	"testing"
	"time"

	"llm-router/internal/config"
)

func TestIsTransientStatus(t *testing.T) {
	cases := map[int]bool{
		http.StatusBadGateway:          true,
		http.StatusServiceUnavailable:  true,
		http.StatusGatewayTimeout:      true,
		http.StatusInternalServerError: false, // deterministic; retry can't fix
		http.StatusTooManyRequests:     false, // cooldown handles it
		http.StatusBadRequest:          false,
		200:                            false,
	}
	for status, want := range cases {
		if got := isTransientStatus(status); got != want {
			t.Errorf("isTransientStatus(%d) = %v, want %v", status, got, want)
		}
	}
}

func TestStreamRequested(t *testing.T) {
	if streamRequested(map[string]any{"stream": true}) != true {
		t.Error("stream=true should be detected")
	}
	if streamRequested(map[string]any{"stream": false}) != false {
		t.Error("stream=false is not streaming")
	}
	if streamRequested(map[string]any{}) != false {
		t.Error("absent stream field is not streaming")
	}
	if streamRequested(map[string]any{"stream": "yes"}) != false {
		t.Error("non-bool stream value must not panic or match")
	}
}

func TestRetryTransientMaxClamp(t *testing.T) {
	cases := map[config.FallbackCfg]int{
		{RetryTransientMax: 0}:  0,
		{RetryTransientMax: 1}:  1,
		{RetryTransientMax: 3}:  3,
		{RetryTransientMax: 99}: 3, // clamped
		{RetryTransientMax: -1}: 0, // negative treated as disabled
	}
	for fb, want := range cases {
		if got := RetryTransientMax(fb); got != want {
			t.Errorf("RetryTransientMax(%+v) = %d, want %d", fb, got, want)
		}
	}
}

func TestTransientBackoffBounded(t *testing.T) {
	base := int64(transientBase / 2)
	for i := 0; i < 4; i++ {
		max := base << (i + 1)
		for range 20 {
			d := transientBackoff(i)
			if int64(d) <= 0 || int64(d) > max {
				t.Fatalf("transientBackoff(%d) = %v outside (0, %v]", i, d, time.Duration(max))
			}
		}
	}
}

func TestQuotaPenaltyBuckets(t *testing.T) {
	now := time.Now()
	future := now.Add(time.Hour).UTC()
	past := now.Add(-time.Minute).UTC()
	low := 5.0
	full := 90.0
	zero := 0.0
	data := map[string]QuotaEntry{
		"lowplan":  {PercentLeft: &low},
		"dryplan":  {PercentLeft: &zero, ResetAt: &future},
		"refilled": {PercentLeft: &zero, ResetAt: &past},
		"fullplan": {PercentLeft: &full},
	}
	cases := map[string]int{
		"lowplan":  1,
		"dryplan":  2,
		"refilled": 0, // reset passed → unknown
		"fullplan": 0,
		"unknown":  0, // absent from file → no penalty
	}
	for name, want := range cases {
		if got := QuotaPenalty(name, data, now); got != want {
			t.Errorf("QuotaPenalty(%q) = %d, want %d", name, got, want)
		}
	}
}

func TestQuotaPenaltyNormalizesNames(t *testing.T) {
	now := time.Now()
	low := 10.0
	data := map[string]QuotaEntry{"openrouter": {PercentLeft: &low}}
	for _, variant := range []string{"OpenRouter", "open-router", "OPENROUTER", "open_router"} {
		if got := QuotaPenalty(variant, data, now); got != 1 {
			t.Errorf("QuotaPenalty(%q) = %d, want 1", variant, got)
		}
	}
}

func TestQuotaSortedEntriesStableAndGrouped(t *testing.T) {
	now := time.Now()
	low := 5.0
	full := 80.0
	data := map[string]QuotaEntry{"a": {PercentLeft: &low}, "c": {PercentLeft: &full}}
	entries := []string{"a:m1", "b:m2", "c:m3", "d:m4"}
	got := QuotaSortedEntries(entries, data, now)
	// Expected: headroom group keeps declared order (b, c, d), then low (a).
	want := []string{"b:m2", "c:m3", "d:m4", "a:m1"}
	if len(got) != len(want) {
		t.Fatalf("length mismatch: %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order = %v, want %v", got, want)
		}
	}
}

func TestQuotaSortedEntriesNoDataUnchanged(t *testing.T) {
	now := time.Now()
	entries := []string{"a:1", "b:2", "c:3"}
	if got := QuotaSortedEntries(entries, nil, now); len(got) != 3 || got[0] != "a:1" {
		t.Errorf("nil data should return entries unchanged, got %v", got)
	}
	if got := QuotaSortedEntries(entries, map[string]QuotaEntry{}, now); got[2] != "c:3" {
		t.Errorf("empty data should return entries unchanged, got %v", got)
	}
}

func TestProviderOf(t *testing.T) {
	cases := map[string]string{
		"gemini:model-x": "gemini",
		"a:b:c":          "a",
		"nocolon":        "nocolon",
		"":               "",
	}
	for ref, want := range cases {
		if got := providerOf(ref); got != want {
			t.Errorf("providerOf(%q) = %q, want %q", ref, got, want)
		}
	}
}

func TestSetQuotaFileEmptyDisables(t *testing.T) {
	SetQuotaFile("")
	if data := quotaState(); data != nil {
		t.Errorf("empty path should disable quota state, got %v", data)
	}
}
