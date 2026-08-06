package catalog

import "testing"

func seedGate() *Gate {
	return NewGate(map[string]ModelInfo{
		"xkiro/minimax/m3":            {ContextWindow: 1000000, Vision: true},
		"charm/deepseek-v4-flash":     {ContextWindow: 393000, Vision: false},
		"ollama/llama3:8b":            {ContextWindow: 4096, Vision: false},
	})
}

func TestVisionCapablePassesWithImage(t *testing.T) {
	g := seedGate()
	ok, reason := g.Check("xkiro/minimax/m3", true, false, false, 1000, 0)
	if !ok {
		t.Fatalf("vision model with image should pass, got reason %q", reason)
	}
}

func TestNonVisionExcludedWithImage(t *testing.T) {
	g := seedGate()
	ok, reason := g.Check("charm/deepseek-v4-flash", true, false, false, 1000, 0)
	if ok {
		t.Fatal("non-vision model with image must be structurally excluded")
	}
	if reason == "" {
		t.Fatal("exclusion should carry a reason")
	}
}

func TestUnknownModelFailsOpen(t *testing.T) {
	g := seedGate()
	ok, reason := g.Check("brandnew/provider/model", true, false, false, 1000, 0)
	if !ok {
		t.Fatalf("unknown model must fail open (not classified), got reason %q", reason)
	}
	if reason != "unknown" {
		t.Fatalf("unknown model reason should be 'unknown', got %q", reason)
	}
}

func TestContextGateExcludesSmallWindow(t *testing.T) {
	g := seedGate()
	ok, _ := g.Check("ollama/llama3:8b", false, false, false, 50000, 0)
	if ok {
		t.Fatal("model with 4k context must be excluded when request needs 50k")
	}
}

func TestContextGateAllowsEnoughWindow(t *testing.T) {
	g := seedGate()
	ok, _ := g.Check("ollama/llama3:8b", false, false, false, 2000, 0)
	if !ok {
		t.Fatal("model with 4k context should pass a 2k request")
	}
}

func TestRefreshMergesRemoteWithoutClearingSeed(t *testing.T) {
	g := seedGate()
	remote := []byte(`{"charm/deepseek-v4-flash":{"context_window":500000,"vision":false},"remote/new-model":{"context_window":128000,"vision":true}}`)
	if err := g.RefreshBytes(remote); err != nil {
		t.Fatalf("RefreshBytes: %v", err)
	}
	ok, _ := g.Check("xkiro/minimax/m3", true, false, false, 1000, 0) // seed must survive
	if !ok {
		t.Fatal("seed entries must survive a remote refresh")
	}
	ok, _ = g.Check("remote/new-model", true, false, false, 1000, 0) // remote entry visible
	if !ok {
		t.Fatal("remote entry should be visible after refresh")
	}
}

func TestRefreshHandlesAlternateFieldSpellings(t *testing.T) {
	g := seedGate()
	remote := []byte(`{"alt/model":{"context":8192,"multimodal":true}}`)
	if err := g.RefreshBytes(remote); err != nil {
		t.Fatalf("RefreshBytes: %v", err)
	}
	ok, _ := g.Check("alt/model", true, false, false, 1000, 0)
	if !ok {
		t.Fatal("alternate spellings context/multimodal must be recognized")
	}
}
