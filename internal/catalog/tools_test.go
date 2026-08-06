package catalog

import "testing"

func TestToolCallGate(t *testing.T) {
	g := NewGate(map[string]ModelInfo{
		"no-tools/model":    {ContextWindow: 100000, Vision: false, Tools: false},
		"with-tools/model":  {ContextWindow: 100000, Vision: true, Tools: true},
	})
	// request with tools — non-tool model excluded
	ok, reason := g.Check("no-tools/model", false, false, false, 1000, 1)
	if ok {
		t.Fatalf("non-tool model must be excluded when tools requested")
	}
	if reason != "no tool support" {
		t.Fatalf("reason = %q, want 'no tool support'", reason)
	}
	// request with tools — tool model passes
	ok, _ = g.Check("with-tools/model", false, false, false, 1000, 1)
	if !ok {
		t.Fatal("tool-capable model should pass with tools")
	}
}

func TestHasToolCapability(t *testing.T) {
	g := NewGate(map[string]ModelInfo{
		"has-tools":  {Tools: true},
		"no-tools":   {Tools: false},
		"unknown":    {},
	})
	// unknown models fail open — assumed tool-capable
	if !g.HasTools("unknown-model") {
		t.Fatal("unknown model should fail open on tool check")
	}
	if !g.HasTools("has-tools") {
		t.Fatal("explicit tool model should be tool-capable")
	}
	if g.HasTools("no-tools") {
		t.Fatal("no-tools must not report tool-capable")
	}
}
