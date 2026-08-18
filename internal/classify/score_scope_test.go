package classify

import "testing"

func TestScoresOnlyLastUserTurn(t *testing.T) {
	h := Heuristics{"code": {"import ", "def "}, "reasoning": {"why "}}

	// system + developer preamble full of "why" and "prove", then a user "import" turn
	msgs := []any{
		map[string]any{"role": "system", "content": "you are a helpful assistant. explain the tradeoffs of everything. prove all claims."},
		map[string]any{"role": "developer", "content": "you must always be thorough. why is everything the way it is."},
		map[string]any{"role": "user", "content": "def parse(json):\n    import os"},
	}
	pool, rule, text, _ := PoolForMedia(h, msgs, "chat", "", "")
	if pool != "code" {
		t.Fatalf("should classify to code from last user turn, not reasoning from preamble; got pool=%q", pool)
	}
	if rule != "code-heuristic" {
		t.Fatalf("rule = %q, want code-heuristic", rule)
	}
	if text != "def parse(json):\n    import os" {
		t.Fatalf("scored text should be just the last user turn, got %q", text)
	}
}

func TestStripsAgentInjectedWrappers(t *testing.T) {
	h := Heuristics{"code": {"import "}, "reasoning": {"why "}}
	// agent-injected preamble that looks like a user message but is instructions
	msgs := []any{
		map[string]any{"role": "user", "content": "<system-reminder>import antigravity</system-reminder>\n<env>TOKEN=bearer</env>\nactually, can you help me?"},
	}
	pool, _, text, _ := PoolForMedia(h, msgs, "chat", "", "")
	// the <system-reminder> contains "import " which would trigger code heuristics
	// — it must be stripped so we don't misroute to code
	if pool == "code" {
		t.Fatalf("agent-injected wrapper must be stripped before scoring; got pool=%q text=%q", pool, text)
	}
	if pool != "chat" {
		t.Fatalf("cleaned to default pool 'chat', got %q", pool)
	}
}

func TestMultiPartUserTurnTextExtracted(t *testing.T) {
	h := Heuristics{"code": {"import "}}
	msgs := []any{
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "import this — what does it do?"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://x.png"}},
		}},
	}
	pool, rule, text, media := PoolForMedia(h, msgs, "default", "", "")
	if !media.Image {
		t.Fatal("image_url part must set media.Image")
	}
	if pool != "code" && rule != "code-heuristic" {
		t.Fatalf("text part with 'import' should classify to code: pool=%q rule=%q", pool, rule)
	}
	if !contains(text, "import this") {
		t.Fatalf("extracted text should contain user text: %q", text)
	}
}

func TestNoUserTurnFallsToDefault(t *testing.T) {
	h := Heuristics{"code": {"import "}}
	msgs := []any{
		map[string]any{"role": "system", "content": "import a module"},
		map[string]any{"role": "user", "content": "hi there"},
	}
	pool, rule, text, media := PoolForMedia(h, msgs, "chat", "", "")
	if pool != "chat" || rule != "default" || text != "hi there" || media.Any() {
		t.Fatalf("system turn must be skipped; got pool=%q rule=%q text=%q media=%+v", pool, rule, text, media)
	}
}

func contains(s, sub string) bool {
	if len(sub) == 0 {
		return true
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
