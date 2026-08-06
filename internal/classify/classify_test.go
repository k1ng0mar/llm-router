package classify

import "testing"

func userMsg(text string) []any {
	return []any{map[string]any{"role": "user", "content": text}}
}

func userMsgImage(text string) []any {
	return []any{map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "text", "text": text},
		map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://x.png"}},
	}}}
}

func TestHintWins(t *testing.T) {
	h := Heuristics{"code": {"def ", "import "}, "reasoning": {"why ", "prove"}}
	pool, rule, _ := PoolForFull(h, userMsg("hi there"), "code", "chat")
	if pool != "code" || rule != "hint" {
		t.Fatalf("hint should win: got pool=%q rule=%q", pool, rule)
	}
}

func TestCodeHeuristic(t *testing.T) {
	h := Heuristics{"code": {"def ", "import "}, "reasoning": {"why "}}
	pool, rule, _ := PoolForFull(h, userMsg("def parse(token):\n    import json"), "", "chat")
	if pool != "code" || rule != "code-heuristic" {
		t.Fatalf("code signal should classify to code: got pool=%q rule=%q", pool, rule)
	}
}

func TestReasoningHeuristic(t *testing.T) {
	h := Heuristics{"code": {"import "}, "reasoning": {"prove"}}
	pool, rule, _ := PoolForFull(h, userMsg("prove that a b tree is balanced"), "", "chat")
	if pool != "reasoning" || rule != "reasoning-heuristic" {
		t.Fatalf("reasoning signal should classify: got pool=%q rule=%q", pool, rule)
	}
}

func TestImageFallsThroughToDefault(t *testing.T) {
	h := Heuristics{"code": {"import "}}
	pool, rule, _ := PoolForFull(h, userMsgImage("what is in this picture"), "", "chat")
	if pool != "chat" || rule != "image" {
		t.Fatalf("image-only should fall to default pool: got pool=%q rule=%q", pool, rule)
	}
}

func TestNoSignalDefaults(t *testing.T) {
	h := Heuristics{"code": {"import "}}
	pool, rule, _ := PoolForFull(h, userMsg("good morning"), "", "chat")
	if pool != "chat" || rule != "default" {
		t.Fatalf("no signal should default: got pool=%q rule=%q", pool, rule)
	}
}

func TestDetectNoMatch(t *testing.T) {
	h := Heuristics{"code": {"import "}, "reasoning": {"prove"}}
	_, ok := Detect(h, "how's the weather")
	if ok {
		t.Fatal("Detect should return false when nothing matches")
	}
}
