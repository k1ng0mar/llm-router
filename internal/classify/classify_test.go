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
	pool, rule, _, _ := PoolForMedia(h, userMsg("hi there"), "chat", "code", "")
	if pool != "code" || rule != "hint" {
		t.Fatalf("hint should win: got pool=%q rule=%q", pool, rule)
	}
}

func TestCodeHeuristic(t *testing.T) {
	h := Heuristics{"code": {"def ", "import "}, "reasoning": {"why "}}
	pool, rule, _, _ := PoolForMedia(h, userMsg("def parse(token):\n    import json"), "chat", "", "")
	if pool != "code" || rule != "code-heuristic" {
		t.Fatalf("code signal should classify to code: got pool=%q rule=%q", pool, rule)
	}
}

func TestReasoningHeuristic(t *testing.T) {
	h := Heuristics{"code": {"import "}, "reasoning": {"prove"}}
	pool, rule, _, _ := PoolForMedia(h, userMsg("prove that a b tree is balanced"), "chat", "", "")
	if pool != "reasoning" || rule != "reasoning-heuristic" {
		t.Fatalf("reasoning signal should classify: got pool=%q rule=%q", pool, rule)
	}
}

// With no media pool configured, a media-carrying request keeps the historical
// behavior of landing in the default pool — but the rule is now reported as
// "media" rather than "image", since audio and video take the same path.
func TestMediaFallsThroughToDefaultWhenNoMediaPool(t *testing.T) {
	h := Heuristics{"code": {"import "}}
	pool, rule, _, media := PoolForMedia(h, userMsgImage("what is in this picture"), "chat", "", "")
	if pool != "chat" || rule != "media" {
		t.Fatalf("media with no media pool should fall to default: got pool=%q rule=%q", pool, rule)
	}
	if !media.Image || media.Audio || media.Video {
		t.Fatalf("media = %+v, want image only", media)
	}
}

func TestNoSignalDefaults(t *testing.T) {
	h := Heuristics{"code": {"import "}}
	pool, rule, _, _ := PoolForMedia(h, userMsg("good morning"), "chat", "", "")
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

func userMsgPart(text, partType, field string) []any {
	return []any{map[string]any{"role": "user", "content": []any{
		map[string]any{"type": "text", "text": text},
		map[string]any{"type": partType, field: map[string]any{"url": "https://x"}},
	}}}
}

// Each modality routes to the media pool, not just images — that is the whole
// point of the pool being "media" rather than "vision".
func TestEachModalityRoutesToMediaPool(t *testing.T) {
	h := Heuristics{"code": {"import "}}
	cases := []struct {
		name      string
		partType  string
		field     string
		wantImage bool
		wantAudio bool
		wantVideo bool
	}{
		{"image", "image_url", "image_url", true, false, false},
		{"audio", "input_audio", "input_audio", false, true, false},
		{"video", "video_url", "video_url", false, false, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			msgs := userMsgPart("what is this", c.partType, c.field)
			pool, rule, _, media := PoolForMedia(h, msgs, "chat", "", "media")
			if pool != "media" || rule != "media" {
				t.Fatalf("%s should route to media pool: got pool=%q rule=%q", c.name, pool, rule)
			}
			if media.Image != c.wantImage || media.Audio != c.wantAudio || media.Video != c.wantVideo {
				t.Fatalf("%s: media = %+v", c.name, media)
			}
		})
	}
}

// A keyword heuristic still outranks media, so "explain this code" plus a
// screenshot goes to the code pool where the describe-hop can handle the pixels.
func TestHeuristicOutranksMedia(t *testing.T) {
	h := Heuristics{"code": {"import "}}
	pool, rule, _, media := PoolForMedia(h, userMsgImage("import this and explain the screenshot"), "chat", "", "media")
	if pool != "code" || rule != "code-heuristic" {
		t.Fatalf("heuristic should outrank media: got pool=%q rule=%q", pool, rule)
	}
	if !media.Image {
		t.Fatal("media must still report the image so the gate can act on it")
	}
}

// An explicit pool hint outranks everything, media included.
func TestHintOutranksMedia(t *testing.T) {
	h := Heuristics{"code": {"import "}}
	pool, rule, _, _ := PoolForMedia(h, userMsgImage("what is this"), "chat", "reasoning", "media")
	if pool != "reasoning" || rule != "hint" {
		t.Fatalf("hint should win over media: got pool=%q rule=%q", pool, rule)
	}
}

// Media anywhere in the history counts, not just the last turn: the gate has to
// account for every modality the upstream will actually receive.
func TestDetectMediaSpansWholeHistory(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://x.png"}},
		}},
		map[string]any{"role": "assistant", "content": "a cat"},
		map[string]any{"role": "user", "content": "and now in french"},
	}
	media := DetectMedia(msgs)
	if !media.Image {
		t.Fatal("an image three turns back must still be detected")
	}
	// routing still scores the last turn, but the modality travels with it
	pool, _, _, m := PoolForMedia(Heuristics{}, msgs, "chat", "", "media")
	if pool != "media" || !m.Image {
		t.Fatalf("pool=%q media=%+v, want media pool with image set", pool, m)
	}
}

func TestMediaNamesStable(t *testing.T) {
	got := Media{Image: true, Video: true}.Names()
	if len(got) != 2 || got[0] != "image" || got[1] != "video" {
		t.Fatalf("Names() = %v, want [image video]", got)
	}
	if (Media{}).Any() {
		t.Fatal("zero Media must report Any() == false")
	}
}

func TestStripMediaKeepsTextDropsPixels(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "system", "content": "be terse"},
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "text", "text": "import this and build it"},
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:x"}},
		}},
	}
	out := StripMedia(msgs)
	if len(out) != 2 {
		t.Fatalf("len = %d, want 2 (both turns kept)", len(out))
	}
	if got := out[0].(map[string]any)["content"]; got != "be terse" {
		t.Fatalf("system turn = %v", got)
	}
	last := out[1].(map[string]any)
	if got := last["content"]; got != "import this and build it" {
		t.Fatalf("user text must survive with pixels dropped, got %v", got)
	}
	if last["role"] != "user" {
		t.Fatalf("role must survive, got %v", last["role"])
	}
	if DetectMedia(out).Any() {
		t.Fatal("stripped messages must carry no media")
	}
	// the input must not be mutated — the original body is still logged
	orig := msgs[1].(map[string]any)["content"].([]any)
	if len(orig) != 2 {
		t.Fatal("StripMedia must not mutate its input")
	}
}

func TestStripMediaDropsMediaOnlyTurn(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "user", "content": []any{
			map[string]any{"type": "image_url", "image_url": map[string]any{"url": "data:x"}},
		}},
		map[string]any{"role": "user", "content": "what is it"},
	}
	out := StripMedia(msgs)
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1 — a media-only turn leaves no content to send", len(out))
	}
	if got := out[0].(map[string]any)["content"]; got != "what is it" {
		t.Fatalf("surviving turn = %v", got)
	}
}

func TestAppendToLastUserAttachesToTheQuestion(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "user", "content": "build it"},
		map[string]any{"role": "assistant", "content": "sure"},
		map[string]any{"role": "user", "content": "now with a header"},
	}
	out := AppendToLastUser(msgs, "\n\nDESC")
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3 — no extra message appended", len(out))
	}
	if got := out[2].(map[string]any)["content"]; got != "now with a header\n\nDESC" {
		t.Fatalf("last user turn = %q", got)
	}
	// earlier turns untouched, input unmutated
	if got := out[0].(map[string]any)["content"]; got != "build it" {
		t.Fatalf("earlier user turn changed: %q", got)
	}
	if got := msgs[2].(map[string]any)["content"]; got != "now with a header" {
		t.Fatal("AppendToLastUser must not mutate its input")
	}
}

func TestAppendToLastUserWithNoUserTurn(t *testing.T) {
	msgs := []any{map[string]any{"role": "system", "content": "be terse"}}
	out := AppendToLastUser(msgs, "\n\nDESC")
	if len(out) != 2 {
		t.Fatalf("len = %d, want 2 (a user turn was added)", len(out))
	}
	last := out[1].(map[string]any)
	if last["role"] != "user" || last["content"] != "DESC" {
		t.Fatalf("added turn = %+v", last)
	}
}
