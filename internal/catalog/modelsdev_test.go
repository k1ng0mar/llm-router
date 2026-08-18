package catalog

import "testing"

// A trimmed slice of the real https://models.dev/api.json shape.
const modelsDevFixture = `{
  "groq": {
    "id": "groq", "name": "Groq",
    "models": {
      "llama-3.3-70b-versatile": {
        "tool_call": true,
        "modalities": {"input": ["text"], "output": ["text"]},
        "limit": {"context": 131072, "output": 32768}
      },
      "whisper-large-v3": {
        "tool_call": false,
        "modalities": {"input": ["audio"], "output": ["text"]},
        "limit": {"context": 0, "output": 0}
      }
    }
  },
  "openrouter": {
    "id": "openrouter",
    "models": {
      "omni-30b": {
        "tool_call": true,
        "modalities": {"input": ["text", "image", "video"], "output": ["text"]},
        "limit": {"context": 200000, "output": 8192}
      }
    }
  },
  "emptyprovider": {"id": "emptyprovider", "name": "Nothing Here", "models": {}}
}`

func modelsDevGate(t *testing.T) *Gate {
	t.Helper()
	g := NewGate(nil)
	if err := g.RefreshBytes([]byte(modelsDevFixture)); err != nil {
		t.Fatalf("RefreshBytes: %v", err)
	}
	return g
}

func TestModelsDevPopulatesInputModalities(t *testing.T) {
	g := modelsDevGate(t)
	cases := []struct {
		name                   string
		ref                    string
		image, audio, video    bool
		minContext, needsTools int
		wantOK                 bool
	}{
		{"text-only model rejects an image", "groq:llama-3.3-70b-versatile", true, false, false, 0, 0, false},
		{"text-only model rejects audio", "groq:llama-3.3-70b-versatile", false, true, false, 0, 0, false},
		{"text-only model takes text", "groq:llama-3.3-70b-versatile", false, false, false, 1000, 0, true},
		{"audio model takes audio", "groq:whisper-large-v3", false, true, false, 0, 0, true},
		{"audio model rejects an image", "groq:whisper-large-v3", true, false, false, 0, 0, false},
		{"omni model takes image and video", "openrouter:omni-30b", true, false, true, 0, 0, true},
		{"omni model rejects audio it cannot take", "openrouter:omni-30b", false, true, false, 0, 0, false},
		{"context window is read from limit.context", "groq:llama-3.3-70b-versatile", false, false, false, 200000, 0, false},
		{"context window admits a fitting request", "groq:llama-3.3-70b-versatile", false, false, false, 100000, 0, true},
		{"tool_call true passes the tools gate", "groq:llama-3.3-70b-versatile", false, false, false, 0, 1, true},
		{"tool_call false fails the tools gate", "groq:whisper-large-v3", false, false, false, 0, 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ok, reason := g.Check(tc.ref, tc.image, tc.audio, tc.video, tc.minContext, tc.needsTools)
			if ok != tc.wantOK {
				t.Fatalf("Check(%q) ok=%v reason=%q, want ok=%v", tc.ref, ok, reason, tc.wantOK)
			}
			if reason == "unknown" {
				t.Fatalf("Check(%q) fell through to unknown — the catalog entry was not found", tc.ref)
			}
		})
	}
}

// Capabilities belong to the model, not the gateway reselling it, so a model
// served by a provider models.dev has never heard of still gates correctly.
func TestModelsDevMatchesBareModelIDAcrossProviders(t *testing.T) {
	g := modelsDevGate(t)
	if ok, reason := g.Check("some-random-gateway:omni-30b", true, false, false, 0, 0); !ok {
		t.Fatalf("bare model id should carry its capabilities across providers, got %q", reason)
	}
	if ok, _ := g.Check("some-random-gateway:llama-3.3-70b-versatile", true, false, false, 0, 0); ok {
		t.Fatal("a text-only model must reject an image whichever gateway serves it")
	}
}

// The old parser turned each of models.dev's ~190 providers into an empty
// ModelInfo, which reads as "this model can do nothing" rather than "unknown".
func TestModelsDevDropsEntriesWithNoFacts(t *testing.T) {
	g := modelsDevGate(t)
	for _, ref := range []string{"emptyprovider", "emptyprovider:whatever", "groq"} {
		ok, reason := g.Check(ref, true, false, false, 0, 0)
		if !ok || reason != "unknown" {
			t.Fatalf("Check(%q) = (%v, %q), want (true, \"unknown\") — a provider name is not a model", ref, ok, reason)
		}
	}
}

// Runtime asks with "provider:model"; seeds have always been written
// "provider/model". Before this, the hand-verified seed never matched.
func TestSeedMatchesColonFormAtRuntime(t *testing.T) {
	g := NewGate(map[string]ModelInfo{
		"xkiro/minimax/m3":        {ContextWindow: 1000000, Vision: true, Tools: true},
		"charm/deepseek-v4-flash": {ContextWindow: 393000, Vision: false, Tools: true},
	})
	if ok, reason := g.Check("xkiro:minimax/m3", true, false, false, 1000, 0); !ok || reason != "ok" {
		t.Fatalf("seeded vision model = (%v, %q), want (true, \"ok\")", ok, reason)
	}
	if ok, reason := g.Check("charm:deepseek-v4-flash", true, false, false, 1000, 0); ok {
		t.Fatalf("seeded non-vision model should be excluded, got (%v, %q)", ok, reason)
	}
}

// Two providers can describe the same model differently; the merge must not
// depend on Go's randomized map iteration order.
func TestBareModelIndexMergesDeterministically(t *testing.T) {
	payload := []byte(`{
	  "a": {"models": {"shared": {"tool_call": false, "modalities": {"input": ["text"]}, "limit": {"context": 8192}}}},
	  "b": {"models": {"shared": {"tool_call": true, "modalities": {"input": ["text","image"]}, "limit": {"context": 128000}}}}
	}`)
	for i := 0; i < 25; i++ {
		g := NewGate(nil)
		if err := g.RefreshBytes(payload); err != nil {
			t.Fatalf("RefreshBytes: %v", err)
		}
		if ok, reason := g.Check("third-party:shared", true, false, false, 100000, 1); !ok {
			t.Fatalf("run %d: widened entry should allow image+tools+128k, got %q", i, reason)
		}
		// The provider-scoped entries stay exact — only the bare index widens.
		if ok, _ := g.Check("a:shared", true, false, false, 0, 0); ok {
			t.Fatalf("run %d: provider-scoped entry for a:shared must stay text-only", i)
		}
	}
}

// A flat custom catalog and models.dev can arrive in the same payload.
func TestRefreshAcceptsFlatAndNestedTogether(t *testing.T) {
	g := NewGate(nil)
	payload := []byte(`{
	  "groq": {"models": {"m1": {"modalities": {"input": ["text","image"]}, "limit": {"context": 1000}}}},
	  "custom/legacy-model": {"context_window": 4096, "vision": true}
	}`)
	if err := g.RefreshBytes(payload); err != nil {
		t.Fatalf("RefreshBytes: %v", err)
	}
	if ok, reason := g.Check("groq:m1", true, false, false, 0, 0); !ok {
		t.Fatalf("nested entry lost: %q", reason)
	}
	if ok, reason := g.Check("custom/legacy-model", true, false, false, 0, 0); !ok || reason != "ok" {
		t.Fatalf("flat entry lost: (%v, %q)", ok, reason)
	}
}
