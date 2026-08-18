package catalog

import "strings"

// modelsDevProvider is one top-level entry of https://models.dev/api.json.
// The payload is keyed by provider id, and the capability facts live one level
// down under "models" — not in the flat model_id → info map this package
// originally assumed.
type modelsDevProvider struct {
	Models map[string]modelsDevModel `json:"models"`
}

// modelsDevModel is the subset of a models.dev model entry the gate can use.
// Input modality is a list ("text", "image", "audio", "video") rather than the
// booleans ModelInfo carries, and the context window is nested under "limit".
type modelsDevModel struct {
	ToolCall   bool `json:"tool_call"`
	Modalities struct {
		Input []string `json:"input"`
	} `json:"modalities"`
	Limit struct {
		Context int `json:"context"`
	} `json:"limit"`
}

// info projects a models.dev entry onto the gate's capability shape. Anything
// outside the known input modalities (notably "text", which every model takes)
// is ignored rather than guessed at.
func (m modelsDevModel) info() ModelInfo {
	info := ModelInfo{ContextWindow: m.Limit.Context, Tools: m.ToolCall}
	for _, in := range m.Modalities.Input {
		switch strings.ToLower(strings.TrimSpace(in)) {
		case "image":
			info.Vision = true
		case "audio":
			info.Audio = true
		case "video":
			info.Video = true
		}
	}
	return info
}

// isZero reports whether an entry carries no usable fact. Such entries are
// dropped rather than stored: a ModelInfo that claims nothing is
// indistinguishable from one that denies everything, and storing it would turn
// a model the gate should fail open on into one it excludes.
func (m ModelInfo) isZero() bool {
	return m.Ctx() == 0 && !m.HasVision() && !m.HasAudio() && !m.HasVideo() && !m.HasTools()
}

// widen merges two views of the same model into the most permissive one.
//
// The bare-model index can see the same id from several providers, and Go
// randomizes map iteration, so "last one wins" would make the catalog differ
// between restarts. Union is both deterministic and the safe direction: this
// router treats any non-200 as a fallback trigger, so over-stating a capability
// costs one attempt that rotates onward, while under-stating it removes a
// candidate that would have worked.
func widen(a, b ModelInfo) ModelInfo {
	out := ModelInfo{
		ContextWindow: a.Ctx(),
		Vision:        a.HasVision() || b.HasVision(),
		Audio:         a.HasAudio() || b.HasAudio(),
		Video:         a.HasVideo() || b.HasVideo(),
		Tools:         a.HasTools() || b.HasTools(),
	}
	if b.Ctx() > out.ContextWindow {
		out.ContextWindow = b.Ctx()
	}
	return out
}

// splitRef splits a routing ref ("provider:model") at the first colon. Model
// ids routinely contain both slashes and further colons
// ("xkiro:xiaomi/mimo-v2.5:free"), so only the first separator is structural.
func splitRef(ref string) (provider, model string, ok bool) {
	i := strings.Index(ref, ":")
	if i <= 0 || i == len(ref)-1 {
		return "", "", false
	}
	return ref[:i], ref[i+1:], true
}

// altSeparator rewrites "provider:model" as "provider/model". Seed entries have
// historically been written with a slash while the router looks models up with
// a colon, so without this the hand-verified seed never matched at runtime.
func altSeparator(ref string) (string, bool) {
	p, m, ok := splitRef(ref)
	if !ok {
		return "", false
	}
	return p + "/" + m, true
}
