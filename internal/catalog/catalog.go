// Package catalog supplies model capability metadata (context window, vision,
// tool-call support) and applies the structural capability gate.
// Unknown models fail open.
package catalog

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
)

// ModelInfo is the capability subset the gate cares about. Both primary and
// alternate JSON spellings (models.dev uses several across exporters) are
// accepted when unmarshaling.
type ModelInfo struct {
	ContextWindow int  `json:"context_window"`
	Context       int  `json:"context"`
	Vision        bool `json:"vision"`
	Multimodal    bool `json:"multimodal"`
	Tools         bool `json:"tools"`
	ToolCall      bool `json:"tool_call"`
	Audio         bool `json:"audio"`
	Video         bool `json:"video"`
}

// Ctx returns the effective context window.
func (m ModelInfo) Ctx() int {
	if m.ContextWindow > 0 {
		return m.ContextWindow
	}
	return m.Context
}

// HasVision reports vision capability.
func (m ModelInfo) HasVision() bool {
	return m.Vision || m.Multimodal
}

// HasTools reports tool-call capability.
func (m ModelInfo) HasTools() bool {
	return m.Tools || m.ToolCall
}

// HasAudio reports audio-input capability.
func (m ModelInfo) HasAudio() bool {
	return m.Audio
}

// HasVideo reports video-input capability.
func (m ModelInfo) HasVideo() bool {
	return m.Video
}

// DefaultSeed holds verified capability facts for models in active use,
// so the router gates correctly even before any catalog fetch.
func DefaultSeed() map[string]ModelInfo {
	return map[string]ModelInfo{
		"xkiro/minimax/m3":                       {ContextWindow: 1000000, Vision: true, Tools: true},
		"xkiro/xiaomi/mimo-v2.5-pro:free":        {ContextWindow: 1000000, Vision: false, Tools: true},
		"agnes/agnes-2.0-flash":                  {ContextWindow: 524288, Vision: true, Tools: true},
		"charm/deepseek-v4-flash":                {ContextWindow: 393000, Vision: false, Tools: true},
		"lightning/lightning-ai/deepseek-v4-pro": {ContextWindow: 1000000, Vision: false, Tools: true},
		"generalcompute/minimax-m2.7":            {ContextWindow: 200000, Vision: false, Tools: true},
	}
}

// Gate merges a hand-seeded baseline with remote catalog refreshes.
// Remote entries override seed entries; seed entries always survive refreshes.
type Gate struct {
	mu     sync.RWMutex
	seed   map[string]ModelInfo
	remote map[string]ModelInfo
}

// NewGate builds a gate from seed entries (seed may be nil).
func NewGate(seed map[string]ModelInfo) *Gate {
	cp := map[string]ModelInfo{}
	for k, v := range seed {
		cp[k] = v
	}
	return &Gate{seed: cp, remote: map[string]ModelInfo{}}
}

func (g *Gate) lookup(model string) (ModelInfo, bool) {
	if info, ok := g.remote[model]; ok {
		return info, true
	}
	info, ok := g.seed[model]
	return info, ok
}

// Check applies the gate: context first, then vision, audio, video, then tools.
// Unknown models fail open (reason "unknown") — only detectable constraints
// (insufficient context, no vision, no audio, no video, no tools) exclude.
// Check signature: Check(model, hasImage, hasAudio, hasVideo, minContext, needsTools)
func (g *Gate) Check(model string, hasImage, hasAudio, hasVideo bool, minContext, needsTools int) (bool, string) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	info, ok := g.lookup(model)
	if !ok {
		return true, "unknown"
	}
	ctx := info.Ctx()
	if ctx > 0 && minContext > ctx {
		return false, fmt.Sprintf("context %d < needed %d", ctx, minContext)
	}
	if hasImage && !info.HasVision() {
		return false, "no vision capability"
	}
	if hasAudio && !info.HasAudio() {
		return false, "no audio capability"
	}
	if hasVideo && !info.HasVideo() {
		return false, "no video capability"
	}
	if needsTools > 0 && !info.HasTools() {
		return false, "no tool support"
	}
	return true, "ok"
}

// HasVision reports whether a model can see images; unknown models return true.
func (g *Gate) HasVision(model string) bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	info, ok := g.lookup(model)
	if !ok {
		return true
	}
	return info.HasVision()
}

// HasTools reports whether a model supports tool calls; unknown returns true.
func (g *Gate) HasTools(model string) bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	info, ok := g.lookup(model)
	if !ok {
		return true
	}
	return info.HasTools()
}

// Refresh fetches the remote catalog and merges it. Failures never clear state.
func (g *Gate) Refresh(ctx context.Context, url string, client *http.Client) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("catalog fetch: HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return err
	}
	return g.RefreshBytes(body)
}

// RefreshBytes merges a raw catalog JSON payload (map[model_id]ModelInfo).
func (g *Gate) RefreshBytes(raw []byte) error {
	var parsed map[string]ModelInfo
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return fmt.Errorf("catalog parse: %w", err)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	for k, v := range parsed {
		if k != "" {
			g.remote[k] = v
		}
	}
	return nil
}
