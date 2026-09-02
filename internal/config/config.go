// Package config loads and validates the router YAML config.
package config

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	neturl "net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// isPrivateHost reports whether host refers to a private, loopback, link-local,
// or otherwise non-public address. Used to block SSRF via user-controlled URLs.
func isPrivateHost(host string) bool {
	// Fast path for localhost names.
	if host == "localhost" || host == "127.0.0.1" || host == "::1" {
		return true
	}
	// Resolve and classify via netip.
	addr, err := netip.ParseAddr(host)
	if err != nil {
		// If it resolves to any private range, treat as private.
		addrs, _ := net.LookupHost(host)
		for _, a := range addrs {
			if pa, perr := netip.ParseAddr(a); perr == nil && (pa.IsLoopback() || pa.IsPrivate() || pa.IsLinkLocalUnicast() || pa.IsLinkLocalMulticast() || pa.IsUnspecified()) {
				return true
			}
		}
		return false
	}
	return addr.IsLoopback() || addr.IsPrivate() || addr.IsLinkLocalUnicast() || addr.IsLinkLocalMulticast() || addr.IsUnspecified()
}

const defaultCatalogURL = "https://models.dev/api.json"

// keepSentinel marks a key the caller wants preserved without transmitting it.
// Prefer the indexed form "__KEEP__:<n>"; see SetProviderKeys.
const keepSentinel = "__KEEP__"

// Media policy values, used per model per modality. A modality left empty (or
// set to PolicyAuto) defers to the model catalog. PolicyAllow forces the
// modality through even when the catalog says the model can't take it — use it
// when a model has native support the catalog under-reports. PolicyDeny
// excludes the model from any request carrying that modality, whatever the
// catalog claims.
const (
	PolicyAuto  = "auto"
	PolicyAllow = "allow"
	PolicyDeny  = "deny"
)

// MediaPolicy is a per-model override of catalog media capability, one value
// per modality. The zero value defers entirely to the catalog, so a provider
// that has never been touched behaves exactly as it did before policies existed.
type MediaPolicy struct {
	Image string `yaml:"image" json:"image"`
	Audio string `yaml:"audio" json:"audio"`
	Video string `yaml:"video" json:"video"`
}

// Stance returns this policy's value for one modality ("image", "audio",
// "video"), normalized to PolicyAuto when unset or unrecognized. Callers can
// switch on the result without re-checking for the empty string.
func (m MediaPolicy) Stance(modality string) string {
	var v string
	switch modality {
	case "image":
		v = m.Image
	case "audio":
		v = m.Audio
	case "video":
		v = m.Video
	}
	switch v {
	case PolicyAllow, PolicyDeny:
		return v
	default:
		return PolicyAuto
	}
}

// IsZero reports whether the policy expresses no opinion at all, so callers can
// skip persisting or rendering it.
func (m MediaPolicy) IsZero() bool {
	return m.Stance("image") == PolicyAuto &&
		m.Stance("audio") == PolicyAuto &&
		m.Stance("video") == PolicyAuto
}

// Validate rejects unrecognized policy values so a typo in router.yaml surfaces
// at load time rather than silently degrading to "auto" on every request.
func (m MediaPolicy) Validate() error {
	for _, f := range []struct{ name, val string }{{"image", m.Image}, {"audio", m.Audio}, {"video", m.Video}} {
		switch f.val {
		case "", PolicyAuto, PolicyAllow, PolicyDeny:
		default:
			return fmt.Errorf("media policy %s = %q must be one of auto, allow, deny", f.name, f.val)
		}
	}
	return nil
}

// Decide resolves the policy against the modalities a request actually carries.
// It returns the modality flags to hand the catalog gate — a modality this
// policy explicitly allows is masked off, so the gate cannot veto it — plus a
// non-empty reason when a modality that IS present is explicitly denied.
// Modalities absent from the request are always false and never produce a
// reason: denying video says nothing about a text-only request.
func (m MediaPolicy) Decide(hasImage, hasAudio, hasVideo bool) (gateImage, gateAudio, gateVideo bool, denyReason string) {
	one := func(present bool, modality string) (bool, string) {
		if !present {
			return false, ""
		}
		switch m.Stance(modality) {
		case PolicyDeny:
			return false, modality + " denied by per-model media policy"
		case PolicyAllow:
			return false, "" // allowed outright: don't let the catalog veto it
		default:
			return true, "" // auto: let the catalog decide
		}
	}
	for _, c := range []struct {
		present  bool
		modality string
		out      *bool
	}{
		{hasImage, "image", &gateImage},
		{hasAudio, "audio", &gateAudio},
		{hasVideo, "video", &gateVideo},
	} {
		flag, reason := one(c.present, c.modality)
		if reason != "" {
			return false, false, false, reason
		}
		*c.out = flag
	}
	return gateImage, gateAudio, gateVideo, ""
}

// Provider is one upstream endpoint with a pool of API keys.
// Enabled controls whether the router will route to this provider. When
// false, the provider is skipped as if it's not in any pool. This is a
// soft toggle — the provider's config (keys, base_url) is preserved.
// ModelLimits optionally caps max_tokens per model id, applied as an
// upper bound before the request is sent upstream.
//
// A provider is configured with EITHER a base_url (single-endpoint mode)
// OR a list of deployment_links (multi-endpoint mode). The two are mutually
// exclusive — Validate() rejects a provider that has both or neither.
type Provider struct {
	BaseURL     string         `yaml:"base_url"`
	Keys        []string       `yaml:"keys"`
	Enabled     *bool          `yaml:"enabled"` // pointer so "nil" = enabled (default), false = disabled
	ModelLimits map[string]int `yaml:"model_limits"`
	// AccountID is an opaque per-provider identifier (e.g. a Cloudflare
	// account id) substituted into base_url at request time via the
	// "{account_id}" placeholder. Empty means no substitution.
	AccountID string `yaml:"account_id"`
	// DisabledModels is a list of model ids that should be skipped during
	// candidate selection even when the provider is enabled. This allows
	// per-model disable without disabling the whole provider.
	DisabledModels []string `yaml:"disabled_models"`
	// Preset marks a provider as a built-in/preset entry in the dashboard
	// (shown in the "Preset Providers" group). Custom providers added via the
	// API default to false and appear under "Custom Providers".
	Preset bool `yaml:"preset"`
	// APIMode selects the upstream wire format: "openai" (default),
	// "anthropic", or "gemini". Non-OpenAI modes translate the request and
	// response at the provider boundary so the router stays OpenAI-shaped.
	APIMode string `yaml:"api_mode"`
	// StripParams is a list of request-body fields to drop before
	// forwarding to this provider. Use when an upstream rejects params
	// that other providers accept (e.g. groq rejects `reasoning_effort`).
	StripParams []string `yaml:"strip_params"`
	// RepairReasoningContent applies to DeepSeek-family gateways (e.g.
	// TokenRouter) that reject a conversation history where an assistant
	// message carrying tool_calls lacks a non-empty `reasoning_content`
	// field, answering 400 "reasoning_content is required for thinking
	// tool-call history". OpenAI-compatible clients routinely drop that
	// field when building follow-up requests. When enabled, the router
	// injects a placeholder before forwarding so the request round-trips
	// instead of bouncing.
	RepairReasoningContent bool `yaml:"repair_reasoning_content"`
	// KeyLabels are operator-chosen nicknames for the entries in Keys, aligned
	// by index ("prod", "billing account", "burner"). Purely descriptive — the
	// router never routes on them — but with several interchangeable keys on one
	// provider they are the only way to tell which one is in cooldown or dead.
	// Shorter than Keys (or absent) is fine; the dashboard falls back to
	// "Key <n>" for any entry without a label.
	KeyLabels []string `yaml:"key_labels"`
	// MediaPolicies overrides catalog media capability per model id (same
	// keying as ModelLimits). It lets an operator force a modality through
	// for a model the catalog under-reports, or deny one the catalog wrongly
	// claims — which is what keeps a mixed-modality `media` pool from
	// handing an audio request to an image-only model. Absent entries defer
	// entirely to the catalog.
	MediaPolicies map[string]MediaPolicy `yaml:"media_policies"`
	// Deployments holds per-deployment endpoints for multi-tenant providers
	// (e.g. Modal: every deployment is its own hostname + key pair). Each
	// deployment is addressable in pool refs as "<provider>/<deployment>:<model>"
	// and resolves to its own base_url + keys. The parent provider's own
	// BaseURL/Keys are untouched and can serve as a default.
	Deployments map[string]*Deployment `yaml:"deployments,omitempty"`
	// DeploymentLinks is an independent list of deployment URLs for providers
	// like Modal where each URL serves exactly one model. Each link carries
	// its own API key. When set, BaseURL must be empty (mutually exclusive).
	// All links pool together under one provider name; models are auto-discovered
	// from each link's /v1/models endpoint.
	DeploymentLinks []DeploymentLink `yaml:"deployment_links"`
	// modelLinks maps a model id to the DeploymentLink that serves it.
	// Populated at fetch time (not persisted) — the server fills this after
	// discovering which model lives at which URL. Access only under c.mu.
	modelLinks map[string]DeploymentLink `yaml:"-" json:"-"`
}

// Deployment is one named endpoint under a multi-deployment provider. Each
// deployment serves exactly one model: Model is the upstream model id sent
// in requests to BaseURL. In pool refs the deployment NAME is the model slot
// ("Modal:glm53"); the router rewrites it to Model at the upstream boundary.
type Deployment struct {
	BaseURL string   `yaml:"base_url"`
	Keys    []string `yaml:"keys"`
	// Model is the upstream model id this deployment serves. Required — a
	// deployment without one would silently send a garbage model name.
	Model string `yaml:"model"`
	Enabled *bool `yaml:"enabled,omitempty"` // nil = enabled (default)
}

// DeploymentLink is one independent deployment URL within a provider.
// Each URL serves exactly one model (auto-discovered via /v1/models) and
// carries its own API key. Multiple DeploymentLinks pool together under
// one provider name.
type DeploymentLink struct {
	URL string `yaml:"url" json:"url"`
	Key string `yaml:"key" json:"key"`
}

// ShortHash returns a short hash of the URL for use in breaker keys
// so that each deployment link gets its own circuit-breaker state.
func (d DeploymentLink) ShortHash() string {
	h := sha256.Sum256([]byte(d.URL))
	return fmt.Sprintf("%x", h[:4])
}

// HasDeploymentLinks reports whether this provider uses deployment links
// (multi-endpoint mode) instead of a single base_url.
func (p *Provider) HasDeploymentLinks() bool {
	return len(p.DeploymentLinks) > 0
}

// EffectiveBaseURL returns the base_url if set, otherwise the first
// deployment link's URL. For backwards compatibility in places that
// just need *a* URL from the provider.
func (p *Provider) EffectiveBaseURL() string {
	if p.BaseURL != "" {
		return p.BaseURL
	}
	if len(p.DeploymentLinks) > 0 {
		return p.DeploymentLinks[0].URL
	}
	return ""
}

// DeploymentLinkCount returns the number of deployment links configured.
func (p *Provider) DeploymentLinkCount() int {
	return len(p.DeploymentLinks)
}

// Providers groups provider kinds. A provider is addressed as its name:
// "openrouter", "ollama", or any key under custom.
type Providers struct {
	OpenRouter *Provider            `yaml:"openrouter"`
	Ollama     *Provider            `yaml:"ollama"`
	Custom     map[string]*Provider `yaml:"custom"`
}

// Get resolves a provider by name.
func (p Providers) Get(name string) (*Provider, bool) {
	switch name {
	case "openrouter":
		return p.OpenRouter, p.OpenRouter != nil
	case "ollama":
		return p.Ollama, p.Ollama != nil
	default:
		pr, ok := p.Custom[name]
		return pr, ok
	}
}

// ClassifierCfg drives the per-request pool classification.
type ClassifierCfg struct {
	HintsFirst bool                `yaml:"hints_first"`
	Heuristics map[string][]string `yaml:"heuristics"`
}

// FallbackCfg controls failover behavior.
type FallbackCfg struct {
	TimeoutS     int    `yaml:"timeout_s"`
	Strategy     string `yaml:"strategy"`
	KeyCooldownS int    `yaml:"key_cooldown_s"`
	// ProviderCooldownS is the circuit-breaker window: after a provider:model
	// candidate fails with a transport error or 5xx, it is skipped (recorded as
	// an "excluded" attempt) for this many seconds instead of being re-dialed at
	// the front of the pool on every request. 0 disables it. Guards against a
	// dead upstream silently adding its full per-attempt timeout to every
	// request in a pool.
	ProviderCooldownS int `yaml:"provider_cooldown_s"`
	// ProviderFailureThreshold is how many *consecutive* hard failures a
	// provider:model candidate may take before the short cooldown escalates to
	// ProviderLockoutS. The streak survives the short cooldown expiring — that
	// is what lets a persistently dead upstream escalate instead of being
	// re-probed every minute forever — and only a 200 resets it. 0 disables
	// escalation, leaving every failure on the short cooldown.
	ProviderFailureThreshold int `yaml:"provider_failure_threshold"`
	// ProviderLockoutS is the escalated cooldown applied once a candidate has
	// failed ProviderFailureThreshold times in a row. Named "lockout" rather
	// than "timeout" because timeout_s already means something else here (the
	// per-attempt TTFB deadline). 0 disables escalation.
	ProviderLockoutS int `yaml:"provider_lockout_s"`
	// RetryTransientMax is how many times a non-streamed attempt that received
	// a 502/503/504 (with no Retry-After) is retried in place on the same key
	// before falling through to normal fallback. A gateway blip shouldn't burn
	// the next candidate's quota. 0 disables; values above 3 are clamped to 3.
	RetryTransientMax int `yaml:"retry_transient_max"`
}

// Config is the whole router configuration.
type Config struct {
	mu                sync.RWMutex
	Path              string              `yaml:"-"`
	DBPath            string              `yaml:"db"`
	Listen            string              `yaml:"listen"`
	RouterKey         string              `yaml:"router_key"`
	InsecureNoAuth    bool                `yaml:"insecure_no_auth"`
	CatalogURL        string              `yaml:"catalog_url"`
	QuotaFile         string              `yaml:"quota_file"`
	Default           string              `yaml:"default"`
	Pools             map[string][]string `yaml:"pools"`
	Chains            map[string][]string `yaml:"chains"`
	Tiers             map[string][]string `yaml:"tiers"`
	PoolContext       map[string]int      `yaml:"pool_context,omitempty"`
	AllowDirectVision bool                `yaml:"allow_direct_vision"`
	Vision            []string            `yaml:"vision"`
	Providers         Providers           `yaml:"providers"`
	Classifier        ClassifierCfg       `yaml:"classifier"`
	Fallback          FallbackCfg         `yaml:"fallback"`
}

// DefaultConfig returns a sane starting config.
func DefaultConfig() *Config {
	return &Config{
		Listen:     "127.0.0.1:8011",
		DBPath:     "router.db",
		CatalogURL: defaultCatalogURL,
		Default:    "chat",
		Pools: map[string][]string{
			"chat":  {},
			"code":  {},
			"media": {},
		},
		PoolContext:       map[string]int{},
		Chains:            map[string][]string{},
		Tiers:             map[string][]string{},
		AllowDirectVision: true,
		Providers:         Providers{Custom: map[string]*Provider{}},
		Classifier: ClassifierCfg{
			HintsFirst: true,
			Heuristics: map[string][]string{
				"code": {"def ", "import ", "func ", "select ", "gradle", "npm ", "git ", "docker", "python", "class ", "const ", "function "},
			},
		},
		Fallback: FallbackCfg{
			TimeoutS: 90, Strategy: "round_robin", KeyCooldownS: 60,
			ProviderCooldownS: 60, ProviderFailureThreshold: 5, ProviderLockoutS: 600,
		},
	}
}

// Load parses config from a reader and fills defaults for zero fields.
func Load(r io.Reader) (*Config, error) {
	base := DefaultConfig()
	dec := yaml.NewDecoder(r)
	if err := dec.Decode(base); err != nil && !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	if base.DBPath == "" {
		base.DBPath = "router.db"
	}
	if base.Listen == "" {
		base.Listen = "127.0.0.1:8011"
	}
	if base.CatalogURL == "" {
		base.CatalogURL = defaultCatalogURL
	}
	if base.Default == "" {
		base.Default = "chat"
	}
	if base.Fallback.TimeoutS == 0 {
		base.Fallback.TimeoutS = 90
	}
	if base.Fallback.Strategy == "" {
		base.Fallback.Strategy = "round_robin"
	}
	if base.Fallback.KeyCooldownS == 0 {
		base.Fallback.KeyCooldownS = 60
	}
	if base.Providers.Custom == nil {
		base.Providers.Custom = map[string]*Provider{}
	}
	base.autodetectProviderStates()
	if base.Classifier.Heuristics == nil {
		base.Classifier.Heuristics = map[string][]string{}
	}
	if err := base.Validate(); err != nil {
		return nil, err
	}
	return base, nil
}

// LoadFile loads config from a YAML file path.
func LoadFile(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	c, err := Load(f)
	if err != nil {
		return nil, err
	}
	c.Path = path
	return c, nil
}

// Validate checks the config is internally consistent.
func (c *Config) Validate() error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if _, ok := c.Pools[c.Default]; !ok {
		return fmt.Errorf("default pool %q does not exist in pools", c.Default)
	}
	for pool, entries := range c.Pools {
		for _, ref := range entries {
			if _, _, err := c.resolveNoLock(ref); err != nil {
				return fmt.Errorf("pool %q entry %q: %w", pool, ref, err)
			}
		}
	}
	for _, ref := range c.Vision {
		if _, _, err := c.resolveNoLock(ref); err != nil {
			return fmt.Errorf("vision entry %q: %w", ref, err)
		}
	}
	// media policies: reject typos at load time rather than degrading to "auto"
	// silently on every request.
	for name, p := range c.allProvidersNoLock() {
		for model, pol := range p.MediaPolicies {
			if err := pol.Validate(); err != nil {
				return fmt.Errorf("provider %q model %q: %w", name, model, err)
			}
		}
		// A provider must have EITHER base_url OR at least one deployment_link,
		// but NOT both. Neither set means no upstream to call; both set is
		// ambiguous (which does the router use?).
		hasBase := p.BaseURL != ""
		hasLinks := len(p.DeploymentLinks) > 0
		if hasBase && hasLinks {
			return fmt.Errorf("provider %q has both base_url and deployment_links — set one or the other", name)
		}
		if !hasBase && !hasLinks {
			// Built-in/local providers (openrouter, ollama) may not have a base_url
			// configured if the user removed it — but they should still be valid
			// as placeholders. Skip validation for them.
			if name == "openrouter" || name == "ollama" {
				continue
			}
			return fmt.Errorf("provider %q has neither base_url nor deployment_links", name)
		}
		// validate each deployment link has a URL
		for i, dl := range p.DeploymentLinks {
			if dl.URL == "" {
				return fmt.Errorf("provider %q deployment_links[%d] missing url", name, i)
			}
		}
	}
	switch c.Fallback.Strategy {
	case "round_robin", "least_used":
	default:
		return fmt.Errorf("fallback.strategy %q must be round_robin or least_used", c.Fallback.Strategy)
	}
	// validate chains: every entry must resolve to a known provider
	for name, refs := range c.Chains {
		for _, ref := range refs {
			if _, _, err := c.resolveNoLock(ref); err != nil {
				return fmt.Errorf("chain %q entry %q: %w", name, ref, err)
			}
		}
	}
	// validate tiers: every tier list must be non-empty
	for pool, tiers := range c.Tiers {
		if len(tiers) == 0 {
			return fmt.Errorf("tiers for pool %q is empty", pool)
		}
	}
	return nil
}

// allProvidersNoLock returns every configured provider keyed by its routing
// name, including the two built-ins. Caller must hold the lock.
func (c *Config) allProvidersNoLock() map[string]*Provider {
	out := make(map[string]*Provider, len(c.Providers.Custom)+2)
	if c.Providers.OpenRouter != nil {
		out["openrouter"] = c.Providers.OpenRouter
	}
	if c.Providers.Ollama != nil {
		out["ollama"] = c.Providers.Ollama
	}
	for name, p := range c.Providers.Custom {
		if p != nil {
			out[name] = p
		}
	}
	return out
}

// resolveNoLock is the internal resolver (caller must hold lock).
func (c *Config) resolveNoLock(ref string) (string, string, error) {
	prov, model, ok := strings.Cut(ref, ":")
	if !ok || prov == "" || model == "" {
		return "", "", fmt.Errorf("bad model ref %q (want provider:model)", ref)
	}
	if _, ok := c.Providers.Get(prov); !ok {
		return "", "", fmt.Errorf("unknown provider %q in ref %q", prov, ref)
	}
	return prov, model, nil
}

// Resolve splits a pool entry "provider:model" into provider name and model id
// (model ids may contain slashes, so split on the first colon only).
func (c *Config) Resolve(ref string) (string, string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.resolveNoLock(ref)
}

// Redacted returns a JSON-safe copy of the config with key material masked.
func (c *Config) Redacted() map[string]any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	mask := func(k string) string {
		if len(k) <= 4 {
			return "****"
		}
		return k[:2] + "**" + k[len(k)-2:]
	}
	maskDL := func(links []DeploymentLink) []map[string]any {
		out := make([]map[string]any, 0, len(links))
		for _, dl := range links {
			out = append(out, map[string]any{"url": dl.URL, "key": mask(dl.Key)})
		}
		return out
	}
	provs := map[string]any{}
	if c.Providers.OpenRouter != nil {
		provs["openrouter"] = map[string]any{"base_url": c.Providers.OpenRouter.BaseURL, "keys": maskList(c.Providers.OpenRouter.Keys, mask), "enabled": c.Providers.OpenRouter.IsEnabled(), "disabled_models": c.Providers.OpenRouter.DisabledModels, "strip_params": c.Providers.OpenRouter.StripParams, "key_labels": c.Providers.OpenRouter.KeyLabels, "model_limits": c.Providers.OpenRouter.ModelLimits, "media_policies": c.Providers.OpenRouter.MediaPolicies, "repair_reasoning_content": c.Providers.OpenRouter.RepairReasoningContent, "preset": true}
	}
	if c.Providers.Ollama != nil {
		provs["ollama"] = map[string]any{"base_url": c.Providers.Ollama.BaseURL, "keys": maskList(c.Providers.Ollama.Keys, mask), "enabled": c.Providers.Ollama.IsEnabled(), "disabled_models": c.Providers.Ollama.DisabledModels, "strip_params": c.Providers.Ollama.StripParams, "key_labels": c.Providers.Ollama.KeyLabels, "model_limits": c.Providers.Ollama.ModelLimits, "media_policies": c.Providers.Ollama.MediaPolicies, "repair_reasoning_content": c.Providers.Ollama.RepairReasoningContent, "preset": true}
	}
	for name, p := range c.Providers.Custom {
		entry := map[string]any{"base_url": p.BaseURL, "account_id": p.AccountID, "keys": maskList(p.Keys, mask), "enabled": p.IsEnabled(), "disabled_models": p.DisabledModels, "strip_params": p.StripParams, "key_labels": p.KeyLabels, "model_limits": p.ModelLimits, "media_policies": p.MediaPolicies, "repair_reasoning_content": p.RepairReasoningContent, "preset": p.Preset, "api_mode": p.APIMode}
		if len(p.DeploymentLinks) > 0 {
			entry["deployment_links"] = maskDL(p.DeploymentLinks)
		}
		provs[name] = entry
	}
	heuristics := map[string][]string{}
	for pool, kws := range c.Classifier.Heuristics {
		heuristics[pool] = kws
	}
	return map[string]any{
		"default":    c.Default,
		"pools":      c.Pools,
		"chains":     c.Chains,
		"tiers":      c.Tiers,
		"vision":     c.Vision,
		"providers":  provs,
		"classifier": map[string]any{"heuristics": heuristics},
		"fallback":   map[string]any{"timeout_s": c.Fallback.TimeoutS, "strategy": c.Fallback.Strategy, "key_cooldown_s": c.Fallback.KeyCooldownS, "provider_cooldown_s": c.Fallback.ProviderCooldownS, "provider_failure_threshold": c.Fallback.ProviderFailureThreshold, "provider_lockout_s": c.Fallback.ProviderLockoutS},
	}
}

func maskList(keys []string, mask func(string) string) []string {
	out := make([]string, 0, len(keys))
	for _, k := range keys {
		if k == "" {
			out = append(out, "")
			continue
		}
		out = append(out, mask(k))
	}
	return out
}

// autodetectProviderStates auto-enables custom providers that have keys
// and auto-disables keyless ones (except localhost). Only acts when
// Enabled is nil (not explicitly set by the user).
func (c *Config) autodetectProviderStates() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.autodetectProviderStatesLocked()
}

// IsEnabled reports whether the provider is enabled.
func (p *Provider) IsEnabled() bool {
	if p == nil || p.Enabled == nil {
		return true
	}
	return *p.Enabled
}

// IsModelDisabled reports whether the given model id is in the provider's
// DisabledModels list. A disabled model is skipped during candidate selection
// even when the provider itself is enabled.
func (p *Provider) IsModelDisabled(model string) bool {
	if p == nil {
		return false
	}
	for _, m := range p.DisabledModels {
		if m == model {
			return true
		}
	}
	return false
}

// MediaPolicyFor returns the media policy for a model id. A provider or model
// with no policy configured yields the zero policy, which defers to the catalog.
func (p *Provider) MediaPolicyFor(model string) MediaPolicy {
	if p == nil {
		return MediaPolicy{}
	}
	return p.MediaPolicies[model]
}

// ToggleProvider enables or disables a provider by name. Persists to disk.
func (c *Config) ToggleProvider(name string, enabled bool) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	p, ok := c.Providers.Get(name)
	if !ok {
		return fmt.Errorf("provider %q does not exist", name)
	}
	p.Enabled = &enabled
	return c.persistNoLock()
}

// ToggleModel adds or removes a model from a provider's DisabledModels list.
// When disabled is true, the model is added (disabled); when false, it's
// removed (re-enabled). Persists to disk.
func (c *Config) ToggleModel(provider, model string, disabled bool) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	p, ok := c.Providers.Get(provider)
	if !ok {
		return fmt.Errorf("provider %q does not exist", provider)
	}
	if disabled {
		for _, m := range p.DisabledModels {
			if m == model {
				return nil // already disabled — idempotent
			}
		}
		p.DisabledModels = append(p.DisabledModels, model)
	} else {
		var kept []string
		for _, m := range p.DisabledModels {
			if m != model {
				kept = append(kept, m)
			}
		}
		p.DisabledModels = kept
	}
	return c.persistNoLock()
}

// SetProviderModelSettings updates model_limits, disabled_models and
// media_policies for a provider. Policies that express no opinion are dropped
// rather than persisted, so router.yaml doesn't accumulate an "auto/auto/auto"
// entry for every model the dashboard has ever rendered.
func (c *Config) SetProviderModelSettings(name string, modelLimits map[string]int, disabledModels []string, mediaPolicies map[string]MediaPolicy) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	p, ok := c.Providers.Get(name)
	if !ok {
		return fmt.Errorf("provider %q not found", name)
	}
	cleaned := map[string]MediaPolicy{}
	for model, pol := range mediaPolicies {
		if err := pol.Validate(); err != nil {
			return fmt.Errorf("model %q: %w", model, err)
		}
		if pol.IsZero() {
			continue
		}
		cleaned[model] = pol
	}
	p.ModelLimits = modelLimits
	p.DisabledModels = disabledModels
	p.MediaPolicies = cleaned
	return c.persistNoLock()
}

// persistNoLock writes the config to disk atomically: the new YAML goes to a
// temp file in the same directory, is fsynced, and is then renamed over the
// target, so a reader never observes a half-written file and a crash mid-save
// leaves the old config intact. The previous contents are kept alongside as
// "<path>.bak" first.
//
// The care is warranted: this file holds every provider API key, and a plain
// in-place truncate-and-write that fails partway destroys credentials that
// exist nowhere else.
func (c *Config) persistNoLock() error {
	if c.Path == "" {
		return nil
	}
	out, err := yaml.Marshal(c)
	if err != nil {
		return err
	}
	// Best effort: a backup we could not write must not block a valid save.
	if prev, rerr := os.ReadFile(c.Path); rerr == nil && len(prev) > 0 {
		_ = os.WriteFile(c.Path+".bak", prev, 0o600)
	}
	tmp, err := os.CreateTemp(filepath.Dir(c.Path), ".router-config-*.yaml")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	// No-op once the rename below succeeds; on any early return it cleans up.
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(out); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, c.Path)
}

// SetDefault records a new default pool in memory and on disk.
func (c *Config) SetDefault(pool string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.Pools[pool]; !ok {
		return fmt.Errorf("pool %q does not exist", pool)
	}
	c.Default = pool
	return c.persistNoLock()
}

// SetPool replaces a pool's entries in memory and on disk.
func (c *Config) SetPool(name string, entries []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.Pools[name]; !ok {
		return fmt.Errorf("pool %q does not exist", name)
	}
	c.Pools[name] = entries
	return c.persistNoLock()
}

// SetChain creates or updates a named chain in memory and on disk.
func (c *Config) SetChain(name string, entries []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if name == "" {
		return fmt.Errorf("chain name cannot be empty")
	}
	if c.Chains == nil {
		c.Chains = map[string][]string{}
	}
	// validate each entry resolves
	for _, ref := range entries {
		if _, _, err := c.resolveNoLock(ref); err != nil {
			return fmt.Errorf("chain %q entry %q: %w", name, ref, err)
		}
	}
	c.Chains[name] = entries
	return c.persistNoLock()
}

// RemoveChain deletes a chain by name.
func (c *Config) RemoveChain(name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.Chains == nil {
		return fmt.Errorf("chain %q does not exist", name)
	}
	if _, ok := c.Chains[name]; !ok {
		return fmt.Errorf("chain %q does not exist", name)
	}
	delete(c.Chains, name)
	return c.persistNoLock()
}

// SetTier replaces the tier ordering for a pool in memory and on disk.
func (c *Config) SetTier(pool string, tierList []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.Pools[pool]; !ok {
		return fmt.Errorf("pool %q does not exist", pool)
	}
	if c.Tiers == nil {
		c.Tiers = map[string][]string{}
	}
	c.Tiers[pool] = tierList
	return c.persistNoLock()
}

// TierSortedEntries returns the entries of pool sorted cheapest-first by tier
// order. The tier convention is positional: the i-th pool entry corresponds to
// the i-th tier name in Tiers[pool] (e.g. Tiers["chat"] = ["cheap","standard"]
// means pool entry 0 is the "cheap" tier, entry 1 the "standard" tier). Sorting
// is stable so entries sharing a tier keep their declared relative order.
//
// Entries whose positional index is >= len(Tiers[pool]) have no assigned tier
// and are placed last, preserving their declared relative order.
//
// If no tiers are configured for pool, the raw pool entries are returned
// unchanged (existing behavior).
func (c *Config) TierSortedEntries(pool string) []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tierSortedEntriesNoLock(pool)
}

func (c *Config) tierSortedEntriesNoLock(pool string) []string {
	entries, ok := c.Pools[pool]
	if !ok {
		return nil
	}
	tierList, ok := c.Tiers[pool]
	if !ok || len(tierList) == 0 {
		return entries
	}
	type tagged struct {
		ref   string
		order int
	}
	named := make([]tagged, len(entries))
	for i, ref := range entries {
		order := i
		if order >= len(tierList) {
			order = len(tierList) // untagged entries sink to the end
		}
		named[i] = tagged{ref: ref, order: order}
	}
	// stable insertion sort by order: keeps declared order within a tier and
	// preserves the cheapest-first layout when pools are already declared that way.
	for i := 1; i < len(named); i++ {
		for j := i; j > 0 && named[j].order < named[j-1].order; j-- {
			named[j], named[j-1] = named[j-1], named[j]
		}
	}
	out := make([]string, len(named))
	for i, t := range named {
		out[i] = t.ref
	}
	return out
}

// SetProvider creates or updates a custom provider. If keys is nil, existing
// keys are preserved (only base_url changes). If keys is non-nil, it replaces
// all keys for this provider. accountID is stored as the provider's AccountID
// and is substituted into base_url at request time via the "{account_id}"
// placeholder; pass "" for providers that don't need substitution. apiMode
// selects the upstream wire format ("", "openai", "anthropic", "gemini").
//
// A provider must have EITHER base_url OR deployment_links. If deployment_links
// are provided (as a non-nil slice), base_url may be empty. If both are set
// or neither is set, an error is returned.
func (c *Config) SetProvider(name, baseURL, accountID string, keys []string, apiMode string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if name == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	// built-in providers can't be edited this way
	if name == "openrouter" || name == "ollama" {
		return fmt.Errorf("cannot edit built-in provider %q via this API", name)
	}
	if c.Providers.Custom == nil {
		c.Providers.Custom = map[string]*Provider{}
	}
	existing, ok := c.Providers.Custom[name]
	if !ok {
		if baseURL == "" {
			return fmt.Errorf("provider %q base_url cannot be empty for new providers (use deployment_links for multi-endpoint mode)", name)
		}
		c.Providers.Custom[name] = &Provider{BaseURL: baseURL, AccountID: accountID, Keys: keys, APIMode: apiMode}
	} else {
		existing.BaseURL = baseURL
		existing.AccountID = accountID
		existing.APIMode = apiMode
		if keys != nil {
			existing.Keys = keys
		}
	}
	c.autodetectProviderStatesLocked()
	return c.persistNoLock()
}

func discoverDeploymentModel(baseURL, key string) (string, error) {
	client := &http.Client{Timeout: 15 * time.Second}
	url := strings.TrimRight(baseURL, "/") + "/models"
	// SSRF guard: reject private/internal addresses before issuing the request.
	// An attacker with provider-CRUD access could otherwise point baseURL at
	// 169.254.169.254 (cloud metadata) or 127.0.0.1:<internal-port>.
	u, err := neturl.Parse(url)
	if err != nil {
		return "", fmt.Errorf("invalid baseURL: %w", err)
	}
	if host := u.Hostname(); host != "" {
		if isPrivateHost(host) {
			return "", fmt.Errorf("baseURL host %q is not allowed (private/internal)", host)
		}
	}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("model discovery returned HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err
	}
	var parsed struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", err
	}
	if len(parsed.Data) == 0 {
		return "", errors.New("deployment reports no models")
	}
	if len(parsed.Data) > 1 {
		return "", fmt.Errorf("deployment reports %d models; a deployment link serves exactly one", len(parsed.Data))
	}
	return parsed.Data[0].ID, nil
}

// SetProviderDeployment adds, updates, or deletes one named deployment under a
// multi-deployment provider (e.g. Modal). Parent provider must already exist.
func (c *Config) SetProviderDeployment(parent, deploy, baseURL, model string, keys []string, action string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if parent == "" || deploy == "" {
		return fmt.Errorf("provider and deployment names cannot be empty")
	}
	if parent == "openrouter" || parent == "ollama" {
		return fmt.Errorf("built-in provider %q does not support deployments", parent)
	}
	if strings.Contains(parent, "/") || strings.Contains(deploy, "/") {
		return fmt.Errorf("deployment names cannot contain '/'")
	}
	pr, ok := c.Providers.Custom[parent]
	if !ok || pr == nil {
		return fmt.Errorf("provider %q does not exist — create it first", parent)
	}
	if pr.Deployments == nil {
		pr.Deployments = map[string]*Deployment{}
	}
	if action == "delete" {
		// refuse if any pool/vision entry references parent/child
		full := parent + "/" + deploy
		for pool, entries := range c.Pools {
			for _, ref := range entries {
				if prov, _, _ := strings.Cut(ref, ":"); prov == full {
					return fmt.Errorf("cannot delete deployment %q: pool %q references it", full, pool)
				}
			}
		}
		for _, ref := range c.Vision {
			if prov, _, _ := strings.Cut(ref, ":"); prov == full {
				return fmt.Errorf("cannot delete deployment %q: vision list references it", full)
			}
		}
		if _, ok := pr.Deployments[deploy]; !ok {
			return fmt.Errorf("deployment %q does not exist on %q", deploy, parent)
		}
		delete(pr.Deployments, deploy)
		return c.persistNoLock()
	}
	if baseURL == "" {
		return fmt.Errorf("deployment %q base_url cannot be empty", deploy)
	}
	if keys == nil {
		keys = []string{}
	}
	if model == "" {
		// auto-discover: query the deployment's /models and use the single
		// reported model ID. The caller passes keys (masked lists may carry
		// real keys on create); try each until one authenticates.
		var lastErr error
		for _, k := range keys {
			if k == "" {
				continue
			}
			m, err := discoverDeploymentModel(baseURL, k)
			if err == nil {
				model = m
				break
			}
			lastErr = err
		}
		if model == "" {
			if lastErr != nil {
				return fmt.Errorf("deployment %q: model discovery failed: %v (pass model explicitly to override)", deploy, lastErr)
			}
			return fmt.Errorf("deployment %q: no usable key for model discovery — pass key and model explicitly", deploy)
		}
	}
	pr.Deployments[deploy] = &Deployment{BaseURL: baseURL, Keys: keys, Model: model}
	return c.persistNoLock()
}

// SetProviderDeploymentLinks replaces the deployment links on an existing
// provider. It clears base_url (the two are mutually exclusive) and updates
// the provider to multi-endpoint mode. Pass an empty slice to clear links
// and revert to single-endpoint mode (base_url must then be set separately).
func (c *Config) SetProviderDeploymentLinks(name string, links []DeploymentLink) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if name == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	if name == "openrouter" || name == "ollama" {
		return fmt.Errorf("cannot edit built-in provider %q via this API", name)
	}
	p, ok := c.Providers.Get(name)
	if !ok {
		return fmt.Errorf("provider %q does not exist", name)
	}
	p.DeploymentLinks = links
	p.BaseURL = "" // mutual exclusivity
	p.Keys = nil   // keys live on the links now
	c.autodetectProviderStatesLocked()
	return c.persistNoLock()
}

// SetProviderKeys replaces the key list on an existing provider, and the
// per-key labels alongside it. Built-in providers (openrouter, ollama) are
// allowed — they need key management too. ollama simply ignores keys (localhost).
//
// labels is index-aligned with keys and may be nil or short; missing entries
// become empty, which the dashboard renders as "Key <n>". Labels are resolved in
// lockstep with the values so a label always travels with the key it names, even
// when __KEEP__ sentinels shift positions.
func (c *Config) SetProviderKeys(name string, keys []string, labels []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if name == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	p, ok := c.Providers.Get(name)
	if !ok {
		return fmt.Errorf("provider %q does not exist", name)
	}
	labelAt := func(i int) string {
		if i < len(labels) {
			return strings.TrimSpace(labels[i])
		}
		return ""
	}
	// A keep-sentinel means "preserve an existing real key". The dashboard sends
	// masked keys this way so it can submit the full desired list without ever
	// seeing or round-tripping the real secret.
	//
	// "__KEEP__:<n>" names the key by its original index, which is the only form
	// that survives a removal: a bare "__KEEP__" is consumed in order, so
	// deleting any key but the last one kept the wrong secret and silently
	// discarded another. The bare form is still honored for older clients.
	resolved := make([]string, 0, len(keys))
	resolvedLabels := make([]string, 0, len(keys))
	keepIdx := 0
	for i, k := range keys {
		if strings.HasPrefix(k, keepSentinel) {
			if rest := strings.TrimPrefix(k, keepSentinel); strings.HasPrefix(rest, ":") {
				n, err := strconv.Atoi(strings.TrimSpace(rest[1:]))
				if err == nil && n >= 0 && n < len(p.Keys) && p.Keys[n] != "" {
					resolved = append(resolved, p.Keys[n])
					resolvedLabels = append(resolvedLabels, labelAt(i))
				}
				continue // an out-of-range reference drops the entry rather than guessing
			}
			// bare sentinel: consume the next existing key in order
			for keepIdx < len(p.Keys) && p.Keys[keepIdx] == "" {
				keepIdx++
			}
			if keepIdx < len(p.Keys) {
				resolved = append(resolved, p.Keys[keepIdx])
				resolvedLabels = append(resolvedLabels, labelAt(i))
				keepIdx++
			}
			continue
		}
		resolved = append(resolved, k)
		resolvedLabels = append(resolvedLabels, labelAt(i))
	}
	p.Keys = resolved
	// drop a trailing run of empty labels so an all-unlabeled provider doesn't
	// grow a pointless ["", "", ""] in router.yaml
	for len(resolvedLabels) > 0 && resolvedLabels[len(resolvedLabels)-1] == "" {
		resolvedLabels = resolvedLabels[:len(resolvedLabels)-1]
	}
	p.KeyLabels = resolvedLabels
	c.autodetectProviderStatesLocked()
	return c.persistNoLock()
}

// KeyLabelAt returns the label for the i-th key, or "" when unlabeled.
func (p *Provider) KeyLabelAt(i int) string {
	if p == nil || i < 0 || i >= len(p.KeyLabels) {
		return ""
	}
	return p.KeyLabels[i]
}

// DeleteProvider removes a custom provider. Returns an error if any pool or
// vision entry still references it — the caller must clean those up first.
func (c *Config) DeleteProvider(name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if name == "openrouter" || name == "ollama" {
		return fmt.Errorf("cannot delete built-in provider %q", name)
	}
	if c.Providers.Custom == nil {
		return fmt.Errorf("provider %q does not exist", name)
	}
	if _, ok := c.Providers.Custom[name]; !ok {
		return fmt.Errorf("provider %q does not exist", name)
	}
	// check no pools reference it
	for pool, entries := range c.Pools {
		for _, ref := range entries {
			prov, _, _ := strings.Cut(ref, ":")
			if prov == name {
				return fmt.Errorf("cannot delete provider %q: pool %q references it", name, pool)
			}
		}
	}
	// check vision doesn't reference it
	for _, ref := range c.Vision {
		prov, _, _ := strings.Cut(ref, ":")
		if prov == name {
			return fmt.Errorf("cannot delete provider %q: vision list references it", name)
		}
	}
	delete(c.Providers.Custom, name)
	return c.persistNoLock()
}

func (c *Config) autodetectProviderStatesLocked() {
	// same as autodetectProviderStates but assumes lock held
	if c.Providers.OpenRouter != nil && c.Providers.OpenRouter.Enabled == nil {
		if len(c.Providers.OpenRouter.Keys) > 0 {
			enabled := true
			c.Providers.OpenRouter.Enabled = &enabled
		} else {
			enabled := false
			c.Providers.OpenRouter.Enabled = &enabled
		}
	}
	if c.Providers.Ollama != nil && c.Providers.Ollama.Enabled == nil {
		enabled := true
		c.Providers.Ollama.Enabled = &enabled
	}
	for _, p := range c.Providers.Custom {
		if p == nil || p.Enabled != nil {
			continue
		}
		hasKeys := len(p.Keys) > 0 || len(p.DeploymentLinks) > 0
		isLocal := strings.Contains(p.BaseURL, "127.0.0.1") || strings.Contains(p.BaseURL, "localhost")
		if hasKeys || isLocal {
			enabled := true
			p.Enabled = &enabled
		} else {
			enabled := false
			p.Enabled = &enabled
		}
	}
}

// ----- getters (thread-safe) -----

// GetDefault returns the current default pool.
func (c *Config) GetDefault() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.Default
}

// GetPools returns a shallow copy of the pools map.
func (c *Config) GetPools() map[string][]string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make(map[string][]string, len(c.Pools))
	for k, v := range c.Pools {
		out[k] = append([]string(nil), v...)
	}
	return out
}

// GetPoolContext returns a copy of the per-pool advertised context overrides.
// Keys are pool names plus the special "router"/"auto" aliases. A missing key
// means "compute from the pool's models" (see Server.poolContext).
func (c *Config) GetPoolContext() map[string]int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make(map[string]int, len(c.PoolContext))
	for k, v := range c.PoolContext {
		out[k] = v
	}
	return out
}

// GetClassifierHeuristics returns a shallow copy of the heuristics map.
func (c *Config) GetClassifierHeuristics() map[string][]string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make(map[string][]string, len(c.Classifier.Heuristics))
	for k, v := range c.Classifier.Heuristics {
		out[k] = append([]string(nil), v...)
	}
	return out
}

// GetVision returns a copy of the vision slice.
func (c *Config) GetVision() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return append([]string(nil), c.Vision...)
}

// GetMediaPool returns the name of the pool that serves requests carrying
// media (images, audio, or video), or "" when none is configured. A pool named
// "media" wins; a legacy "vision" pool is honored second so configs written
// before audio and video were routable keep working. When this returns "",
// callers should fall back to the default pool — the pre-media behavior.
func (c *Config) GetMediaPool() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, name := range []string{"media", "vision"} {
		if entries, ok := c.Pools[name]; ok && len(entries) > 0 {
			return name
		}
	}
	return ""
}

// GetAllowDirectVision returns the allow_direct_vision flag.
func (c *Config) GetAllowDirectVision() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.AllowDirectVision
}

// GetProvider returns a provider by name (thread-safe).
func (c *Config) GetProvider(name string) (*Provider, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.Providers.Get(name)
}

// GetChain returns a named chain (thread-safe).
func (c *Config) GetChain(name string) ([]string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	ch, ok := c.Chains[name]
	return ch, ok
}

// ProviderRouting is the routing-relevant snapshot of one provider/model pair.
// It exists so the hot path can take everything it needs in a single read lock
// (see GetProviderRouting) instead of a value per call.
type ProviderRouting struct {
	OK                     bool // false = no such provider
	BaseURL                string
	Keys                   []string
	Enabled                bool
	ModelDisabled          bool
	ModelLimit             int
	HasLimit               bool
	AccountID              string
	APIMode                string
	StripParams            []string
	RepairReasoningContent bool
	MediaPolicy            MediaPolicy
	// UpstreamModel is the real upstream model id when the pool-facing model
	// slot names a deployment ("Modal:glm53" -> "glm-5.3-flash"). Empty when
	// the provider has no deployment-based model mapping.
	UpstreamModel string
}

// GetProviderDeploymentLink returns the deployment link that serves the given
// model for a provider. If the provider has no deployment links, or the model
// has not been discovered yet, it returns false.
func (p *Provider) GetProviderDeploymentLink(model string) (DeploymentLink, bool) {
	if p == nil || !p.HasDeploymentLinks() {
		return DeploymentLink{}, false
	}
	if p.modelLinks != nil {
		dl, ok := p.modelLinks[model]
		return dl, ok
	}
	return DeploymentLink{}, false
}

// SetModelLink records that the given model is served by the given link.
// Called by the server after fetching /v1/models from a deployment link.
func (p *Provider) SetModelLink(model string, dl DeploymentLink) {
	if p == nil {
		return
	}
	if p.modelLinks == nil {
		p.modelLinks = map[string]DeploymentLink{}
	}
	p.modelLinks[model] = dl
}

// GetProviderRouting returns the routing-relevant fields of a provider under a
// single read lock, so the hot path never reads provider fields while an admin
// write (key rotation, toggle, model limits, media policy) is in flight. The
// returned Keys slice is used to (re)build the per-provider KeyPicker.
//
// For providers with deployment links, the BaseURL and Keys are resolved from
// the link that serves the requested model (discovered at fetch time). If the
// model hasn't been discovered yet, the first link's URL and key are returned
// as a fallback so the request can still proceed (the breaker key will differ
// per actual URL at trip time).
func (c *Config) GetProviderRouting(name, model string) ProviderRouting {
	c.mu.RLock()
	defer c.mu.RUnlock()
	p, ok := c.Providers.Get(name)
	if !ok || p == nil {
		return ProviderRouting{}
	}

	// For deployment-links providers, resolve the specific link for this model.
	if p.HasDeploymentLinks() {
		dl, found := p.GetProviderDeploymentLink(model)
		if !found {
			// Model not yet discovered — fall back to first link so the
			// request can still go through. The breaker key computed later
			// in route.go uses the actual base URL from this struct.
			dl = p.DeploymentLinks[0]
		}
		pr := ProviderRouting{
			OK:                     true,
			BaseURL:                dl.URL,
			Keys:                   []string{dl.Key},
			Enabled:                p.IsEnabled(),
			ModelDisabled:          p.IsModelDisabled(model),
			AccountID:              p.AccountID,
			APIMode:                p.APIMode,
			StripParams:            p.StripParams,
			RepairReasoningContent: p.RepairReasoningContent,
			MediaPolicy:            p.MediaPolicyFor(model),
		}
		if lim, has := p.ModelLimits[model]; has {
			pr.ModelLimit = lim
			pr.HasLimit = true
		}
		return pr
	}

	pr := ProviderRouting{
		OK:                     true,
		BaseURL:                p.BaseURL,
		Keys:                   p.Keys,
		Enabled:                p.IsEnabled(),
		ModelDisabled:          p.IsModelDisabled(model),
		AccountID:              p.AccountID,
		APIMode:                p.APIMode,
		StripParams:            p.StripParams,
		RepairReasoningContent: p.RepairReasoningContent,
		MediaPolicy:            p.MediaPolicyFor(model),
	}
	if lim, has := p.ModelLimits[model]; has {
		pr.ModelLimit = lim
		pr.HasLimit = true
	}
	return pr
}

// ReloadFile re-reads and re-validates the config from disk, replacing the
// in-memory field values (but keeping Path and the mutex). Used by SIGHUP.
func (c *Config) ReloadFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return c.ReloadFrom(f)
}

// ReloadFrom loads a fresh config from r and copies its field values into the
// receiver. The mutex and Path are preserved (never overwritten) so concurrent
// readers keep working and we don't copy a live lock.
func (c *Config) ReloadFrom(r io.Reader) error {
	fresh, err := Load(r)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.DBPath = fresh.DBPath
	c.Listen = fresh.Listen
	c.RouterKey = fresh.RouterKey
	c.InsecureNoAuth = fresh.InsecureNoAuth
	c.CatalogURL = fresh.CatalogURL
	c.Default = fresh.Default
	c.Pools = fresh.Pools
	c.Chains = fresh.Chains
	c.Tiers = fresh.Tiers
	c.PoolContext = fresh.PoolContext
	c.AllowDirectVision = fresh.AllowDirectVision
	c.Vision = fresh.Vision
	c.Providers = fresh.Providers
	c.Classifier = fresh.Classifier
	c.Fallback = fresh.Fallback
	return nil
}

// GetFallback returns a copy of the fallback config.
func (c *Config) GetFallback() FallbackCfg {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.Fallback
}

// SetFallback persists failover knobs. Zero values keep the current field.
func (c *Config) SetFallback(in FallbackCfg) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	cur := c.Fallback
	if in.TimeoutS > 0 {
		cur.TimeoutS = in.TimeoutS
	}
	if in.Strategy != "" {
		cur.Strategy = in.Strategy
	}
	if in.KeyCooldownS > 0 {
		cur.KeyCooldownS = in.KeyCooldownS
	}
	if in.ProviderCooldownS > 0 {
		cur.ProviderCooldownS = in.ProviderCooldownS
	}
	if in.ProviderFailureThreshold > 0 {
		cur.ProviderFailureThreshold = in.ProviderFailureThreshold
	}
	if in.ProviderLockoutS > 0 {
		cur.ProviderLockoutS = in.ProviderLockoutS
	}
	if in.RetryTransientMax > 0 {
		cur.RetryTransientMax = in.RetryTransientMax
	}
	c.Fallback = cur
	return c.persistNoLock()
}

// GetTierSortedEntries returns tier-sorted entries for a pool.
func (c *Config) GetTierSortedEntries(pool string) []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tierSortedEntriesNoLock(pool)
}
