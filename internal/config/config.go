// Package config loads and validates the router YAML config.
package config

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

const defaultCatalogURL = "https://models.dev/api.json"

// Provider is one upstream endpoint with a pool of API keys.
// Enabled controls whether the router will route to this provider. When
// false, the provider is skipped as if it's not in any pool. This is a
// soft toggle — the provider's config (keys, base_url) is preserved.
// ModelLimits optionally caps max_tokens per model id, applied as an
// upper bound before the request is sent upstream.
type Provider struct {
	BaseURL string           `yaml:"base_url"`
	Keys    []string         `yaml:"keys"`
	Enabled *bool            `yaml:"enabled"` // pointer so "nil" = enabled (default), false = disabled
	ModelLimits map[string]int `yaml:"model_limits"`
	// AccountID is an opaque per-provider identifier (e.g. a Cloudflare
	// account id) substituted into base_url at request time via the
	// "{account_id}" placeholder. Empty means no substitution.
	AccountID string          `yaml:"account_id"`
	// DisabledModels is a list of model ids that should be skipped during
	// candidate selection even when the provider is enabled. This allows
	// per-model disable without disabling the whole provider.
	DisabledModels []string  `yaml:"disabled_models"`
	// Preset marks a provider as a built-in/preset entry in the dashboard
	// (shown in the "Preset Providers" group). Custom providers added via the
	// API default to false and appear under "Custom Providers".
	Preset bool               `yaml:"preset"`
	// APIMode selects the upstream wire format: "openai" (default),
	// "anthropic", or "gemini". Non-OpenAI modes translate the request and
	// response at the provider boundary so the router stays OpenAI-shaped.
	APIMode string             `yaml:"api_mode"`
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
}

// Config is the whole router configuration.
type Config struct {
	mu                sync.RWMutex
	Path              string             `yaml:"-"`
	DBPath            string             `yaml:"db"`
	Listen            string             `yaml:"listen"`
	RouterKey         string             `yaml:"router_key"`
	InsecureNoAuth    bool               `yaml:"insecure_no_auth"`
	CatalogURL        string             `yaml:"catalog_url"`
	Default           string             `yaml:"default"`
	Pools             map[string][]string `yaml:"pools"`
	Chains            map[string][]string `yaml:"chains"`
	Tiers             map[string][]string `yaml:"tiers"`
	AllowDirectVision bool               `yaml:"allow_direct_vision"`
	Vision            []string           `yaml:"vision"`
	Providers         Providers          `yaml:"providers"`
	Classifier        ClassifierCfg      `yaml:"classifier"`
	Fallback          FallbackCfg        `yaml:"fallback"`
}

// DefaultConfig returns a sane starting config.
func DefaultConfig() *Config {
	return &Config{
		Listen:     "127.0.0.1:8011",
		DBPath:     "router.db",
		CatalogURL: defaultCatalogURL,
		Default:    "chat",
		Pools: map[string][]string{
			"chat":      {},
			"code":      {},
			"creative":  {},
			"reasoning": {},
		},
		Chains:            map[string][]string{},
		Tiers:             map[string][]string{},
		AllowDirectVision: true,
		Providers: Providers{Custom: map[string]*Provider{}},
		Classifier: ClassifierCfg{
			HintsFirst: true,
			Heuristics: map[string][]string{
				"code":      {"def ", "import ", "func ", "select ", "gradle", "npm ", "git ", "docker", "python", "class ", "const ", "function "},
				"reasoning": {"why ", "explain the tradeoffs", "prove", "plan the"},
			},
		},
		Fallback: FallbackCfg{TimeoutS: 90, Strategy: "round_robin", KeyCooldownS: 60},
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
	provs := map[string]any{}
	if c.Providers.OpenRouter != nil {
		provs["openrouter"] = map[string]any{"base_url": c.Providers.OpenRouter.BaseURL, "keys": maskList(c.Providers.OpenRouter.Keys, mask), "enabled": c.Providers.OpenRouter.IsEnabled(), "disabled_models": c.Providers.OpenRouter.DisabledModels, "preset": true}
	}
	if c.Providers.Ollama != nil {
		provs["ollama"] = map[string]any{"base_url": c.Providers.Ollama.BaseURL, "keys": maskList(c.Providers.Ollama.Keys, mask), "enabled": c.Providers.Ollama.IsEnabled(), "disabled_models": c.Providers.Ollama.DisabledModels, "preset": true}
	}
	for name, p := range c.Providers.Custom {
		provs[name] = map[string]any{"base_url": p.BaseURL, "account_id": p.AccountID, "keys": maskList(p.Keys, mask), "enabled": p.IsEnabled(), "disabled_models": p.DisabledModels, "preset": p.Preset, "api_mode": p.APIMode}
	}
	heuristics := map[string][]string{}
	for pool, kws := range c.Classifier.Heuristics {
		heuristics[pool] = kws
	}
	return map[string]any{
		"default":      c.Default,
		"pools":        c.Pools,
		"chains":       c.Chains,
		"tiers":        c.Tiers,
		"vision":       c.Vision,
		"providers":    provs,
		"classifier":   map[string]any{"heuristics": heuristics},
		"fallback":     map[string]any{"timeout_s": c.Fallback.TimeoutS, "strategy": c.Fallback.Strategy, "key_cooldown_s": c.Fallback.KeyCooldownS},
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

// SetProviderModelSettings updates model_limits and disabled_models for a provider.
func (c *Config) SetProviderModelSettings(name string, modelLimits map[string]int, disabledModels []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	p, ok := c.Providers.Get(name)
	if !ok {
		return fmt.Errorf("provider %q not found", name)
	}
	p.ModelLimits = modelLimits
	p.DisabledModels = disabledModels
	return c.persistNoLock()
}

func (c *Config) persistNoLock() error {
	if c.Path == "" {
		return nil
	}
	out, err := yaml.Marshal(c)
	if err != nil {
		return err
	}
	return os.WriteFile(c.Path, out, 0o600)
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
func (c *Config) SetProvider(name, baseURL, accountID string, keys []string, apiMode string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if name == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	if baseURL == "" {
		return fmt.Errorf("provider %q base_url cannot be empty", name)
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

// SetProviderKeys replaces the key list on an existing provider.
// Built-in providers (openrouter, ollama) are allowed — they need key
// management too. ollama simply ignores keys (localhost).
func (c *Config) SetProviderKeys(name string, keys []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if name == "" {
		return fmt.Errorf("provider name cannot be empty")
	}
	p, ok := c.Providers.Get(name)
	if !ok {
		return fmt.Errorf("provider %q does not exist", name)
	}
	// A "__KEEP__" sentinel means "preserve the existing real key at this
	// position". The dashboard sends masked keys as __KEEP__ so it can
	// submit the full desired list without ever seeing/round-tripping the
	// real secret.
	resolved := make([]string, 0, len(keys))
	keepIdx := 0
	for _, k := range keys {
		if k == "__KEEP__" {
			// pull the next existing real key
			for keepIdx < len(p.Keys) && p.Keys[keepIdx] == "" {
				keepIdx++
			}
			if keepIdx < len(p.Keys) {
				resolved = append(resolved, p.Keys[keepIdx])
				keepIdx++
			}
			continue
		}
		resolved = append(resolved, k)
	}
	p.Keys = resolved
	c.autodetectProviderStatesLocked()
	return c.persistNoLock()
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
		hasKeys := len(p.Keys) > 0
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

// GetProviderRouting returns the routing-relevant fields of a provider under a
// single read lock, so the hot path never reads provider fields while an admin
// write (key rotation, toggle, model limits) is in flight. The returned keys
// slice is used to (re)build the per-provider KeyPicker.
func (c *Config) GetProviderRouting(name, model string) (baseURL string, keys []string, enabled, modelDisabled bool, modelLimit int, hasLimit, ok bool, accountID string, apiMode string) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	p, ok := c.Providers.Get(name)
	if !ok || p == nil {
		return "", nil, false, false, 0, false, false, "", ""
	}
	baseURL = p.BaseURL
	keys = p.Keys
	enabled = p.IsEnabled()
	modelDisabled = p.IsModelDisabled(model)
	accountID = p.AccountID
	apiMode = p.APIMode
	if lim, has := p.ModelLimits[model]; has {
		modelLimit = lim
		hasLimit = true
	}
	return
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

// GetTierSortedEntries returns tier-sorted entries for a pool.
func (c *Config) GetTierSortedEntries(pool string) []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tierSortedEntriesNoLock(pool)
}