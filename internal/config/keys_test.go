package config

import "testing"

func keyProvider() *Config {
	return &Config{
		Pools: map[string][]string{"chat": {"p:m"}},
		Providers: Providers{Custom: map[string]*Provider{
			"p": {BaseURL: "https://example.com/v1", Keys: []string{"aaaa1111", "bbbb2222", "cccc3333"}},
		}},
	}
}

func TestKeyLabelsPersistAndAlign(t *testing.T) {
	cfg := keyProvider()
	err := cfg.SetProviderKeys("p",
		[]string{"__KEEP__:0", "__KEEP__:1", "__KEEP__:2"},
		[]string{"prod", "burner", "spare"})
	if err != nil {
		t.Fatalf("set: %v", err)
	}
	p := cfg.Providers.Custom["p"]
	if got := p.KeyLabels; len(got) != 3 || got[0] != "prod" || got[2] != "spare" {
		t.Fatalf("labels = %v", got)
	}
	if got := p.KeyLabelAt(1); got != "burner" {
		t.Fatalf("KeyLabelAt(1) = %q", got)
	}
	// out of range is empty, never a panic
	if got := p.KeyLabelAt(99); got != "" {
		t.Fatalf("KeyLabelAt(99) = %q, want empty", got)
	}
}

// The indexed sentinel is the whole point: removing any key but the last used to
// keep the wrong secret, because a bare __KEEP__ is consumed in order.
func TestIndexedKeepRemovesTheRightKey(t *testing.T) {
	cfg := keyProvider()
	if err := cfg.SetProviderKeys("p",
		[]string{"__KEEP__:1", "__KEEP__:2"},
		[]string{"burner", "spare"}); err != nil {
		t.Fatalf("set: %v", err)
	}
	p := cfg.Providers.Custom["p"]
	if len(p.Keys) != 2 || p.Keys[0] != "bbbb2222" || p.Keys[1] != "cccc3333" {
		t.Fatalf("keys = %v, want the 2nd and 3rd secrets (1st removed)", p.Keys)
	}
	if p.KeyLabels[0] != "burner" || p.KeyLabels[1] != "spare" {
		t.Fatalf("labels = %v, must travel with their keys", p.KeyLabels)
	}
}

// The bare sentinel still works for older clients, consuming in order.
func TestBareKeepStillConsumesInOrder(t *testing.T) {
	cfg := keyProvider()
	if err := cfg.SetProviderKeys("p", []string{"__KEEP__", "__KEEP__"}, nil); err != nil {
		t.Fatalf("set: %v", err)
	}
	p := cfg.Providers.Custom["p"]
	if len(p.Keys) != 2 || p.Keys[0] != "aaaa1111" || p.Keys[1] != "bbbb2222" {
		t.Fatalf("keys = %v, want the first two in order", p.Keys)
	}
}

func TestOutOfRangeKeepIsDropped(t *testing.T) {
	cfg := keyProvider()
	if err := cfg.SetProviderKeys("p",
		[]string{"__KEEP__:0", "__KEEP__:99", "__KEEP__:-1"},
		[]string{"prod", "ghost", "ghost2"}); err != nil {
		t.Fatalf("set: %v", err)
	}
	p := cfg.Providers.Custom["p"]
	if len(p.Keys) != 1 || p.Keys[0] != "aaaa1111" {
		t.Fatalf("keys = %v, want only the valid reference", p.Keys)
	}
	if len(p.KeyLabels) != 1 || p.KeyLabels[0] != "prod" {
		t.Fatalf("labels = %v", p.KeyLabels)
	}
}

func TestNewKeysMixWithKeptOnes(t *testing.T) {
	cfg := keyProvider()
	if err := cfg.SetProviderKeys("p",
		[]string{"__KEEP__:2", "dddd4444"},
		[]string{"spare", "fresh"}); err != nil {
		t.Fatalf("set: %v", err)
	}
	p := cfg.Providers.Custom["p"]
	if len(p.Keys) != 2 || p.Keys[0] != "cccc3333" || p.Keys[1] != "dddd4444" {
		t.Fatalf("keys = %v", p.Keys)
	}
	if p.KeyLabels[0] != "spare" || p.KeyLabels[1] != "fresh" {
		t.Fatalf("labels = %v", p.KeyLabels)
	}
}

// An entirely unlabeled provider must not grow ["", "", ""] in router.yaml.
func TestTrailingEmptyLabelsPruned(t *testing.T) {
	cfg := keyProvider()
	if err := cfg.SetProviderKeys("p",
		[]string{"__KEEP__:0", "__KEEP__:1"},
		[]string{"", ""}); err != nil {
		t.Fatalf("set: %v", err)
	}
	if got := cfg.Providers.Custom["p"].KeyLabels; len(got) != 0 {
		t.Fatalf("labels = %v, want none stored", got)
	}
	// but a label followed by a blank keeps its position
	if err := cfg.SetProviderKeys("p",
		[]string{"__KEEP__:0", "__KEEP__:1"},
		[]string{"named", ""}); err != nil {
		t.Fatalf("set: %v", err)
	}
	if got := cfg.Providers.Custom["p"].KeyLabels; len(got) != 1 || got[0] != "named" {
		t.Fatalf("labels = %v, want [named]", got)
	}
}

// Labels are exposed by Redacted so the dashboard can render what was saved —
// key values stay masked.
func TestRedactedExposesLabelsNotSecrets(t *testing.T) {
	cfg := keyProvider()
	if err := cfg.SetProviderKeys("p",
		[]string{"__KEEP__:0"}, []string{"prod"}); err != nil {
		t.Fatalf("set: %v", err)
	}
	out := cfg.Redacted()
	p := out["providers"].(map[string]any)["p"].(map[string]any)
	labels, _ := p["key_labels"].([]string)
	if len(labels) != 1 || labels[0] != "prod" {
		t.Fatalf("key_labels = %v", p["key_labels"])
	}
	for _, k := range p["keys"].([]string) {
		if k == "aaaa1111" {
			t.Fatal("Redacted must not return a real key")
		}
	}
}
