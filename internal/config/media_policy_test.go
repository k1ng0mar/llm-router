package config

import (
	"strings"
	"testing"
)

func TestMediaPolicyStanceNormalizes(t *testing.T) {
	pol := MediaPolicy{Image: PolicyAllow, Audio: "", Video: "nonsense"}
	if got := pol.Stance("image"); got != PolicyAllow {
		t.Fatalf("image stance = %q, want allow", got)
	}
	// unset and unrecognized both collapse to auto, so callers only switch on
	// the three real cases
	if got := pol.Stance("audio"); got != PolicyAuto {
		t.Fatalf("unset audio stance = %q, want auto", got)
	}
	if got := pol.Stance("video"); got != PolicyAuto {
		t.Fatalf("unrecognized video stance = %q, want auto", got)
	}
	if got := pol.Stance("smell"); got != PolicyAuto {
		t.Fatalf("unknown modality stance = %q, want auto", got)
	}
}

func TestMediaPolicyIsZero(t *testing.T) {
	if !(MediaPolicy{}).IsZero() {
		t.Fatal("empty policy must be zero")
	}
	if !(MediaPolicy{Image: PolicyAuto, Audio: PolicyAuto, Video: PolicyAuto}).IsZero() {
		t.Fatal("all-auto policy must be zero — it expresses no opinion")
	}
	if (MediaPolicy{Audio: PolicyDeny}).IsZero() {
		t.Fatal("a deny is an opinion")
	}
}

func TestMediaPolicyValidate(t *testing.T) {
	if err := (MediaPolicy{Image: PolicyAllow, Audio: PolicyDeny, Video: PolicyAuto}).Validate(); err != nil {
		t.Fatalf("valid policy rejected: %v", err)
	}
	if err := (MediaPolicy{}).Validate(); err != nil {
		t.Fatalf("empty policy rejected: %v", err)
	}
	err := (MediaPolicy{Video: "maybe"}).Validate()
	if err == nil {
		t.Fatal("expected an error for an unrecognized value")
	}
	if !strings.Contains(err.Error(), "video") {
		t.Fatalf("error should name the offending modality: %v", err)
	}
}

// A bad policy value in router.yaml must fail the load rather than silently
// behaving as "auto" on every request.
func TestLoadRejectsBadMediaPolicy(t *testing.T) {
	yaml := `
default: chat
pools:
  chat:
  - p:m
providers:
  custom:
    p:
      base_url: https://example.com/v1
      keys: [k]
      media_policies:
        m:
          image: sometimes
`
	if _, err := Load(strings.NewReader(yaml)); err == nil {
		t.Fatal("expected load to fail on an invalid media policy value")
	} else if !strings.Contains(err.Error(), "media policy") {
		t.Fatalf("error should explain the media policy problem: %v", err)
	}
}

func TestLoadAcceptsMediaPolicies(t *testing.T) {
	yaml := `
default: chat
pools:
  chat:
  - p:m
  media:
  - p:m
providers:
  custom:
    p:
      base_url: https://example.com/v1
      keys: [k]
      media_policies:
        m:
          image: allow
          audio: deny
`
	cfg, err := Load(strings.NewReader(yaml))
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	pol := cfg.Providers.Custom["p"].MediaPolicyFor("m")
	if pol.Stance("image") != PolicyAllow || pol.Stance("audio") != PolicyDeny || pol.Stance("video") != PolicyAuto {
		t.Fatalf("policy = %+v", pol)
	}
	// a model with no entry defers entirely to the catalog
	if !cfg.Providers.Custom["p"].MediaPolicyFor("other").IsZero() {
		t.Fatal("an unlisted model must yield the zero policy")
	}
	if got := cfg.GetMediaPool(); got != "media" {
		t.Fatalf("GetMediaPool() = %q, want media", got)
	}
}

func TestGetMediaPoolPrecedence(t *testing.T) {
	cases := []struct {
		name  string
		pools map[string][]string
		want  string
	}{
		{"media wins", map[string][]string{"media": {"p:m"}, "vision": {"p:v"}}, "media"},
		{"legacy vision honored", map[string][]string{"vision": {"p:v"}}, "vision"},
		{"empty media pool is not a media pool", map[string][]string{"media": {}}, ""},
		{"neither configured", map[string][]string{"chat": {"p:m"}}, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := &Config{Pools: c.pools}
			if got := cfg.GetMediaPool(); got != c.want {
				t.Fatalf("GetMediaPool() = %q, want %q", got, c.want)
			}
		})
	}
}

// SetProviderModelSettings must prune no-opinion policies so router.yaml doesn't
// grow an all-auto entry for every model the dashboard has ever rendered.
func TestSetProviderModelSettingsPrunesAndValidates(t *testing.T) {
	cfg := &Config{
		Pools: map[string][]string{"chat": {"p:m"}},
		Providers: Providers{Custom: map[string]*Provider{
			"p": {BaseURL: "https://example.com/v1", Keys: []string{"k"}},
		}},
	}
	err := cfg.SetProviderModelSettings("p", map[string]int{"m": 100}, nil, map[string]MediaPolicy{
		"m":     {Image: PolicyDeny},
		"other": {Image: PolicyAuto, Audio: PolicyAuto, Video: PolicyAuto},
	})
	if err != nil {
		t.Fatalf("set: %v", err)
	}
	got := cfg.Providers.Custom["p"].MediaPolicies
	if len(got) != 1 || got["m"].Image != PolicyDeny {
		t.Fatalf("policies = %+v, want only the m deny entry", got)
	}

	if err := cfg.SetProviderModelSettings("p", nil, nil, map[string]MediaPolicy{"m": {Audio: "loud"}}); err == nil {
		t.Fatal("expected an error for an invalid policy value")
	}
}
