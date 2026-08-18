package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func persistFixture(t *testing.T) *Config {
	t.Helper()
	c := DefaultConfig()
	c.Path = filepath.Join(t.TempDir(), "router.yaml")
	c.Providers.Custom = map[string]*Provider{
		"acme": {BaseURL: "https://acme.example/v1", Keys: []string{"sk-first"}},
	}
	return c
}

func TestPersistWritesConfigWithTightPermissions(t *testing.T) {
	c := persistFixture(t)
	if err := c.persistNoLock(); err != nil {
		t.Fatalf("persist: %v", err)
	}
	info, err := os.Stat(c.Path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	// The file holds every provider API key; it must not be group/world readable.
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("permissions = %o, want 600", perm)
	}
	body, _ := os.ReadFile(c.Path)
	if !strings.Contains(string(body), "sk-first") {
		t.Fatalf("key missing from persisted config:\n%s", body)
	}
}

// A save keeps the previous contents alongside, so a bad write is recoverable
// rather than silently destroying credentials that exist nowhere else.
func TestPersistKeepsPreviousVersionAsBackup(t *testing.T) {
	c := persistFixture(t)
	if err := c.persistNoLock(); err != nil {
		t.Fatalf("first persist: %v", err)
	}
	c.Providers.Custom["acme"].Keys = []string{"sk-second"}
	if err := c.persistNoLock(); err != nil {
		t.Fatalf("second persist: %v", err)
	}

	current, _ := os.ReadFile(c.Path)
	if !strings.Contains(string(current), "sk-second") {
		t.Fatalf("current config should hold the new key:\n%s", current)
	}
	backup, err := os.ReadFile(c.Path + ".bak")
	if err != nil {
		t.Fatalf("no backup written: %v", err)
	}
	if !strings.Contains(string(backup), "sk-first") {
		t.Fatalf("backup should hold the superseded key:\n%s", backup)
	}
	if info, err := os.Stat(c.Path + ".bak"); err == nil {
		if perm := info.Mode().Perm(); perm != 0o600 {
			t.Fatalf("backup permissions = %o, want 600", perm)
		}
	}
}

// The write goes via a temp file and a rename; none of that scratch may survive.
func TestPersistLeavesNoTempFiles(t *testing.T) {
	c := persistFixture(t)
	for i := 0; i < 3; i++ {
		if err := c.persistNoLock(); err != nil {
			t.Fatalf("persist %d: %v", i, err)
		}
	}
	entries, err := os.ReadDir(filepath.Dir(c.Path))
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".router-config-") {
			t.Fatalf("temp file left behind: %s", e.Name())
		}
	}
}

// The persisted file must load back into an equivalent config — the round trip
// is what makes a dashboard save safe.
func TestPersistRoundTrips(t *testing.T) {
	c := persistFixture(t)
	if err := c.persistNoLock(); err != nil {
		t.Fatalf("persist: %v", err)
	}
	back, err := LoadFile(c.Path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	p, ok := back.Providers.Get("acme")
	if !ok {
		t.Fatalf("custom provider lost on round trip; providers = %+v", back.Providers)
	}
	if len(p.Keys) != 1 || p.Keys[0] != "sk-first" {
		t.Fatalf("keys = %v, want [sk-first]", p.Keys)
	}
}
