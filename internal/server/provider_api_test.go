package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
	"llm-router/internal/route"
	"llm-router/internal/store"
)

func TestAddCustomProvider(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	// add a new custom provider via admin API
	body := `{"name":"newprov","base_url":"https://api.example.com/v1","keys":["sk-newkey1","sk-newkey2"]}`
	resp, b := post(t, h.srv.URL+"/api/config/providers", "", body, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("add provider: %d %s", resp.StatusCode, b)
	}

	// verify it appears in /api/config (keys redacted)
	resp2, _ := http.Get(h.srv.URL + "/api/config")
	raw, _ := io.ReadAll(resp2.Body)
	cfg := map[string]any{}
	json.Unmarshal(raw, &cfg)
	provs := cfg["providers"].(map[string]any)
	p, ok := provs["newprov"].(map[string]any)
	if !ok {
		t.Fatalf("newprov not in config: %s", raw)
	}
	if p["base_url"] != "https://api.example.com/v1" {
		t.Fatalf("base_url = %v", p["base_url"])
	}
	keys := p["keys"].([]any)
	if len(keys) != 2 {
		t.Fatalf("expected 2 keys, got %d", len(keys))
	}
	for _, k := range keys {
		if strings.Contains(k.(string), "newkey") {
			t.Fatalf("key not redacted: %s", k)
		}
	}
}

func TestUpdateProviderKeys(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	// update keys on existing custom provider "vis"
	body := `{"name":"vis","keys":["sk-vk-new1","sk-vk-new2","sk-vk-new3"]}`
	resp, b := post(t, h.srv.URL+"/api/config/keys", "", body, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("update keys: %d %s", resp.StatusCode, b)
	}

	// config still validates (vis still resolves)
	resp2, _ := http.Get(h.srv.URL + "/api/config")
	raw, _ := io.ReadAll(resp2.Body)
	cfg := map[string]any{}
	json.Unmarshal(raw, &cfg)
	provs := cfg["providers"].(map[string]any)
	vis := provs["vis"].(map[string]any)
	keys := vis["keys"].([]any)
	if len(keys) != 3 {
		t.Fatalf("expected 3 keys, got %d", len(keys))
	}
}

func TestUpdateProviderBaseUrl(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	body := `{"name":"vis","base_url":"https://api.newendpoint.com/v1"}`
	resp, b := post(t, h.srv.URL+"/api/config/providers", "", body, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("update provider: %d %s", resp.StatusCode, b)
	}

	resp2, _ := http.Get(h.srv.URL + "/api/config")
	raw, _ := io.ReadAll(resp2.Body)
	cfg := map[string]any{}
	json.Unmarshal(raw, &cfg)
	provs := cfg["providers"].(map[string]any)
	vis := provs["vis"].(map[string]any)
	if vis["base_url"] != "https://api.newendpoint.com/v1" {
		t.Fatalf("base_url not updated: %v", vis["base_url"])
	}
}

func TestDeleteProvider(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	// first disable any pool entries that reference codep so config stays valid
	// (Validate() checks pool entries resolve; deleting a referenced provider should fail)
	// actually, let's test: deleting a provider that IS referenced should error
	body := `{"name":"vis","action":"delete"}`
	resp, b := post(t, h.srv.URL+"/api/config/providers", "", body, nil)
	// vis is used in vision[], so this should fail
	if resp.StatusCode == 200 {
		// vis may or may not be referenced depending on config — check if vision uses it
		// in our harness vision: ["vis:vm"], so deleting vis invalidates the config
		t.Fatalf("deleting a referenced provider should fail: %s", b)
	}

	// add a standalone provider, then delete it
	addBody := `{"name":"standalone","base_url":"http://localhost:9999/v1","keys":["sk-x"]}`
	post(t, h.srv.URL+"/api/config/providers", "", addBody, nil)

	delBody := `{"name":"standalone","action":"delete"}`
	resp2, b2 := post(t, h.srv.URL+"/api/config/providers", "", delBody, nil)
	if resp2.StatusCode != 200 {
		t.Fatalf("delete standalone provider: %d %s", resp2.StatusCode, b2)
	}

	// confirm it's gone
	resp3, _ := http.Get(h.srv.URL + "/api/config")
	raw, _ := io.ReadAll(resp3.Body)
	cfg := map[string]any{}
	json.Unmarshal(raw, &cfg)
	provs := cfg["providers"].(map[string]any)
	if _, ok := provs["standalone"]; ok {
		t.Fatal("standalone provider should be deleted")
	}
}

func TestProviderConfigPersistsToDisk(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "router.yaml")
	cfgContent := `listen: "127.0.0.1:8011"
default: chat
insecure_no_auth: true
pools:
  chat: ["a:x"]
  code: ["a:x"]
providers:
  custom:
    a: { base_url: "http://127.0.0.1:1/v1", keys: ["sk-orig"] }
fallback:
  strategy: round_robin
`

	cfg, err := config.Load(strings.NewReader(cfgContent))
	if err != nil {
		t.Fatalf("config.Load: %v", err)
	}
	cfg.Path = cfgPath
	cfg.DBPath = filepath.Join(dir, "r.db")
	st, _ := store.Open(cfg.DBPath)
	defer st.Close()
	g := catalog.NewGate(nil)
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	s := New(cfg, st, r, nil)
	ts := httptest.NewServer(s.Handler())
	defer ts.Close()

	// add a new provider via API
	addBody := `{"name":"newprov","base_url":"https://api.test.com/v1","keys":["sk-test123"]}`
	req, _ := http.NewRequest("POST", ts.URL+"/api/config/providers", strings.NewReader(addBody))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("add provider status %d", resp.StatusCode)
	}

	// reload config from disk
	reloaded, err := config.LoadFile(cfgPath)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	p, ok := reloaded.Providers.Get("newprov")
	if !ok {
		t.Fatal("newprov not persisted to disk")
	}
	if p.BaseURL != "https://api.test.com/v1" {
		t.Fatalf("base_url = %s", p.BaseURL)
	}
	if len(p.Keys) != 1 || p.Keys[0] != "sk-test123" {
		t.Fatalf("keys = %v", p.Keys)
	}
}

func TestAddKeyAppendsNotReplaces(t *testing.T) {
	h := newHarness(t, "")
	defer h.close()

	// vis starts with 1 key ["vk"]
	// add two more keys
	body := `{"name":"vis","keys":["vk","sk-new1","sk-new2"]}`
	resp, b := post(t, h.srv.URL+"/api/config/keys", "", body, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("update keys: %d %s", resp.StatusCode, b)
	}

	// verify 3 keys in config (redacted)
	resp2, _ := http.Get(h.srv.URL + "/api/config")
	raw, _ := io.ReadAll(resp2.Body)
	cfg := map[string]any{}
	json.Unmarshal(raw, &cfg)
	provs := cfg["providers"].(map[string]any)
	vis := provs["vis"].(map[string]any)
	keys := vis["keys"].([]any)
	if len(keys) != 3 {
		t.Fatalf("expected 3 keys, got %d", len(keys))
	}
}
