package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"llm-router/internal/catalog"
	"llm-router/internal/classify"
	"llm-router/internal/config"
	"llm-router/internal/provider"
	"llm-router/internal/route"
	"llm-router/internal/store"
)

// mediaHarness builds a server whose `media` pool holds three single-modality
// models: an image-only, an audio-only, and a video-only one. Each records the
// hits it received, so a test can assert which model a request actually reached.
type mediaHarness struct {
	srv                       *httptest.Server
	cfg                       *config.Config
	store                     *store.Store
	up                        *httptest.Server
	imgHits, audHits, vidHits *atomic.Int32
	chatHits                  *atomic.Int32
}

func newMediaHarness(t *testing.T, policies map[string]config.MediaPolicy) *mediaHarness {
	t.Helper()
	var imgHits, audHits, vidHits, chatHits atomic.Int32
	counters := map[string]*atomic.Int32{
		"img": &imgHits, "aud": &audHits, "vid": &vidHits, "cm": &chatHits,
	}

	// one upstream for all models; the requested model id picks the counter
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		model, _ := body["model"].(string)
		if c, ok := counters[model]; ok {
			c.Add(1)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"answered:`+model+`"}}]}`)
	}))

	cfg := &config.Config{
		Default:        "chat",
		InsecureNoAuth: true,
		Pools: map[string][]string{
			"chat":  {"p:cm"},
			"media": {"p:img", "p:aud", "p:vid"},
		},
		Classifier: config.ClassifierCfg{Heuristics: map[string][]string{"code": {"import "}}},
		Providers: config.Providers{
			Custom: map[string]*config.Provider{
				"p": {BaseURL: up.URL, Keys: []string{"k"}, MediaPolicies: policies},
			},
		},
		Fallback: config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}

	// catalog: each media model handles exactly one modality
	g := catalog.NewGate(map[string]catalog.ModelInfo{
		"p:cm":  {ContextWindow: 100000},
		"p:img": {ContextWindow: 100000, Vision: true},
		"p:aud": {ContextWindow: 100000, Audio: true},
		"p:vid": {ContextWindow: 100000, Video: true},
	})

	st, err := store.Open(filepath.Join(t.TempDir(), "media.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	ts := httptest.NewServer(New(cfg, st, r, g).Handler())
	return &mediaHarness{srv: ts, cfg: cfg, store: st, up: up,
		imgHits: &imgHits, audHits: &audHits, vidHits: &vidHits, chatHits: &chatHits}
}

func (h *mediaHarness) close() {
	h.srv.Close()
	h.up.Close()
	h.store.Close()
}

func mediaBody(partType, field string) string {
	return `{"model":"auto","messages":[{"role":"user","content":[` +
		`{"type":"text","text":"what is this"},` +
		`{"type":"` + partType + `","` + field + `":{"url":"https://x"}}]}]}`
}

// TestMediaPoolRoutesEachModality is the headline behavior: the media pool serves
// images, audio and video alike, and the capability gate picks the one model in
// the pool that can actually handle the modality present.
func TestMediaPoolRoutesEachModality(t *testing.T) {
	cases := []struct {
		name     string
		partType string
		field    string
		want     func(*mediaHarness) *atomic.Int32
		others   func(*mediaHarness) []*atomic.Int32
	}{
		{"image", "image_url", "image_url",
			func(h *mediaHarness) *atomic.Int32 { return h.imgHits },
			func(h *mediaHarness) []*atomic.Int32 { return []*atomic.Int32{h.audHits, h.vidHits} }},
		{"audio", "input_audio", "input_audio",
			func(h *mediaHarness) *atomic.Int32 { return h.audHits },
			func(h *mediaHarness) []*atomic.Int32 { return []*atomic.Int32{h.imgHits, h.vidHits} }},
		{"video", "video_url", "video_url",
			func(h *mediaHarness) *atomic.Int32 { return h.vidHits },
			func(h *mediaHarness) []*atomic.Int32 { return []*atomic.Int32{h.imgHits, h.audHits} }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := newMediaHarness(t, nil)
			defer h.close()

			resp, body := post(t, h.srv.URL+"/v1/chat/completions", "", mediaBody(c.partType, c.field), nil)
			if resp.StatusCode != 200 {
				t.Fatalf("status = %d: %s", resp.StatusCode, body)
			}
			if got := c.want(h).Load(); got != 1 {
				t.Fatalf("%s-capable model hits = %d, want 1 (body: %s)", c.name, got, body)
			}
			for _, other := range c.others(h) {
				if other.Load() != 0 {
					t.Fatalf("a model lacking %s capability was called", c.name)
				}
			}
			if h.chatHits.Load() != 0 {
				t.Fatal("media request must not land in the chat pool")
			}
		})
	}
}

// TestMediaPoolAllowLetsUnderReportedModelThrough covers the operator override:
// the catalog says p:aud has no video, but a policy of video:allow says otherwise,
// and audio arrives first in the pool so it is the one that answers.
func TestMediaPoolAllowLetsUnderReportedModelThrough(t *testing.T) {
	h := newMediaHarness(t, map[string]config.MediaPolicy{
		"aud": {Video: config.PolicyAllow},
	})
	defer h.close()

	resp, body := post(t, h.srv.URL+"/v1/chat/completions", "", mediaBody("video_url", "video_url"), nil)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, body)
	}
	if h.audHits.Load() != 1 {
		t.Fatalf("aud hits = %d, want 1 — video:allow should let it take the video request", h.audHits.Load())
	}
	if h.vidHits.Load() != 0 {
		t.Fatal("the allowed model precedes p:vid in the pool, so p:vid should never be reached")
	}
}

// TestMediaPoolDenySkipsCapableModel covers the other direction: denying the one
// model the catalog says can handle a modality must push the request past it.
func TestMediaPoolDenySkipsCapableModel(t *testing.T) {
	h := newMediaHarness(t, map[string]config.MediaPolicy{
		"img": {Image: config.PolicyDeny},
		"aud": {Image: config.PolicyAllow}, // give the request somewhere to land
	})
	defer h.close()

	resp, body := post(t, h.srv.URL+"/v1/chat/completions", "", mediaBody("image_url", "image_url"), nil)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, body)
	}
	if h.imgHits.Load() != 0 {
		t.Fatal("image:deny must keep pixels away from p:img")
	}
	if h.audHits.Load() != 1 {
		t.Fatalf("aud hits = %d, want 1", h.audHits.Load())
	}
}

// TestMediaPoolExhaustsWhenNothingSupportsModality confirms the gate is a hard
// exclusion, not a warning: with every candidate denied, the caller gets a 503
// rather than a request forwarded to a model that cannot read it.
func TestMediaPoolExhaustsWhenNothingSupportsModality(t *testing.T) {
	h := newMediaHarness(t, map[string]config.MediaPolicy{
		"img": {Image: config.PolicyDeny},
	})
	defer h.close()

	resp, body := post(t, h.srv.URL+"/v1/chat/completions", "", mediaBody("image_url", "image_url"), nil)
	if resp.StatusCode != 503 {
		t.Fatalf("status = %d, want 503: %s", resp.StatusCode, body)
	}
	if h.imgHits.Load()+h.audHits.Load()+h.vidHits.Load() != 0 {
		t.Fatal("no upstream call should have been made")
	}
}

// TestNonMediaRequestIgnoresMediaPool guards the blast radius: a plain text
// request must still take the default pool.
func TestNonMediaRequestIgnoresMediaPool(t *testing.T) {
	h := newMediaHarness(t, nil)
	defer h.close()

	resp, body := post(t, h.srv.URL+"/v1/chat/completions", "",
		`{"model":"auto","messages":[{"role":"user","content":"good morning"}]}`, nil)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d: %s", resp.StatusCode, body)
	}
	if h.chatHits.Load() != 1 {
		t.Fatalf("chat hits = %d, want 1", h.chatHits.Load())
	}
}

// TestUpdateProviderMediaPolicies exercises the dashboard's save path end to end:
// the PUT stores policies, /api/config hands them back, and no-opinion rows are
// pruned rather than persisted.
func TestUpdateProviderMediaPolicies(t *testing.T) {
	h := newMediaHarness(t, nil)
	defer h.close()

	body := `{"model_limits":{"img":4096},"disabled_models":[],"media_policies":{` +
		`"img":{"image":"allow","audio":"deny","video":"auto"},` +
		`"aud":{"image":"auto","audio":"auto","video":"auto"}}}`
	req, _ := http.NewRequest("PUT", h.srv.URL+"/api/config/providers/p", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("put: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("put status = %d: %s", resp.StatusCode, raw)
	}

	// the all-auto entry for "aud" carries no opinion and must not be stored
	stored := h.cfg.Providers.Custom["p"].MediaPolicies
	if len(stored) != 1 {
		t.Fatalf("stored policies = %+v, want only the img entry", stored)
	}
	if got := stored["img"]; got.Image != config.PolicyAllow || got.Audio != config.PolicyDeny {
		t.Fatalf("img policy = %+v", got)
	}

	// /api/config must expose policies (and model limits) so the dashboard can
	// render the saved state instead of blank dropdowns
	cresp, err := http.Get(h.srv.URL + "/api/config")
	if err != nil {
		t.Fatalf("get config: %v", err)
	}
	craw, _ := io.ReadAll(cresp.Body)
	cresp.Body.Close()
	var cfg struct {
		Providers map[string]struct {
			ModelLimits   map[string]int                `json:"model_limits"`
			MediaPolicies map[string]config.MediaPolicy `json:"media_policies"`
		} `json:"providers"`
	}
	if err := json.Unmarshal(craw, &cfg); err != nil {
		t.Fatalf("unmarshal config: %v", err)
	}
	p := cfg.Providers["p"]
	if p.MediaPolicies["img"].Image != config.PolicyAllow {
		t.Fatalf("/api/config media_policies = %+v", p.MediaPolicies)
	}
	if p.ModelLimits["img"] != 4096 {
		t.Fatalf("/api/config model_limits = %+v, want img=4096", p.ModelLimits)
	}
}

// TestUpdateProviderRejectsBadPolicyValue keeps a typo from silently degrading to
// "auto" on every subsequent request.
func TestUpdateProviderRejectsBadPolicyValue(t *testing.T) {
	h := newMediaHarness(t, nil)
	defer h.close()

	body := `{"media_policies":{"img":{"image":"yes"}}}`
	req, _ := http.NewRequest("PUT", h.srv.URL+"/api/config/providers/p", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("put: %v", err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("status = %d, want 400: %s", resp.StatusCode, raw)
	}
}

// describeHarness builds a server where the target pool holds a text-only model
// and the media pool holds an image-capable describer. `failing` names a model
// whose upstream returns 500, for exercising the direct-vision fallback.
type describeHarness struct {
	srv   *httptest.Server
	store *store.Store
	up    *httptest.Server
	mu    sync.Mutex
	seen  []seenReq
}

type seenReq struct {
	model    string
	hadMedia bool
	text     string
}

// provSpec is one upstream provider in a describe harness: its own key stack and
// its own per-model media policies. Tests give each model its own provider by
// default, because API keys — and therefore failure cooldowns — are per provider,
// not per model (see TestSharedKeyCooldownBlocksDescribeRetry).
type provSpec struct {
	policies map[string]config.MediaPolicy
}

func newDescribeHarness(t *testing.T, pools map[string][]string, provs map[string]provSpec, failing string) *describeHarness {
	t.Helper()
	h := &describeHarness{}
	h.up = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		model, _ := body["model"].(string)
		msgs, _ := body["messages"].([]any)
		rec := seenReq{model: model, hadMedia: classify.DetectMedia(msgs).Any()}
		for _, m := range msgs {
			if mm, ok := m.(map[string]any); ok {
				if sc, ok := mm["content"].(string); ok {
					rec.text += sc
				}
			}
		}
		h.mu.Lock()
		h.seen = append(h.seen, rec)
		h.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if model == failing {
			w.WriteHeader(500)
			io.WriteString(w, `{"error":"upstream on fire"}`)
			return
		}
		reply := "answered:" + model
		if model == "vis" {
			reply = "a blue login form"
		}
		w.WriteHeader(200)
		io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"`+reply+`"}}]}`)
	}))

	custom := map[string]*config.Provider{}
	for name, spec := range provs {
		custom[name] = &config.Provider{
			BaseURL:       h.up.URL,
			Keys:          []string{name + "-k1", name + "-k2"},
			MediaPolicies: spec.policies,
		}
	}
	cfg := &config.Config{
		Default:           "chat",
		InsecureNoAuth:    true,
		AllowDirectVision: true,
		Pools:             pools,
		Classifier:        config.ClassifierCfg{Heuristics: map[string][]string{"code": {"import "}, "creative": {"write a poem"}}},
		Providers:         config.Providers{Custom: custom},
		Fallback:          config.FallbackCfg{TimeoutS: 30, Strategy: "round_robin", KeyCooldownS: 60},
	}
	g := catalog.NewGate(nil) // no catalog: the media policies are the authority
	st, err := store.Open(filepath.Join(t.TempDir(), "desc.db"))
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	h.store = st
	r := route.NewRouter(cfg, g, &provider.Client{HTTP: http.DefaultClient})
	h.srv = httptest.NewServer(New(cfg, st, r, g).Handler())
	return h
}

func (h *describeHarness) close() {
	h.srv.Close()
	h.up.Close()
	h.store.Close()
}

func (h *describeHarness) models() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]string, 0, len(h.seen))
	for _, s := range h.seen {
		out = append(out, s.model)
	}
	return out
}

func (h *describeHarness) req(t *testing.T, text string) (int, string) {
	t.Helper()
	body := `{"model":"auto","messages":[{"role":"user","content":[` +
		`{"type":"text","text":"` + text + `"},` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}]}`
	resp, b := post(t, h.srv.URL+"/v1/chat/completions", "", body, nil)
	return resp.StatusCode, b
}

// TestDescribeHopRunsForAnyPool covers the pool scope: the hop used to be
// hardcoded to code/reasoning, so an image sent to the creative pool 503'd
// without the describer ever being tried.
func TestDescribeHopRunsForAnyPool(t *testing.T) {
	h := newDescribeHarness(t, map[string][]string{
		"chat":     {"pc:cm"},
		"creative": {"pw:writer"},
		"media":    {"pv:vis"},
	}, map[string]provSpec{
		"pc": {},
		"pw": {policies: map[string]config.MediaPolicy{"writer": {Image: config.PolicyDeny}}},
		"pv": {policies: map[string]config.MediaPolicy{"vis": {Image: config.PolicyAllow}}},
	}, "")
	defer h.close()

	status, body := h.req(t, "write a poem about this")
	if status != 200 {
		t.Fatalf("status = %d, want 200: %s", status, body)
	}
	if got := h.models(); len(got) != 2 || got[0] != "vis" || got[1] != "writer" {
		t.Fatalf("upstream order = %v, want [vis writer]", got)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.seen[0].hadMedia {
		t.Fatal("the describer must receive the pixels")
	}
	if h.seen[1].hadMedia {
		t.Fatal("the text-only model must not receive pixels")
	}
	// the user's question survives the hop alongside the description
	if !strings.Contains(h.seen[1].text, "write a poem about this") {
		t.Fatalf("the original question must reach the pool model: %q", h.seen[1].text)
	}
	if !strings.Contains(h.seen[1].text, "a blue login form") {
		t.Fatalf("the description must reach the pool model: %q", h.seen[1].text)
	}
}

// TestDirectVisionFallsBackToDescribe covers the failure path: pixels went
// straight to a vision-capable model, that model died, and the text-only models
// the gate had to exclude become reachable once the image is described.
func TestDirectVisionFallsBackToDescribe(t *testing.T) {
	h := newDescribeHarness(t, map[string][]string{
		"chat":  {"pc:cm"},
		"code":  {"pd:vcode", "pt:textcoder"},
		"media": {"pv:vis"},
	}, map[string]provSpec{
		"pc": {},
		"pd": {policies: map[string]config.MediaPolicy{"vcode": {Image: config.PolicyAllow}}},
		"pt": {policies: map[string]config.MediaPolicy{"textcoder": {Image: config.PolicyDeny}}},
		"pv": {policies: map[string]config.MediaPolicy{"vis": {Image: config.PolicyAllow}}},
	}, "vcode") // the vision-capable code model is down
	defer h.close()

	status, body := h.req(t, "import this and build it")
	if status != 200 {
		t.Fatalf("status = %d, want 200 — the describe hop should rescue this: %s", status, body)
	}
	got := h.models()
	if len(got) < 3 || got[0] != "vcode" {
		t.Fatalf("upstream order = %v, want the direct pixel attempt first", got)
	}
	// pd carries two keys, so vcode is attempted once per key before the hop
	if got[len(got)-2] != "vis" || got[len(got)-1] != "textcoder" {
		t.Fatalf("upstream order = %v, want [... vis textcoder] after the direct pass failed", got)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.seen[0].hadMedia {
		t.Fatal("the direct pass should have sent pixels to the vision-capable model")
	}
	last := h.seen[len(h.seen)-1]
	if last.hadMedia {
		t.Fatal("the retry must reach the text-only model as text")
	}
	if !strings.Contains(last.text, "import this and build it") {
		t.Fatalf("the question must survive the retry: %q", last.text)
	}
	if !strings.Contains(last.text, "a blue login form") {
		t.Fatalf("the description must reach the retry: %q", last.text)
	}
}

// TestSharedKeyProviderStillReachableAfterFailure is the inverse of a limitation
// this suite used to pin: API keys are per provider, and a 5xx used to cool them,
// so one model failing took every OTHER model on that provider offline — including
// the text-only model the describe retry needs. A 5xx no longer touches key
// availability, so a pool whose entries share a provider now works.
func TestSharedKeyProviderStillReachableAfterFailure(t *testing.T) {
	h := newDescribeHarness(t, map[string][]string{
		"chat":  {"pc:cm"},
		"code":  {"p:vcode", "p:textcoder"}, // same provider == same key stack
		"media": {"pv:vis"},
	}, map[string]provSpec{
		"pc": {},
		"p": {policies: map[string]config.MediaPolicy{
			"vcode":     {Image: config.PolicyAllow},
			"textcoder": {Image: config.PolicyDeny},
		}},
		"pv": {policies: map[string]config.MediaPolicy{"vis": {Image: config.PolicyAllow}}},
	}, "vcode")
	defer h.close()

	status, body := h.req(t, "import this and build it")
	if status != 200 {
		t.Fatalf("status = %d, want 200 — vcode's 5xx must not strand textcoder "+
			"on the same provider: %s", status, body)
	}
	got := h.models()
	if got[len(got)-1] != "textcoder" {
		t.Fatalf("upstream order = %v, want textcoder to answer last", got)
	}
}

// TestDescribeRetryDoesNotRetreadFailedCandidates: the retry excludes candidates
// whose pixel attempts already failed. Re-treading them would fail identically
// and cost a full per-attempt timeout per key on an error path.
func TestDescribeRetryDoesNotRetreadFailedCandidates(t *testing.T) {
	h := newDescribeHarness(t, map[string][]string{
		"chat":  {"pc:cm"},
		"code":  {"pd:vcode", "pt:textcoder"},
		"media": {"pv:vis"},
	}, map[string]provSpec{
		"pc": {},
		"pd": {policies: map[string]config.MediaPolicy{"vcode": {Image: config.PolicyAllow}}},
		"pt": {policies: map[string]config.MediaPolicy{"textcoder": {Image: config.PolicyDeny}}},
		"pv": {policies: map[string]config.MediaPolicy{"vis": {Image: config.PolicyAllow}}},
	}, "vcode")
	defer h.close()

	status, body := h.req(t, "import this and build it")
	if status != 200 {
		t.Fatalf("status = %d: %s", status, body)
	}
	// pd has two keys, so vcode is tried once per key, then the describer runs,
	// then the retry goes straight to textcoder — vcode must not reappear.
	got := h.models()
	visAt := -1
	for i, m := range got {
		if m == "vis" {
			visAt = i
		}
	}
	if visAt < 0 {
		t.Fatalf("upstream order = %v, want the describer to have run", got)
	}
	for _, m := range got[visAt+1:] {
		if m == "vcode" {
			t.Fatalf("upstream order = %v: vcode must not be re-tried after the describe hop", got)
		}
	}
	if got[len(got)-1] != "textcoder" {
		t.Fatalf("upstream order = %v, want textcoder last", got)
	}
}

// TestMediaPoolItselfKeepsPixels guards the exemption: the media pool is where
// describers live, so a request routed there must not be described first.
func TestMediaPoolItselfKeepsPixels(t *testing.T) {
	h := newDescribeHarness(t, map[string][]string{
		"chat":  {"pc:cm"},
		"media": {"pv:vis"},
	}, map[string]provSpec{
		"pc": {},
		"pv": {policies: map[string]config.MediaPolicy{"vis": {Image: config.PolicyAllow}}},
	}, "")
	defer h.close()

	status, body := h.req(t, "what is this")
	if status != 200 {
		t.Fatalf("status = %d: %s", status, body)
	}
	if got := h.models(); len(got) != 1 || got[0] != "vis" {
		t.Fatalf("upstream calls = %v, want a single direct call to [vis]", got)
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.seen[0].hadMedia {
		t.Fatal("the media pool must receive the pixels directly, not a description")
	}
}

// TestNoDescriberKeepsPixelsForGate covers the no-op case: with nothing
// image-capable configured, behavior is unchanged — pixels go out and the gate
// decides, rather than the router inventing a 400.
func TestNoDescriberKeepsPixelsForGate(t *testing.T) {
	h := newDescribeHarness(t, map[string][]string{
		"chat": {"pc:cm"},
		"code": {"pt:textcoder"},
	}, map[string]provSpec{
		"pc": {},
		"pt": {policies: map[string]config.MediaPolicy{"textcoder": {Image: config.PolicyDeny}}},
	}, "")
	defer h.close()

	status, _ := h.req(t, "import this and build it")
	if status != 503 {
		t.Fatalf("status = %d, want 503 from the gate (no describer configured)", status)
	}
	if got := h.models(); len(got) != 0 {
		t.Fatalf("no upstream should be called, got %v", got)
	}
}
