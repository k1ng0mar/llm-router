// Command llm-router is a single-binary, self-hosted LLM router:
// OpenAI-compatible endpoint + capability gate + pools + fallback chain +
// SQLite event log + read/write dashboard.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"llm-router/internal/catalog"
	"llm-router/internal/config"
	"llm-router/internal/provider"
	"llm-router/internal/route"
	"llm-router/internal/server"
	"llm-router/internal/store"

	"gopkg.in/yaml.v3"
)

func main() {
	log.SetPrefix("router: ")
	log.SetFlags(log.LstdFlags)
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	cmd := os.Args[1]
	args := os.Args[2:]

	switch cmd {
	case "serve":
		serve(args)
	case "validate":
		validate(args)
	case "set-default":
		setDefault(args)
	case "log":
		logCmd(args)
	case "example-config":
		exampleConfig(args)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "router: unknown command %q\n", cmd)
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `llm-router — single-binary LLM router

usage:
  router serve  [--config router.yaml]            run the HTTP server
  router validate [--config router.yaml]          check config, exit nonzero on error
  router set-default <pool> [--config router.yaml]
  router log    [--config router.yaml] [--limit N] [--pool P] [--status S]
  router example-config [--out router.yaml]       write a starter config
`)
}

func loadCfg(args []string) *config.Config {
	fs := flag.NewFlagSet("cfg", flag.ExitOnError)
	path := fs.String("config", "router.yaml", "config file path")
	fs.Parse(args)
	cfg, err := config.LoadFile(*path)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		log.Fatalf("config invalid: %v", err)
	}
	return cfg
}

func serve(args []string) {
	cfg := loadCfg(args)

	if cfg.RouterKey == "" {
		if cfg.InsecureNoAuth {
			log.Printf("WARNING: router_key is empty and insecure_no_auth=true — admin API and chat endpoint are UNAUTHENTICATED")
		} else {
			log.Printf("WARNING: router_key is empty — all requests will be rejected unless you set router_key (or set insecure_no_auth: true for local dev)")
		}
	}

	st, err := store.Open(cfg.DBPath)
	if err != nil {
		log.Fatalf("store: %v", err)
	}
	defer st.Close()

	gate := catalog.NewGate(catalog.DefaultSeed())
	// The one deadline the router imposes: fallback.timeout_s bounds the wait
	// for response headers (time to first byte) on a single key. A hit rotates
	// to the next key, then the next provider — it never fails the request on
	// its own. Bodies stream freely once headers arrive, and there is
	// deliberately no overall request deadline.
	//
	// This has to be built here. NewRouter only falls back to a bounded client
	// when it is handed a nil one, which happens in tests and nowhere else — so
	// passing a bare &http.Client{} meant timeout_s had no effect at all in
	// production and a silent upstream could hang a request indefinitely.
	ttfb := time.Duration(cfg.GetFallback().TimeoutS) * time.Second
	client := provider.NewClientWithTTFB(ttfb)
	log.Printf("per-attempt TTFB timeout: %s (per key; rotates to the next key/provider on expiry)", ttfb)
	// refresh the models.dev catalog in the background; never block serving.
	// Run the first fetch immediately on boot (so the gate has remote data
	// right away instead of waiting a full day), then every 24h.
	go func() {
		for {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			if err := gate.Refresh(ctx, cfg.CatalogURL, client.HTTP); err != nil {
				log.Printf("catalog refresh failed (using seed/previous): %v", err)
			} else {
				log.Printf("catalog refreshed from %s", cfg.CatalogURL)
			}
			cancel()
			time.Sleep(24 * time.Hour)
		}
	}()

	// Periodic event-log pruning so the SQLite database cannot grow without
	// bound (full request/response bodies are large, especially with media).
	go func() {
		ticker := time.NewTicker(24 * time.Hour)
		defer ticker.Stop()
		for range ticker.C {
			if n, err := st.Prune(time.Now().Add(-30 * 24 * time.Hour)); err != nil {
				log.Printf("prune: %v", err)
			} else if n > 0 {
				log.Printf("pruned %d request rows older than 30d", n)
			}
		}
	}()

	r := route.NewRouter(cfg, gate, client)
	s := server.New(cfg, st, r, gate)

	// SIGHUP: reload config from disk (in-place, no restart) and drop cached
	// key pickers so any key/provider changes take effect immediately.
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGHUP)
		for range sig {
			if err := cfg.ReloadFile(cfg.Path); err != nil {
				log.Printf("config reload failed: %v", err)
				continue
			}
			r.InvalidateAllPickers()
			log.Printf("config reloaded from %s", cfg.Path)
		}
	}()

	// SIGINT/SIGTERM: graceful shutdown — drain in-flight requests and close
	// the store so pending log writes aren't dropped on restart/deploy.
	httpSrv := &http.Server{
		Addr:              cfg.Listen,
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
		<-sig
		log.Printf("shutting down (signal received)")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		if err := httpSrv.Shutdown(shutdownCtx); err != nil {
			log.Printf("shutdown: %v", err)
		}
		_ = st.Close()
		log.Printf("shutdown complete")
	}()
	log.Printf("listening on %s (db=%s, default pool=%s)", cfg.Listen, cfg.DBPath, cfg.Default)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("serve: %v", err)
	}
}

func validate(args []string) {
	cfg := loadCfg(args)
	providers := 0
	if cfg.Providers.OpenRouter != nil {
		providers++
	}
	if cfg.Providers.Ollama != nil {
		providers++
	}
	providers += len(cfg.Providers.Custom)
	fmt.Printf("OK: config %s valid (default pool=%s, pools=%d, providers=%d)\n",
		cfg.Path, cfg.Default, len(cfg.Pools), providers)
}

func setDefault(args []string) {
	if len(args) == 0 {
		log.Fatal("usage: router set-default <pool> [--config router.yaml]")
	}
	pool := args[0]
	cfg := loadCfg(args[1:])
	if err := cfg.SetDefault(pool); err != nil {
		log.Fatalf("set-default: %v", err)
	}
	fmt.Printf("OK: default pool is now %q\n", pool)
}

func logCmd(args []string) {
	fs := flag.NewFlagSet("log", flag.ExitOnError)
	path := fs.String("config", "router.yaml", "config file path")
	limit := fs.Int("limit", 20, "rows to show")
	pool := fs.String("pool", "", "filter by pool")
	status := fs.Int("status", 0, "filter by final status")
	fs.Parse(args)

	cfg, err := config.LoadFile(*path)
	if err != nil {
		log.Fatalf("config: %v", err)
	}
	st, err := store.Open(cfg.DBPath)
	if err != nil {
		log.Fatalf("store: %v", err)
	}
	defer st.Close()
	rows, err := st.ListRequests(store.Filter{Pool: *pool, Status: *status, Limit: *limit}, 0)
	if err != nil {
		log.Fatalf("log: %v", err)
	}
	fmt.Printf("%-22s %-10s %-14s %-7s %-24s attempts\n", "time", "pool", "rule", "status", "route")
	for _, x := range rows {
		route := x.FinalProvider + "/" + x.FinalModel
		fmt.Printf("%-22s %-10s %-14s %-7d %-24s %d\n", x.TS, x.Pool, x.Rule, x.FinalStatus, route, len(x.Attempts))
	}
}

func exampleConfig(args []string) {
	fs := flag.NewFlagSet("example", flag.ExitOnError)
	out := fs.String("out", "router.yaml", "output path")
	fs.Parse(args)
	cfg := config.DefaultConfig()
	cfg.Listen = "127.0.0.1:8011"
	cfg.RouterKey = "change-me"
	cfg.Pools = map[string][]string{
		"chat":      {"openrouter:openai/gpt-5.6-luna", "agnes:agnes-2.0-flash"},
		"code":      {"charm:deepseek-v4-flash", "xkiro:minimax/m3"},
		"creative":  {"openrouter:openai/gpt-5.6-luna", "xkiro:minimax/m3"},
		"reasoning": {"charm:deepseek-v4-flash"},
	}
	cfg.Vision = []string{"agnes:agnes-2.0-flash"}

	// Named chains — explicit fallback sequences that bypass the pool classifier.
	// Send model="chain:fast" or model="chain:smart" to use them.
	cfg.Chains = map[string][]string{
		"fast":    {"openrouter:openai/gpt-5.6-luna", "agnes:agnes-2.0-flash"},
		"smart":   {"charm:deepseek-v4-flash", "xkiro:minimax/m3"},
		"cheapest":{"agnes:agnes-2.0-flash"},
	}
	// Tier ordering per pool — cheapest first. The router tries the cheapest
	// tier that can handle the request; 429/5xx escalates to the next tier.
	cfg.Tiers = map[string][]string{
		"chat":      {"cheap", "standard"},
		"code":      {"cheap", "standard"},
		"creative":  {"cheap", "standard"},
		"reasoning": {"cheap", "standard"},
	}
	cfg.AllowDirectVision = true
	// All known providers pre-seeded with real OpenAI-compatible base URLs.
	// Keys are nil — the user sets them from the dashboard. A provider with
	// no keys is passively present: add it to a pool to activate routing.
	cfg.Providers = config.Providers{
		OpenRouter: &config.Provider{BaseURL: "https://openrouter.ai/api/v1", Keys: nil},
		Ollama:     &config.Provider{BaseURL: "http://127.0.0.1:11434", Keys: nil},
		Custom: map[string]*config.Provider{
			// key: provider name → real base URL (all OpenAI-compatible)
			"gh-copilot":     {BaseURL: "https://api.githubcopilot.com", Keys: nil},
			"google":         {BaseURL: "https://generativelanguage.googleapis.com/v1beta/openai", Keys: nil},
			"nous":           {BaseURL: "https://api.nousresearch.com/v1", Keys: nil},
			"ollama-cloud":   {BaseURL: "https://api.cloud.ollama.ai/v1", Keys: nil},
			"opencode":       {BaseURL: "https://api.opencode.ai/v1", Keys: nil},
			"charm":          {BaseURL: "https://hyper.charm.land/v1", Keys: nil},
			"nvidia-nim":     {BaseURL: "https://integrate.api.nvidia.com/v1", Keys: nil},
			"makora":         {BaseURL: "https://api.makora.ai/v1", Keys: nil},
			"tokenrouter":    {BaseURL: "https://api.tokenrouter.ai/v1", Keys: nil},
			"zai":            {BaseURL: "https://api.z.ai/v1", Keys: nil},
			"fireworks":      {BaseURL: "https://api.fireworks.ai/v1", Keys: nil},
			"agnes":          {BaseURL: "https://apihub.agnes-ai.com/v1", Keys: nil},
			"scaleway":       {BaseURL: "https://api.scaleway.com/ai/v1", Keys: nil},
			"xkiro":          {BaseURL: "https://api.xkiro.com/v1", Keys: nil},
			"freemodels":     {BaseURL: "https://api.freemodels.dev/v1", Keys: nil},
			"groq":           {BaseURL: "https://api.groq.com/openai/v1", Keys: nil},
			"kilo":           {BaseURL: "https://api.kilocode.ai/v1", Keys: nil},
			"openrouter-alt": {BaseURL: "https://openrouter.ai/api/v1", Keys: nil},
			"cerebras":       {BaseURL: "https://api.cerebras.ai/v1", Keys: nil},
			"lightning-ai":   {BaseURL: "https://api.lightning.ai/v1", Keys: nil},
			"hf-router":      {BaseURL: "https://router.huggingface.co/v1", Keys: nil},
			"cohere":         {BaseURL: "https://api.cohere.ai/compatibility/v1", Keys: nil},
			"cloudflare":     {BaseURL: "https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1", Keys: nil},
		},
	}
	f, err := os.Create(*out)
	if err != nil {
		log.Fatalf("create: %v", err)
	}
	defer f.Close()
	b, err := yaml.Marshal(cfg)
	if err != nil {
		log.Fatalf("marshal: %v", err)
	}
	f.Write(b)
	fmt.Printf("OK: wrote %s — set the keys inside it, then: router validate --config %s && router serve --config %s\n", *out, *out, *out)
}
