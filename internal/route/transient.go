package route

// Transient-failure in-place retry: a 502/503/504 from an otherwise healthy
// upstream is usually a blip (load-balancer hiccup, brief deploy), not a
// verdict on the candidate. Retrying the same key once in place is cheaper
// than cascading to the next candidate and burning its quota for what would
// have succeeded. Never applies to streamed requests (the body may already be
// arriving) or when upstream sent Retry-After.

import (
	"math/rand"
	"net/http"
	"time"

	"llm-router/internal/config"
)

// DefaultRetryTransientMax is the default number of in-place retries per attempt.
const DefaultRetryTransientMax = 1

// transientBase is the first backoff; it doubles (with jitter) per retry.
const transientBase = 500 * time.Millisecond

// RetryTransientMax returns the configured number of in-place retries for
// transient statuses. The zero value means disabled; operators opt in with
// fallback.retry_transient_max in router.yaml. Values above 3 are clamped.
func RetryTransientMax(fb config.FallbackCfg) int {
	if fb.RetryTransientMax < 0 {
		return 0
	}
	if fb.RetryTransientMax > 3 {
		return 3
	}
	return fb.RetryTransientMax
}

// isTransientStatus reports whether this status is worth an in-place retry.
// 502/503/504 are gateway-level blips; 500 is excluded because it is usually a
// deterministic server error that a retry cannot fix.
func isTransientStatus(status int) bool {
	return status == http.StatusBadGateway ||
		status == http.StatusServiceUnavailable ||
		status == http.StatusGatewayTimeout
}

// streamRequested reports whether the payload asks for a streamed response.
// A stream's body may already have started arriving when a failure surfaces,
// so replaying the request could duplicate partial output — never retry those.
func streamRequested(payload map[string]any) bool {
	s, ok := payload["stream"].(bool)
	return ok && s
}

// transientBackoff computes the sleep before retry i (0-based): exponential
// with jitter so simultaneous failures don't synchronize.
func transientBackoff(i int) time.Duration {
	d := transientBase << i
	jitter := time.Duration(rand.Int63n(int64(d) / 2))
	return d/2 + jitter
}
