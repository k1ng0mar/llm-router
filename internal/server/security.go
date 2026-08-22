package server

// Security middleware: response headers and auth-failure backoff.
//
// Headers make any escaping miss in the dashboard non-exploitable (no framing,
// no sniffing, scripts confined to what the page already ships). The backoff
// makes brute-forcing router_key expensive instead of free: after
// authBackoffMaxFailures misses inside the window, the client IP gets 429 +
// Retry-After until the window rolls off. A correct key clears its IP's count.

import (
	"net/http"
	"strings"
	"sync"
	"time"
)

// tracker returns the server's auth-failure tracker, building it on first use
// so a hand-constructed Server (tests) never dereferences a nil map.
func (s *Server) tracker() *authBackoffTracker {
	if s.authFails == nil {
		s.authFails = newAuthBackoffTracker()
	}
	return s.authFails
}

// securityHeaders sets defensive response headers on every reply. The CSP keeps
// `unsafe-inline` (the dashboard is a single inline-script HTML file) but locks
// down everything else: no framing, no object embeds, no cross-origin form
// posts, no sniffing.
func (s *Server) securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Content-Security-Policy",
			"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "+
				"img-src 'self' data:; font-src 'self'; connect-src 'self'; "+
				"form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
		h.Set("X-Content-Type-Options", "nosniff")
		h.Set("X-Frame-Options", "DENY")
		h.Set("Referrer-Policy", "no-referrer")
		next.ServeHTTP(w, r)
	})
}

const (
	authBackoffMaxFailures = 10              // misses before the IP is blocked
	authBackoffWindow      = 5 * time.Minute // rolling window + block duration
)

// authFailure tracks one client IP's recent failed authentications.
type authFailure struct {
	count        int
	firstAt      time.Time
	blockedUntil time.Time
}

// authBackoffTracker holds per-IP failure state for the backoff middleware.
type authBackoffTracker struct {
	mu    sync.Mutex
	fails map[string]*authFailure
}

func newAuthBackoffTracker() *authBackoffTracker {
	return &authBackoffTracker{fails: map[string]*authFailure{}}
}

// blocked reports whether ip is currently blocked and for how long longer.
func (t *authBackoffTracker) blocked(ip string, now time.Time) (bool, time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	f, ok := t.fails[ip]
	if !ok || !now.Before(f.blockedUntil) {
		return false, 0
	}
	return true, f.blockedUntil.Sub(now)
}

// record notes a failed auth from ip, blocking it once the failure count
// reaches the threshold within the window. Returns the block status after recording.
func (t *authBackoffTracker) record(ip string, now time.Time) (blocked bool, retryIn time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	f, ok := t.fails[ip]
	if !ok || now.Sub(f.firstAt) > authBackoffWindow {
		f = &authFailure{firstAt: now}
		t.fails[ip] = f
	}
	f.count++
	if f.count >= authBackoffMaxFailures {
		f.blockedUntil = now.Add(authBackoffWindow)
		blocked = true
		retryIn = authBackoffWindow
	}
	// Bound the map: drop stale entries when it grows past a sane size.
	if len(t.fails) > 10_000 {
		for k, v := range t.fails {
			if now.Sub(v.firstAt) > authBackoffWindow && now.After(v.blockedUntil) {
				delete(t.fails, k)
			}
		}
	}
	return blocked, retryIn
}

// clear resets an IP's count after a successful authentication.
func (t *authBackoffTracker) clear(ip string) {
	t.mu.Lock()
	delete(t.fails, ip)
	t.mu.Unlock()
}

// clientIP extracts the caller address. X-Forwarded-For is honored only as the
// leftmost entry (the original client behind one trusted proxy hop); direct
// RemoteAddr is the fallback.
func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		if i := strings.IndexByte(xff, ','); i >= 0 {
			xff = xff[:i]
		}
		return strings.TrimSpace(xff)
	}
	i := strings.LastIndexByte(r.RemoteAddr, ':')
	if i <= 0 {
		return r.RemoteAddr
	}
	return r.RemoteAddr[:i]
}

// authBackoff wraps next with per-IP brute-force protection. A request that is
// currently blocked gets 429 + Retry-After without reaching any handler; a
// failed authentication records a strike via noteAuthFailure; a success clears
// the IP via clearAuthFailures. Both hooks are called from authorized()-based
// handlers through the helpers below.
func (s *Server) authBackoff(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ip := clientIP(r)
		if blocked, retryIn := s.tracker().blocked(ip, time.Now()); blocked {
			w.Header().Set("Retry-After", secondsCeil(retryIn))
			writeJSON(w, http.StatusTooManyRequests, map[string]any{
				"error": map[string]string{
					"message": "too many failed authentications; retry later",
				},
			})
			return
		}
		next.ServeHTTP(w, r)
	})
}

// noteAuthFailure records one bad credential from the request's IP.
func (s *Server) noteAuthFailure(r *http.Request) {
	s.tracker().record(clientIP(r), time.Now())
}

// clearAuthFailures resets the request's IP after a successful auth.
func (s *Server) clearAuthFailures(r *http.Request) {
	s.tracker().clear(clientIP(r))
}

func secondsCeil(d time.Duration) string {
	s := int(d.Seconds())
	if s < 1 {
		s = 1
	}
	return itoa(s)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
