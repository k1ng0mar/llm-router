// Quota-aware pool ordering.
//
// quota.json (path set with SetQuotaFile, typically from a quota_file config
// key, default ./quota.json when enabled) carries per-provider remaining-quota
// snapshots, e.g. written by a plan tracker like hermes-resetwatch:
//
//	{"gemini": {"percent_left": 40, "reset_at": "2026-08-22T18:00:00Z"}}
//	{"gemini": 40}
//	[{"provider": "Gemini", "percent_left": 40}]
//
// Pool entries whose provider is running low sort after providers with
// headroom; exhausted ones (0% left, reset still in the future) sink further.
// Providers absent from the file are unpenalized; declared pool order is kept
// within each penalty bucket (stable sort), so without a file nothing changes.
package route

import (
	"encoding/json"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
)

// QuotaEntry is one provider's remaining-quota snapshot.
type QuotaEntry struct {
	PercentLeft *float64   `json:"percent_left"`
	ResetAt     *time.Time `json:"reset_at"`
}

var (
	qmu       sync.Mutex
	quotaPath string
	quotaData map[string]QuotaEntry
	quotaMod  time.Time
	quotaSize int64
	quotaLoad time.Time // rate-limit disk reads to once per second
)

var quotaKeyNorm = regexp.MustCompile(`[^a-z0-9]`)

func normProviderKey(s string) string {
	return quotaKeyNorm.ReplaceAllString(strings.ToLower(s), "")
}

// SetQuotaFile points quota lookups at path. Empty string disables quota
// awareness entirely (no disk access).
func SetQuotaFile(path string) {
	qmu.Lock()
	defer qmu.Unlock()
	quotaPath = path
	quotaData = nil
	quotaMod = time.Time{}
	quotaSize = 0
}

// QuotaHeadroomPct is the percent-left threshold below which a provider counts
// as "running low".
const QuotaHeadroomPct = 25.0

// quotaState loads quota.json if it changed since the last read (at most once
// per second). Fail-soft: any error keeps the previous data in place.
func quotaState() map[string]QuotaEntry {
	qmu.Lock()
	defer qmu.Unlock()
	if quotaPath == "" {
		return nil
	}
	now := time.Now()
	if quotaData != nil && now.Sub(quotaLoad) < time.Second {
		return quotaData
	}
	st, err := os.Stat(quotaPath)
	if err != nil {
		return quotaData
	}
	if quotaData != nil && st.ModTime().Equal(quotaMod) && st.Size() == quotaSize {
		return quotaData
	}
	raw, err := os.ReadFile(quotaPath)
	if err != nil {
		return quotaData
	}
	parsed := parseQuota(raw)
	if parsed != nil {
		quotaData = parsed
		quotaMod = st.ModTime()
		quotaSize = st.Size()
		quotaLoad = now
	}
	return quotaData
}

func parseQuota(raw []byte) map[string]QuotaEntry {
	parsed := map[string]QuotaEntry{}
	// Object form: {"name": {"percent_left": ..}} or shorthand {"name": 40}.
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err == nil {
		for name, v := range obj {
			var entry QuotaEntry
			if err := json.Unmarshal(v, &entry); err == nil && entry.PercentLeft != nil {
				parsed[normProviderKey(name)] = entry
				continue
			}
			var pct float64
			if err := json.Unmarshal(v, &pct); err == nil {
				p := pct // copy out of loop var
				parsed[normProviderKey(name)] = QuotaEntry{PercentLeft: &p}
			}
		}
		return parsed
	}
	// Array form: [{"provider": "Name", "percent_left": ..}].
	var arr []struct {
		Provider    string     `json:"provider"`
		Name        string     `json:"name"` // accepted alias
		PercentLeft *float64   `json:"percent_left"`
		ResetAt     *time.Time `json:"reset_at"`
	}
	if err := json.Unmarshal(raw, &arr); err == nil {
		for _, item := range arr {
			name := item.Provider
			if name == "" {
				name = item.Name
			}
			if name != "" && item.PercentLeft != nil {
				parsed[normProviderKey(name)] = QuotaEntry{PercentLeft: item.PercentLeft, ResetAt: item.ResetAt}
			}
		}
		return parsed
	}
	return nil
}

// QuotaPenalty buckets a provider's quota state: 0 = headroom or unknown,
// 1 = running low, 2 = exhausted with reset still pending. A provider whose
// reset time has already passed counts as unknown — its window may have filled.
func QuotaPenalty(provider string, data map[string]QuotaEntry, now time.Time) int {
	entry, ok := data[normProviderKey(provider)]
	if !ok || entry.PercentLeft == nil {
		return 0
	}
	pct := *entry.PercentLeft
	if pct <= 0 {
		if entry.ResetAt != nil && entry.ResetAt.Before(now) {
			return 0
		}
		return 2
	}
	if pct >= QuotaHeadroomPct {
		return 0
	}
	return 1
}

// providerOf extracts the provider name from a pool entry ("provider:model").
func providerOf(ref string) string {
	if i := strings.IndexByte(ref, ':'); i >= 0 {
		return ref[:i]
	}
	return ref
}

// QuotaSortedEntries reorders pool entries by quota penalty: group 0 (headroom/
// unknown) first, then low (1), then exhausted (2), keeping the declared order
// within each group. With no data it returns entries unchanged.
func QuotaSortedEntries(entries []string, data map[string]QuotaEntry, now time.Time) []string {
	if len(data) == 0 || len(entries) < 2 {
		return entries
	}
	type tagged struct {
		ref     string
		penalty int
	}
	named := make([]tagged, len(entries))
	for i, ref := range entries {
		named[i] = tagged{ref: ref, penalty: QuotaPenalty(providerOf(ref), data, now)}
	}
	sort.SliceStable(named, func(a, b int) bool { return named[a].penalty < named[b].penalty })
	out := make([]string, len(entries))
	for i, t := range named {
		out[i] = t.ref
	}
	return out
}
