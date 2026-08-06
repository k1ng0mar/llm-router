// Package store persists the request/attempt event log in SQLite (WAL).
// This is the single source of truth for both the dashboard and the CLI.
package store

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

// RequestRow is one routed request.
type RequestRow struct {
	ID            string  `json:"id"`
	TS            string  `json:"ts"`
	TsUnix        int64   `json:"ts_unix"`
	Pool          string  `json:"pool"`
	Rule          string  `json:"rule"`
	FinalStatus   int     `json:"final_status"`
	FinalProvider string  `json:"final_provider"`
	FinalModel    string  `json:"final_model"`
	PromptTokens  int     `json:"prompt_tokens"`
	CompletionTok int     `json:"completion_tokens"`
	Cost          float64 `json:"cost"` // always 0 for now — pending a pricing table
	TotalMs       int     `json:"total_ms"`
	ErrorOrigin   string  `json:"error_origin"`
	RequestBody   string  `json:"request_body"`
	ResponseBody  string  `json:"response_body"`
}

// AttemptRow is one upstream attempt within a request.
type AttemptRow struct {
	ID          int64  `json:"id"`
	RequestID   string `json:"request_id"`
	Seq         int    `json:"seq"`
	Provider    string `json:"provider"`
	Model       string `json:"model"`
	KeyID       string `json:"key_id"`
	Status      int    `json:"status"`
	LatencyMs   int    `json:"latency_ms"`
	Cost        float64 `json:"cost"` // always 0 for now — pending a pricing table
	Err         string `json:"err"`
	ErrorOrigin string `json:"error_origin"`
}

// RequestWithAttempts is a request plus its attempt trail.
type RequestWithAttempts struct {
	RequestRow
	Attempts []AttemptRow
}

// Filter narrows ListRequests.
type Filter struct {
	Pool      string
	Status    int // 0 = any
	RequestID string
	Limit     int
	Offset    int
}

// Store wraps the SQLite database.
type Store struct {
	db *sql.DB
}

// Open opens (creating if needed) the log database with WAL + relaxed sync.
func Open(path string) (*Store, error) {
	dsn := fmt.Sprintf("file:%s?_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)&_pragma=busy_timeout(5000)", path)
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) migrate() error {
	// Create the tables first (idempotent). Then ALTER in any column a legacy
	// database is missing. PRAGMA table_info reliably reports column presence
	// once the table exists.
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS requests (
			id TEXT PRIMARY KEY,
			ts TEXT NOT NULL,
			ts_unix INTEGER DEFAULT 0,
			pool TEXT NOT NULL,
			rule TEXT NOT NULL,
			final_status INTEGER NOT NULL,
			final_provider TEXT,
			final_model TEXT,
			prompt_tokens INTEGER DEFAULT 0,
			completion_tokens INTEGER DEFAULT 0,
			cost REAL DEFAULT 0,
			total_ms INTEGER DEFAULT 0,
			error_origin TEXT DEFAULT 'router',
			request_body TEXT DEFAULT '',
			response_body TEXT DEFAULT ''
		)`,
		`CREATE TABLE IF NOT EXISTS attempts (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				request_id TEXT NOT NULL,
				seq INTEGER NOT NULL,
				provider TEXT NOT NULL,
				model TEXT NOT NULL,
				key_id TEXT,
				status INTEGER NOT NULL,
				latency_ms INTEGER DEFAULT 0,
				cost REAL DEFAULT 0,
				error TEXT,
				error_origin TEXT DEFAULT 'upstream'
			)`,
		`CREATE TABLE IF NOT EXISTS provider_models (
				provider TEXT NOT NULL,
				model_id TEXT NOT NULL,
				source TEXT DEFAULT 'fetched',
				PRIMARY KEY (provider, model_id)
			)`,
		`CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts DESC)`,
		`CREATE INDEX IF NOT EXISTS idx_attempts_request ON attempts(request_id)`,
		`CREATE INDEX IF NOT EXISTS idx_provider_models ON provider_models(provider)`,
	}
	for _, q := range stmts {
		if _, err := s.db.Exec(q); err != nil {
			return fmt.Errorf("migrate: %w", err)
		}
	}

	hasColumn := func(table, col string) bool {
		rows, err := s.db.Query(fmt.Sprintf(`PRAGMA table_info(%s)`, table))
		if err != nil {
			return false
		}
		defer rows.Close()
		for rows.Next() {
			var cid int
			var name, ctype string
			var notnull, pk int
			var dflt valueScanner
			if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err != nil {
				return false
			}
			if name == col {
				return true
			}
		}
		return false
	}
	addCol := func(table, col, dtype, dfltVal string) error {
		if hasColumn(table, col) {
			return nil
		}
		if _, err := s.db.Exec(fmt.Sprintf(`ALTER TABLE %s ADD COLUMN %s %s DEFAULT %s`, table, col, dtype, dfltVal)); err != nil {
			return fmt.Errorf("migrate %s.%s: %w", table, col, err)
		}
		return nil
	}
	if err := addCol("requests", "error_origin", "TEXT", "'router'"); err != nil {
		return err
	}
	if err := addCol("requests", "request_body", "TEXT", "''"); err != nil {
		return err
	}
	if err := addCol("requests", "response_body", "TEXT", "''"); err != nil {
		return err
	}
	if err := addCol("requests", "ts_unix", "INTEGER", "0"); err != nil {
		return err
	}
	if err := addCol("attempts", "error_origin", "TEXT", "'upstream'"); err != nil {
		return err
	}
	// The ts_unix index must be created AFTER the column is guaranteed to
	// exist (a legacy DB may lack it until the addCol above). SQLite's
	// CREATE INDEX IF NOT EXISTS does NOT tolerate a missing column — it
	// errors even with IF NOT EXISTS — so this has to come after the ALTERs.
	if _, err := s.db.Exec(`CREATE INDEX IF NOT EXISTS idx_requests_ts_unix ON requests(ts_unix)`); err != nil {
		return fmt.Errorf("migrate idx_requests_ts_unix: %w", err)
	}
	return nil
}

// valueScanner is a minimal sql.Scanner target for PRAGMA default-value column.
type valueScanner struct{ v any }

func (v *valueScanner) Scan(src any) error { v.v = src; return nil }

// Close closes the database.
func (s *Store) Close() error { return s.db.Close() }

// AddRequest writes one request row.
func (s *Store) AddRequest(r *RequestRow) error {
	origin := r.ErrorOrigin
	if origin == "" {
		origin = "router"
	}
	_, err := s.db.Exec(
		`INSERT INTO requests (id, ts, ts_unix, pool, rule, final_status, final_provider, final_model, prompt_tokens, completion_tokens, cost, total_ms, error_origin, request_body, response_body)
 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		r.ID, r.TS, r.TsUnix, r.Pool, r.Rule, r.FinalStatus, r.FinalProvider, r.FinalModel, r.PromptTokens, r.CompletionTok, r.Cost, r.TotalMs, origin, r.RequestBody, r.ResponseBody)
	return err
}

// AddAttempt writes one attempt row. ErrorOrigin defaults to "upstream"
// when empty; set "router" when the router refused before forwarding.
func (s *Store) AddAttempt(a *AttemptRow) error {
	origin := a.ErrorOrigin
	if origin == "" {
		origin = "upstream"
	}
	_, err := s.db.Exec(
		`INSERT INTO attempts (request_id, seq, provider, model, key_id, status, latency_ms, cost, error, error_origin)
	 VALUES (?,?,?,?,?,?,?,?,?,?)`,
		a.RequestID, a.Seq, a.Provider, a.Model, a.KeyID, a.Status, a.LatencyMs, a.Cost, a.Err, origin)
	return err
}

// ListRequests returns requests matching the filter, newest first, with attempts.
func (s *Store) ListRequests(f Filter, _ int) ([]RequestWithAttempts, error) {
	limit := f.Limit
	if limit <= 0 {
		limit = 50
	}
	offset := f.Offset
	q := `SELECT id, ts, pool, rule, final_status, final_provider, final_model, prompt_tokens, completion_tokens, cost, total_ms, error_origin, request_body, response_body
      FROM requests WHERE 1=1`
	args := []any{}
	if f.Pool != "" {
		q += ` AND pool = ?`
		args = append(args, f.Pool)
	}
	if f.Status != 0 {
		q += ` AND final_status = ?`
		args = append(args, f.Status)
	}
	if f.RequestID != "" {
		q += ` AND id = ?`
		args = append(args, f.RequestID)
	}
	q += ` ORDER BY ts DESC LIMIT ? OFFSET ?`
	args = append(args, limit, offset)

	rows, err := s.db.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []RequestWithAttempts
	var ids []string
	for rows.Next() {
		var r RequestWithAttempts
		if err := rows.Scan(&r.ID, &r.TS, &r.Pool, &r.Rule, &r.FinalStatus, &r.FinalProvider, &r.FinalModel, &r.PromptTokens, &r.CompletionTok, &r.Cost, &r.TotalMs, &r.ErrorOrigin, &r.RequestBody, &r.ResponseBody); err != nil {
			return nil, err
		}
		ids = append(ids, r.ID)
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return out, nil
	}
	// Batch-fetch all attempts in one query (eliminates N+1).
	attMap, err := s.attemptsForIDs(ids)
	if err != nil {
		return nil, err
	}
	for i := range out {
		out[i].Attempts = attMap[out[i].ID]
	}
	return out, nil
}

// CountRequests returns the total number of requests matching the filter.
func (s *Store) CountRequests(f Filter) (int, error) {
	q := `SELECT COUNT(*) FROM requests WHERE 1=1`
	args := []any{}
	if f.Pool != "" {
		q += ` AND pool = ?`
		args = append(args, f.Pool)
	}
	if f.Status != 0 {
		q += ` AND final_status = ?`
		args = append(args, f.Status)
	}
	if f.RequestID != "" {
		q += ` AND id = ?`
		args = append(args, f.RequestID)
	}
	var total int
	err := s.db.QueryRow(q, args...).Scan(&total)
	return total, err
}

// Stats holds aggregate metrics for the dashboard overview.
type Stats struct {
	Total       int     `json:"total"`
	Success     int     `json:"success"`
	Failed      int     `json:"failed"`
	Recovered   int     `json:"recovered"`
	SuccessRate float64 `json:"success_rate"`
	Tokens      int     `json:"tokens"`
	Cost        float64 `json:"cost"`
	Daily       []DailyStat `json:"daily"`
}

// DailyStat is one day's breakdown for the bar chart.
type DailyStat struct {
	Date     string `json:"date"`
	Success  int    `json:"success"`
	Recovered int   `json:"recovered"`
	Failed   int    `json:"failed"`
}

// ModelUsageRow is one model's aggregate usage.
type ModelUsageRow struct {
	Model       string  `json:"model"`
	Provider    string  `json:"provider"`
	Tokens      int     `json:"tokens"`
	Pct         float64 `json:"pct"`
	Cost        float64 `json:"cost"`
	Attempts    int     `json:"attempts"`
	SuccessRate float64 `json:"success_rate"`
}

// Stats returns aggregate metrics. rangeDays controls the window (7, 30, 0=all).
func (s *Store) Stats(rangeDays int) (*Stats, error) {
	var st Stats
	q := `SELECT COUNT(*),
		SUM(CASE WHEN final_status >= 200 AND final_status < 300 THEN 1 ELSE 0 END),
		SUM(CASE WHEN final_status >= 400 THEN 1 ELSE 0 END),
		SUM(CASE WHEN final_status >= 200 AND final_status < 300 AND id IN (SELECT request_id FROM attempts GROUP BY request_id HAVING COUNT(*) > 1) THEN 1 ELSE 0 END),
		SUM(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)),
		SUM(COALESCE(cost, 0))
		FROM requests`
	args := []any{}
	if rangeDays > 0 {
		q += ` WHERE ts_unix >= CAST(strftime('%s','now', ?) AS INTEGER)`
		args = append(args, fmt.Sprintf("-%d days", rangeDays))
	}
	var total, succ, fail, recovered, tokens int
	var cost float64
	var nSucc, nFail, nRec, nTok sql.NullInt64
	var nCost sql.NullFloat64
	err := s.db.QueryRow(q, args...).Scan(&total, &nSucc, &nFail, &nRec, &nTok, &nCost)
	if err != nil {
		return nil, err
	}
	succ = int(nSucc.Int64); fail = int(nFail.Int64); recovered = int(nRec.Int64); tokens = int(nTok.Int64)
	cost = nCost.Float64
	st.Total = total; st.Success = succ; st.Failed = fail; st.Recovered = recovered; st.Tokens = tokens; st.Cost = cost
	if st.Total > 0 {
		st.SuccessRate = float64(st.Success) / float64(st.Total) * 100
	}
	// daily breakdown
	dq := `SELECT substr(ts, 1, 10) as date,
		SUM(CASE WHEN final_status >= 200 AND final_status < 300 THEN 1 ELSE 0 END),
		SUM(CASE WHEN final_status >= 200 AND final_status < 300 AND id IN (SELECT request_id FROM attempts GROUP BY request_id HAVING COUNT(*) > 1) THEN 1 ELSE 0 END),
		SUM(CASE WHEN final_status >= 400 THEN 1 ELSE 0 END)
		FROM requests`
	if rangeDays > 0 {
		dq += ` WHERE ts_unix >= CAST(strftime('%s','now', ?) AS INTEGER)`
	}
	dq += ` GROUP BY date ORDER BY date`
	dargs := []any{}
	if rangeDays > 0 {
		dargs = append(dargs, fmt.Sprintf("-%d days", rangeDays))
	}
	rows, err := s.db.Query(dq, dargs...)
	if err != nil {
		return &st, nil // return what we have
	}
	defer rows.Close()
	for rows.Next() {
		var d DailyStat
		if err := rows.Scan(&d.Date, &d.Success, &d.Recovered, &d.Failed); err != nil {
			continue
		}
		st.Daily = append(st.Daily, d)
	}
	return &st, nil
}

// ModelUsage returns per-model aggregate usage stats.
func (s *Store) ModelUsage(rangeDays int) ([]ModelUsageRow, error) {
	q := `SELECT final_model, final_provider,
		SUM(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)),
		SUM(COALESCE(cost, 0)),
		COUNT(*),
		SUM(CASE WHEN final_status >= 200 AND final_status < 300 THEN 1 ELSE 0 END)
		FROM requests
		WHERE final_model != '' AND final_model IS NOT NULL`
	args := []any{}
	if rangeDays > 0 {
		q += ` AND ts_unix >= CAST(strftime('%s','now', ?) AS INTEGER)`
		args = append(args, fmt.Sprintf("-%d days", rangeDays))
	}
	q += ` GROUP BY final_model ORDER BY SUM(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)) DESC`
	rows, err := s.db.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ModelUsageRow
	var totalTokens int
	type tmpRow struct {
		mr   ModelUsageRow
		toks int
	}
	var tmps []tmpRow
	for rows.Next() {
		var t tmpRow
		var successCount int
		if err := rows.Scan(&t.mr.Model, &t.mr.Provider, &t.toks, &t.mr.Cost, &t.mr.Attempts, &successCount); err != nil {
			continue
		}
		if t.mr.Attempts > 0 {
			t.mr.SuccessRate = float64(successCount) / float64(t.mr.Attempts) * 100
		}
		t.mr.Tokens = t.toks
		totalTokens += t.toks
		tmps = append(tmps, t)
	}
	for _, t := range tmps {
		if totalTokens > 0 {
			t.mr.Pct = float64(t.toks) / float64(totalTokens) * 100
		}
		out = append(out, t.mr)
	}
	return out, nil
}

// attemptsForIDs batch-fetches attempts for multiple request IDs in a single
// query, returning a map from request_id to its attempts (ordered by seq).
func (s *Store) attemptsForIDs(ids []string) (map[string][]AttemptRow, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	// Build placeholders: ?, ?, ?
	placeholders := make([]string, len(ids))
	args := make([]any, len(ids))
	for i, id := range ids {
		placeholders[i] = "?"
		args[i] = id
	}
	q := fmt.Sprintf(
		`SELECT id, request_id, seq, provider, model, key_id, status, latency_ms, cost, error, error_origin
		 FROM attempts WHERE request_id IN (%s) ORDER BY request_id, seq`,
		strings.Join(placeholders, ","),
	)
	rows, err := s.db.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make(map[string][]AttemptRow, len(ids))
	for rows.Next() {
		var a AttemptRow
		if err := rows.Scan(&a.ID, &a.RequestID, &a.Seq, &a.Provider, &a.Model, &a.KeyID, &a.Status, &a.LatencyMs, &a.Cost, &a.Err, &a.ErrorOrigin); err != nil {
			return nil, err
		}
		out[a.RequestID] = append(out[a.RequestID], a)
	}
	return out, rows.Err()
}

// --- provider model cache ---

// CachedModel is one model entry stored for a provider.
type CachedModel struct {
	Provider string `json:"provider"`
	ModelID  string `json:"model_id"`
	Source   string `json:"source"` // "fetched" or "custom"
}

// SetProviderModels replaces the full model cache for a provider.
// This is called after a successful /v1/models fetch — it wipes the
// old fetched models and inserts the fresh set. Custom models are preserved.
func (s *Store) SetProviderModels(provider string, models []string) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	// only delete fetched models, keep custom ones
	if _, err := tx.Exec(`DELETE FROM provider_models WHERE provider = ? AND source = 'fetched'`, provider); err != nil {
		tx.Rollback()
		return err
	}
	for _, m := range models {
		if _, err := tx.Exec(`INSERT OR IGNORE INTO provider_models (provider, model_id, source) VALUES (?, ?, 'fetched')`, provider, m); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

// AddCustomModel adds a single custom model for a provider.
func (s *Store) AddCustomModel(provider, modelID string) error {
	_, err := s.db.Exec(`INSERT OR IGNORE INTO provider_models (provider, model_id, source) VALUES (?, ?, 'custom')`, provider, modelID)
	return err
}

// RemoveCachedModel removes a single model from a provider's cache (any source).
func (s *Store) RemoveCachedModel(provider, modelID string) error {
	_, err := s.db.Exec(`DELETE FROM provider_models WHERE provider = ? AND model_id = ?`, provider, modelID)
	return err
}

// RemoveCustomModel removes a custom model from a provider's cache.
// Fetched models are preserved — only source='custom' rows are deleted.
func (s *Store) RemoveCustomModel(provider, modelID string) error {
	_, err := s.db.Exec(`DELETE FROM provider_models WHERE provider = ? AND model_id = ? AND source = 'custom'`, provider, modelID)
	return err
}

// GetProviderModels returns all cached models for a provider (fetched + custom).
func (s *Store) GetProviderModels(provider string) ([]CachedModel, error) {
	rows, err := s.db.Query(`SELECT provider, model_id, source FROM provider_models WHERE provider = ? ORDER BY source DESC, model_id`, provider)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []CachedModel
	for rows.Next() {
		var m CachedModel
		if err := rows.Scan(&m.Provider, &m.ModelID, &m.Source); err != nil {
			continue
		}
		out = append(out, m)
	}
	return out, nil
}

// HasProviderModels reports whether a provider has any cached models at all.
func (s *Store) HasProviderModels(provider string) (bool, error) {
	var count int
	err := s.db.QueryRow(`SELECT COUNT(*) FROM provider_models WHERE provider = ?`, provider).Scan(&count)
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

// Prune deletes requests (and their attempt rows) older than before, to keep
// the event-log database from growing without bound. Returns the number of
// request rows removed. Requests with a NULL ts_unix (pre-migration rows) are
// left untouched by the time filter and can be cleaned up manually if desired.
func (s *Store) Prune(before time.Time) (int64, error) {
	epoch := before.Unix()
	if _, err := s.db.Exec(
		`DELETE FROM attempts WHERE request_id IN (SELECT id FROM requests WHERE ts_unix IS NOT NULL AND ts_unix < ?)`,
		epoch,
	); err != nil {
		return 0, err
	}
	res, err := s.db.Exec(`DELETE FROM requests WHERE ts_unix IS NOT NULL AND ts_unix < ?`, epoch)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return n, nil
}
