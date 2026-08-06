package store

import (
	"database/sql"
	"testing"
)

func TestAddAttemptWithErrorOrigin(t *testing.T) {
	db := newTestDB(t)
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	// verify the column exists
	var origin string
	_ = db.QueryRow(`SELECT error_origin FROM attempts WHERE 1=0`).Scan(&origin)

	// old-style insert (no error_origin) should work via AddAttempt
	a := &AttemptRow{RequestID: "r1", Seq: 1, Provider: "p", Model: "m1", Status: 429, Err: "rate limited"}
	if err := s.AddAttempt(a); err != nil {
		t.Fatalf("AddAttempt: %v", err)
	}
	// error_origin defaults to "upstream" since we set Err
	row := db.QueryRow(`SELECT error_origin, error FROM attempts WHERE request_id=? AND seq=1`, "r1")
	var gotOrigin, gotErr string
	if err := row.Scan(&gotOrigin, &gotErr); err != nil {
		t.Fatalf("scan: %v", err)
	}
	if gotOrigin != "upstream" {
		t.Fatalf("origin = %q, want 'upstream'", gotOrigin)
	}
	if gotErr != "rate limited" {
		t.Fatalf("err = %q", gotErr)
	}
}

func TestRouterOriginNoUpstreamError(t *testing.T) {
	db := newTestDB(t)
	s := &Store{db: db}
	_ = s.migrate()
	// no attempt at all (e.g. capability gate refused) → request row has origin "router"
	r := &RequestRow{ID: "r2", TS: "2026-01-01T00:00:00Z", Pool: "code", Rule: "hint", FinalStatus: 400}
	if err := s.AddRequest(r); err != nil {
		t.Fatalf("AddRequest: %v", err)
	}
	var origin string
	err := db.QueryRow(`SELECT error_origin FROM requests WHERE id=?`, "r2").Scan(&origin)
	if err != nil {
		// column should exist after migrate
		t.Fatalf("scan origin: %v", err)
	}
	if origin != "router" {
		t.Fatalf("origin = %q, want 'router'", origin)
	}
}

func newTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", "file::memory:?cache=shared&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}
