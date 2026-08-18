package store

import (
	"path/filepath"
	"testing"
	"time"
)

func TestWriteAndReadBack(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer s.Close()

	req := RequestRow{ID: "r1", TS: time.Now().Format(time.RFC3339), Pool: "code", Rule: "default", FinalStatus: 200, FinalProvider: "p1", FinalModel: "m1", TotalMs: 12}
	if err := s.AddRequest(&req); err != nil {
		t.Fatalf("AddRequest: %v", err)
	}
	atts := []AttemptRow{
		{RequestID: "r1", Seq: 1, Provider: "p1", Model: "m1", KeyID: "k1", Status: 429, LatencyMs: 4, Err: "rate limited"},
		{RequestID: "r1", Seq: 2, Provider: "p1", Model: "m1", KeyID: "k2", Status: 200, LatencyMs: 8},
	}
	for i := range atts {
		if err := s.AddAttempt(&atts[i]); err != nil {
			t.Fatalf("AddAttempt %d: %v", i, err)
		}
	}

	rows, err := s.ListRequests(Filter{Limit: 10}, 0)
	if err != nil {
		t.Fatalf("ListRequests: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 request, got %d", len(rows))
	}
	if rows[0].Pool != "code" || len(rows[0].Attempts) != 2 {
		t.Fatalf("request shape wrong: %+v", rows[0])
	}
	if rows[0].Attempts[0].Err != "rate limited" {
		t.Fatalf("attempt error not persisted: %+v", rows[0].Attempts[0])
	}
}

func TestListRequestsBodiesStrippedAndGetRequest(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer s.Close()

	req := RequestRow{ID: "b1", TS: time.Now().Format(time.RFC3339), Pool: "chat", Rule: "default", FinalStatus: 200, FinalProvider: "p1", FinalModel: "m1", RequestBody: "the-request", ResponseBody: "the-response"}
	if err := s.AddRequest(&req); err != nil {
		t.Fatalf("AddRequest: %v", err)
	}
	if err := s.AddAttempt(&AttemptRow{RequestID: "b1", Seq: 1, Provider: "p1", Model: "m1", KeyID: "k1", Status: 200}); err != nil {
		t.Fatalf("AddAttempt: %v", err)
	}

	// withBodies == 0 (the dashboard list) must not drag bodies across the wire.
	rows, err := s.ListRequests(Filter{Limit: 10}, 0)
	if err != nil {
		t.Fatalf("ListRequests: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 request, got %d", len(rows))
	}
	if rows[0].RequestBody != "" || rows[0].ResponseBody != "" {
		t.Fatalf("bodies should be stripped without withBodies, got req=%q resp=%q", rows[0].RequestBody, rows[0].ResponseBody)
	}

	// withBodies != 0 loads them, with attempts still attached.
	rows, err = s.ListRequests(Filter{RequestID: "b1", Limit: 10}, 1)
	if err != nil {
		t.Fatalf("ListRequests(withBodies): %v", err)
	}
	if len(rows) != 1 || len(rows[0].Attempts) != 1 {
		t.Fatalf("expected 1 request with 1 attempt, got %+v", rows)
	}
	if rows[0].RequestBody != "the-request" || rows[0].ResponseBody != "the-response" {
		t.Fatalf("withBodies should load bodies, got req=%q resp=%q", rows[0].RequestBody, rows[0].ResponseBody)
	}

	// GetRequest is the single-row counterpart with bodies included.
	got, err := s.GetRequest("b1")
	if err != nil {
		t.Fatalf("GetRequest: %v", err)
	}
	if got == nil {
		t.Fatal("GetRequest returned nil for existing id")
	}
	if got.RequestBody != "the-request" || got.ResponseBody != "the-response" || len(got.Attempts) != 1 {
		t.Fatalf("GetRequest shape wrong: %+v", got)
	}

	// GetRequest on a missing id must return nil, not an error.
	miss, err := s.GetRequest("nope")
	if err != nil {
		t.Fatalf("GetRequest missing: %v", err)
	}
	if miss != nil {
		t.Fatalf("GetRequest missing should be nil, got %+v", miss)
	}
}

func TestFilterByStatus(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer s.Close()

	now := time.Now()
	for i, status := range []int{200, 503, 200} {
		id := string(rune('a' + i))
		s.AddRequest(&RequestRow{ID: id, TS: now.Add(-time.Duration(i) * time.Second).Format(time.RFC3339), Pool: "chat", Rule: "default", FinalStatus: status})
	}

	rows, err := s.ListRequests(Filter{Status: 200, Limit: 10}, 0)
	if err != nil {
		t.Fatalf("ListRequests: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("status filter should return 2, got %d", len(rows))
	}

	rows, err = s.ListRequests(Filter{Pool: "chat", Limit: 10}, 0)
	if err != nil {
		t.Fatalf("ListRequests: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("pool filter should return 3, got %d", len(rows))
	}
}

func TestWALEnabled(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer s.Close()
	mode := ""
	s.db.QueryRow("PRAGMA journal_mode").Scan(&mode)
	if mode != "wal" {
		t.Fatalf("journal_mode = %q, want wal", mode)
	}
}
