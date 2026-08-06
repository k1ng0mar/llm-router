package route

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestDescribeAttemptError(t *testing.T) {
	// parent canceled → client-went-away message
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got := describeAttemptError(errors.New(`Post "http://x/chat/completions": context canceled`), 90*time.Second, ctx)
	if !strings.Contains(got, "client disconnected") {
		t.Fatalf("parent-cancel should say client disconnected, got: %s", got)
	}

	// attempt deadline exceeded → timeout message
	got2 := describeAttemptError(context.DeadlineExceeded, 90*time.Second, context.Background())
	if !strings.Contains(got2, "timed out after 1m30s") {
		t.Fatalf("deadline should say timed out, got: %s", got2)
	}

	// plain network error with healthy parent → raw error
	got3 := describeAttemptError(errors.New("dial tcp: connection refused"), 90*time.Second, context.Background())
	if !strings.Contains(got3, "connection refused") {
		t.Fatalf("network error should pass through, got: %s", got3)
	}
}
