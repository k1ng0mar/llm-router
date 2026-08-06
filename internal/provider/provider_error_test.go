package provider

import "testing"

func TestIsProviderError(t *testing.T) {
	body := `{"error":{"message":"Streaming request failed","type":"provider_error","code":"provider_error","param":null}}`
	if !IsProviderError(400, body) {
		t.Fatal("provider_error body should be fallback-eligible")
	}
	if IsProviderError(401, `{"error":{"message":"bad key"}}`) {
		t.Fatal("plain 401 must NOT be fallback-eligible")
	}
	if IsProviderError(400, `{"error":{"message":"invalid parameters"}}`) {
		t.Fatal("plain 400 must NOT be fallback-eligible")
	}
	if IsProviderError(500, body) {
		t.Fatal("5xx is handled by Retryable, not IsProviderError")
	}
	// case-insensitive check
	if !IsProviderError(400, `{"error":{"message":"x","TYPE":"PROVIDER_ERROR"}}`) {
		t.Fatal("case-insensitive provider_error should match")
	}
}
