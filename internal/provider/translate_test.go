package provider

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildAnthropicRequest(t *testing.T) {
	payload := map[string]any{
		"model":      "claude-3-5-sonnet",
		"max_tokens": float64(100),
		"stream":     true,
		"messages": []any{
			map[string]any{"role": "system", "content": "You are helpful."},
			map[string]any{"role": "user", "content": "Hi there"},
			map[string]any{"role": "assistant", "content": "Hello!"},
		},
	}
	body, model, err := buildAnthropicRequest(payload)
	if err != nil {
		t.Fatal(err)
	}
	if model != "claude-3-5-sonnet" {
		t.Fatalf("model = %q", model)
	}
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	// system hoisted, stream forced off, roles mapped
	if out["system"] != "You are helpful." {
		t.Fatalf("system = %v", out["system"])
	}
	if out["stream"] != false {
		t.Fatalf("stream not forced off: %v", out["stream"])
	}
	if out["max_tokens"] != float64(100) {
		t.Fatalf("max_tokens = %v", out["max_tokens"])
	}
	msgs := out["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages len = %d (system should be hoisted)", len(msgs))
	}
	first := msgs[0].(map[string]any)
	if first["role"] != "user" || first["content"] != "Hi there" {
		t.Fatalf("first msg = %v", first)
	}
}

func TestBuildGeminiRequest(t *testing.T) {
	payload := map[string]any{
		"model":      "gemini-pro",
		"max_tokens": float64(200),
		"messages": []any{
			map[string]any{"role": "system", "content": "Be brief."},
			map[string]any{"role": "user", "content": "What is 2+2?"},
			map[string]any{"role": "assistant", "content": "4"},
		},
	}
	body, model, err := buildGeminiRequest(payload)
	if err != nil {
		t.Fatal(err)
	}
	if model != "gemini-pro" {
		t.Fatalf("model = %q", model)
	}
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	contents := out["contents"].([]any)
	// system folded into a user turn + 2 real turns = 3
	if len(contents) != 3 {
		t.Fatalf("contents len = %d", len(contents))
	}
	// assistant role mapped to "model"
	last := contents[2].(map[string]any)
	if last["role"] != "model" {
		t.Fatalf("assistant role not mapped to model: %v", last["role"])
	}
	gc := out["generationConfig"].(map[string]any)
	if gc["maxOutputTokens"] != float64(200) {
		t.Fatalf("maxOutputTokens = %v", gc["maxOutputTokens"])
	}
}

func TestAnthropicToOpenAI(t *testing.T) {
	body := []byte(`{
		"id": "msg_123",
		"model": "claude-3-5-sonnet",
		"content": [{"type":"text","text":"Hello world"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`)
	completion := anthropicToOpenAI(body)
	if completion == nil {
		t.Fatal("nil completion")
	}
	choices := completion["choices"].([]map[string]any)
	if choices[0]["message"].(map[string]any)["content"] != "Hello world" {
		t.Fatalf("content = %v", choices[0]["message"])
	}
	if choices[0]["finish_reason"] != "stop" {
		t.Fatalf("finish_reason = %v", choices[0]["finish_reason"])
	}
	usage := completion["usage"].(map[string]any)
	if usage["prompt_tokens"] != 10 || usage["completion_tokens"] != 5 {
		t.Fatalf("usage = %v", usage)
	}
}

func TestGeminiToOpenAI(t *testing.T) {
	body := []byte(`{
		"candidates": [{"content":{"parts":[{"text":"42"}]},"finishReason":"STOP"}],
		"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3},
		"modelVersion": "gemini-2.0-flash"
	}`)
	completion := geminiToOpenAI(body)
	if completion == nil {
		t.Fatal("nil completion")
	}
	choices := completion["choices"].([]map[string]any)
	if choices[0]["message"].(map[string]any)["content"] != "42" {
		t.Fatalf("content = %v", choices[0]["message"])
	}
	if completion["model"] != "gemini-2.0-flash" {
		t.Fatalf("model = %v", completion["model"])
	}
	usage := completion["usage"].(map[string]any)
	if usage["prompt_tokens"] != 7 || usage["completion_tokens"] != 3 {
		t.Fatalf("usage = %v", usage)
	}
}

func TestTranslateNativeResponseStreaming(t *testing.T) {
	// streaming client + anthropic upstream → single SSE chunk
	body := []byte(`{"id":"m1","model":"c","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":1,"output_tokens":1}}`)
	orig := map[string]any{"stream": true}
	out, err := translateNativeResponse(ModeAnthropic, body, orig)
	if err != nil {
		t.Fatal(err)
	}
	s := string(out)
	if !strings.HasPrefix(s, "data: ") {
		t.Fatalf("not SSE: %q", s)
	}
	if !strings.Contains(s, "data: [DONE]") {
		t.Fatalf("missing DONE: %q", s)
	}
	// non-streaming → plain JSON
	orig2 := map[string]any{"stream": false}
	out2, err := translateNativeResponse(ModeAnthropic, body, orig2)
	if err != nil {
		t.Fatal(err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out2, &parsed); err != nil {
		t.Fatalf("non-stream output not JSON: %v", err)
	}
	if parsed["object"] != "chat.completion" {
		t.Fatalf("object = %v", parsed["object"])
	}
}

func TestOpenAIContentToText(t *testing.T) {
	if got := openAIContentToText("plain"); got != "plain" {
		t.Fatalf("got %q", got)
	}
	parts := []any{
		map[string]any{"type": "text", "text": "a"},
		map[string]any{"type": "text", "text": "b"},
	}
	if got := openAIContentToText(parts); got != "ab" {
		t.Fatalf("parts got %q", got)
	}
}

func TestNormalizeMode(t *testing.T) {
	if normalizeMode("") != ModeOpenAI {
		t.Fatal("empty should be openai")
	}
	if normalizeMode("ANTHROPIC") != ModeAnthropic {
		t.Fatal("case-insensitive anthropic")
	}
	if normalizeMode("gemini") != ModeGemini {
		t.Fatal("gemini")
	}
	if normalizeMode("bogus") != ModeOpenAI {
		t.Fatal("unknown should default openai")
	}
}
