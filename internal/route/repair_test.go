package route

import (
	"reflect"
	"testing"
)

func assistantToolCall() map[string]any {
	return map[string]any{
		"role":       "assistant",
		"content":    nil,
		"tool_calls": []any{map[string]any{"id": "call_1", "type": "function"}},
	}
}

func TestRepairReasoningContentFillsAssistantToolCalls(t *testing.T) {
	payload := map[string]any{
		"model": "deepseek/deepseek-v4-pro",
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			assistantToolCall(),
			map[string]any{"role": "tool", "content": "42"},
		},
	}
	out, changed := repairReasoningContent(payload)
	if !changed {
		t.Fatal("expected the assistant tool-call turn to be repaired")
	}
	msgs := out["messages"].([]any)
	got := msgs[1].(map[string]any)["reasoning_content"]
	if got != reasoningPlaceholder {
		t.Fatalf("reasoning_content = %v, want %q", got, reasoningPlaceholder)
	}
	// The untouched turns must survive verbatim.
	if msgs[0].(map[string]any)["content"] != "hi" || msgs[2].(map[string]any)["content"] != "42" {
		t.Fatalf("neighbouring messages altered: %v", msgs)
	}
}

// The payload is shared by every candidate in the fallback chain, so a repair
// made for one provider must not follow the request to the next one.
func TestRepairReasoningContentDoesNotMutateCaller(t *testing.T) {
	original := assistantToolCall()
	payload := map[string]any{"messages": []any{original}}
	before := reflect.DeepEqual(original, assistantToolCall())

	out, changed := repairReasoningContent(payload)
	if !changed {
		t.Fatal("expected a repair")
	}
	if !before || !reflect.DeepEqual(original, assistantToolCall()) {
		t.Fatalf("caller's message was mutated in place: %v", original)
	}
	if _, leaked := payload["messages"].([]any)[0].(map[string]any)["reasoning_content"]; leaked {
		t.Fatal("repair leaked into the caller's payload")
	}
	if &out == &payload {
		t.Fatal("expected a fresh payload map")
	}
}

func TestRepairReasoningContentLeavesGoodPayloadsAlone(t *testing.T) {
	cases := map[string]map[string]any{
		"already has reasoning_content": {"messages": []any{map[string]any{
			"role": "assistant", "tool_calls": []any{map[string]any{"id": "c"}},
			"reasoning_content": "thought about it",
		}}},
		"assistant without tool calls": {"messages": []any{map[string]any{
			"role": "assistant", "content": "plain answer",
		}}},
		"empty tool_calls": {"messages": []any{map[string]any{
			"role": "assistant", "tool_calls": []any{},
		}}},
		"user turn":       {"messages": []any{map[string]any{"role": "user", "content": "hi"}}},
		"no messages key": {"model": "x"},
	}
	for name, payload := range cases {
		t.Run(name, func(t *testing.T) {
			if _, changed := repairReasoningContent(payload); changed {
				t.Fatalf("payload was rewritten but should not have been: %v", payload)
			}
		})
	}
}

// A blank reasoning_content fails the same upstream check as a missing one.
func TestRepairReasoningContentTreatsBlankAsMissing(t *testing.T) {
	for _, blank := range []any{"", "   ", nil} {
		msg := assistantToolCall()
		msg["reasoning_content"] = blank
		out, changed := repairReasoningContent(map[string]any{"messages": []any{msg}})
		if !changed {
			t.Fatalf("reasoning_content %#v should count as missing", blank)
		}
		got := out["messages"].([]any)[0].(map[string]any)["reasoning_content"]
		if got != reasoningPlaceholder {
			t.Fatalf("reasoning_content = %v, want %q", got, reasoningPlaceholder)
		}
	}
}
