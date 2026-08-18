package route

import "strings"

// reasoningPlaceholder is what gets injected into an assistant message that
// made tool calls but arrived without reasoning_content. The gateways that
// demand the field only check that it is present and non-empty — they do not
// replay it to the model as thinking — so the cheapest sound value wins.
const reasoningPlaceholder = "(omitted)"

// repairReasoningContent injects reasoningPlaceholder into every assistant
// message that carries tool_calls but no non-empty reasoning_content, and
// reports whether it changed anything.
//
// DeepSeek-family gateways (TokenRouter is the one we hit) answer 400
// "messages[N].reasoning_content is required for thinking tool-call history"
// when a conversation replays an assistant tool-call turn without the
// reasoning_content the model originally produced. OpenAI-compatible clients
// drop that field when they build the follow-up request — it is not part of the
// OpenAI schema — so every multi-turn tool conversation dies on the second hop.
//
// Only what changes is copied: the messages slice and the individual message
// maps being rewritten. The caller's payload is shared across every candidate in
// the fallback chain, so mutating it in place would leak this provider's repair
// into the request sent to the next one.
func repairReasoningContent(payload map[string]any) (map[string]any, bool) {
	msgs, ok := payload["messages"].([]any)
	if !ok {
		return payload, false
	}
	var repaired []any
	for i, m := range msgs {
		msg, ok := m.(map[string]any)
		if !ok || !needsReasoningRepair(msg) {
			continue
		}
		if repaired == nil {
			repaired = make([]any, len(msgs))
			copy(repaired, msgs)
		}
		cp := make(map[string]any, len(msg)+1)
		for k, v := range msg {
			cp[k] = v
		}
		cp["reasoning_content"] = reasoningPlaceholder
		repaired[i] = cp
	}
	if repaired == nil {
		return payload, false
	}
	out := make(map[string]any, len(payload))
	for k, v := range payload {
		out[k] = v
	}
	out["messages"] = repaired
	return out, true
}

// needsReasoningRepair reports whether msg is an assistant tool-call turn that
// is missing the reasoning_content such gateways insist on.
func needsReasoningRepair(msg map[string]any) bool {
	if role, _ := msg["role"].(string); role != "assistant" {
		return false
	}
	calls, ok := msg["tool_calls"].([]any)
	if !ok || len(calls) == 0 {
		return false
	}
	switch rc := msg["reasoning_content"].(type) {
	case string:
		return strings.TrimSpace(rc) == ""
	default:
		// Absent, null, or some non-string shape the upstream won't accept.
		return true
	}
}
