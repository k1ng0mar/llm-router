// Package classify decides which pool a request belongs to.
// Deterministic precedence: hint header -> last-user-turn heuristics -> image fallthrough -> default.
// No LLM ever runs in this path.
package classify

import (
	"sort"
	"strings"
)

// Heuristics maps pool name to keyword/pattern list.
type Heuristics map[string][]string

// Detect returns the first pool whose any keyword matches text.
func Detect(h Heuristics, text string) (string, bool) {
	pools := make([]string, 0, len(h))
	for p := range h {
		pools = append(pools, p)
	}
	sort.Strings(pools) // deterministic regardless of map iteration
	for _, pool := range pools {
		for _, kw := range h[pool] {
			if kw != "" && strings.Contains(text, kw) {
				return pool, true
			}
		}
	}
	return "", false
}

// agentWrapperTags is the set of markers that identify agent-injected preamble
// inside what otherwise looks like a user message. These are stripped before
// scoring so a system reminder containing "import " doesn't trigger code routing.
var agentWrapperTags = []string{
	"<system-reminder>", "</system-reminder>",
	"<env>", "</env>",
	"<function_calls>", "</function_calls>",
	"<arg_key>content</arg_key>",
}

// stripAgentWrappers removes agent-injected instruction blocks (including
// their content) from user messages so they can't trigger false heuristics.
// Manifest issue #2575 shows what happens when preamble is scored as user input.
func stripAgentWrappers(s string) string {
	for _, open := range agentWrapperTags {
		if strings.HasPrefix(open, "</") {
			continue
		}
		close := "</" + strings.TrimPrefix(open, "<")
		for {
			start := strings.Index(s, open)
			if start < 0 {
				break
			}
			end := strings.Index(s[start:], close)
			if end < 0 {
				break // no close — leave as-is
			}
			// remove the entire block [open...close]
			s = s[:start] + " " + s[start+end+len(close):]
		}
	}
	return s
}

// LastUserContent extracts the text content of the last user-role message,
// stripping agent-injected wrappers. For multi-part messages, text parts are
// concatenated and image_url parts set hasImage. Returns ("", false) if there
// is no user message.
func LastUserContent(msgs []any) (text string, hasImage bool, found bool) {
	for i := len(msgs) - 1; i >= 0; i-- {
		role, content, ok := msgRoleContent(msgs[i])
		if !ok || role != "user" {
			continue
		}
		text, hasImage = extractContent(content)
		text = stripAgentWrappers(text)
		return text, hasImage, true
	}
	return "", false, false
}

// msgRoleContent extracts role and content from a message object.
func msgRoleContent(m any) (role, content any, ok bool) {
	mm, ok := m.(map[string]any)
	if !ok {
		return "", nil, false
	}
	role, _ = mm["role"].(string)
	return role, mm["content"], true
}

// extractContent pulls text and image presence from a content value, which may
// be a string or an array of part objects.
func extractContent(content any) (text string, hasImage bool) {
	switch c := content.(type) {
	case string:
		return c, false
	case []any:
		var sb strings.Builder
		for _, part := range c {
			pm, ok := part.(map[string]any)
			if !ok {
				continue
			}
			pt, _ := pm["type"].(string)
			switch pt {
			case "text":
				if s, ok := pm["text"].(string); ok {
					sb.WriteString(s)
					sb.WriteString("\n")
				}
			case "image_url":
				if _, ok := pm["image_url"]; ok {
					hasImage = true
				}
			}
		}
		return sb.String(), hasImage
	}
	return "", false
}

// PoolForFull is the convenience entry: extracts last-user-content from messages
// and classifies. Returns pool, rule ("hint"|"code-heuristic"|"image"|"default"),
// and the scored text.
func PoolForFull(h Heuristics, msgs []any, hint, def string) (pool, rule, text string) {
	lastText, hasImage, ok := LastUserContent(msgs)
	if !ok || strings.TrimSpace(lastText) == "" {
		if hasImage {
			return def, "image", ""
		}
		return def, "default", ""
	}
	if hint != "" {
		return hint, "hint", lastText
	}
	if p, ok := Detect(h, lastText); ok {
		return p, p + "-heuristic", lastText
	}
	if hasImage {
		return def, "image", lastText
	}
	return def, "default", lastText
}

// PoolForFullMulti is like PoolForFull but returns hasImage separately so the
// caller can also use it for the vision chain decision.
func PoolForFullMulti(h Heuristics, msgs []any, def, hint string) (pool, rule, text string, hasImage bool) {
	lastText, img, ok := LastUserContent(msgs)
	if !ok {
		return def, "default", "", img
	}
	hasImage = img
	if hint != "" {
		return hint, "hint", lastText, hasImage
	}
	if p, ok := Detect(h, lastText); ok {
		return p, p + "-heuristic", lastText, hasImage
	}
	if hasImage {
		return def, "image", lastText, hasImage
	}
	return def, "default", lastText, hasImage
}
