// Package classify decides which pool a request belongs to.
// Deterministic precedence: hint header -> last-user-turn heuristics -> media
// fallthrough -> default. No LLM ever runs in this path.
//
// "Media" covers all three non-text modalities — images, audio, and video.
// Detection spans the whole message array, not just the last turn, because the
// capability gate has to account for every modality the upstream will actually
// receive: an image three turns back still has to be decodable by whichever
// model answers now.
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

// Media records which non-text modalities a request carries. It is the input to
// both media-pool routing and the capability gate.
type Media struct {
	Image bool
	Audio bool
	Video bool
}

// Any reports whether the request carries any non-text modality at all.
func (m Media) Any() bool { return m.Image || m.Audio || m.Video }

// Names lists the modalities present, in a stable order, for log and error text.
func (m Media) Names() []string {
	var out []string
	if m.Image {
		out = append(out, "image")
	}
	if m.Audio {
		out = append(out, "audio")
	}
	if m.Video {
		out = append(out, "video")
	}
	return out
}

// mediaPartTypes maps an OpenAI content-part type to the modality it carries.
// Both the `image_url`/`input_audio`/`video_url` spellings and the bare
// `image`/`audio`/`video` forms some clients emit are recognized.
var mediaPartTypes = map[string]string{
	"image_url":   "image",
	"image":       "image",
	"input_audio": "audio",
	"audio":       "audio",
	"audio_url":   "audio",
	"video_url":   "video",
	"video":       "video",
	"input_video": "video",
}

// DetectMedia scans every message in the array and reports which modalities
// appear. Unlike LastUserContent (which scores only the last user turn for
// keyword routing), this deliberately spans the whole conversation: the gate
// must exclude a model that cannot decode media sitting anywhere in the history
// it is about to be sent.
func DetectMedia(msgs []any) Media {
	var m Media
	for _, msg := range msgs {
		mm, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		parts, ok := mm["content"].([]any)
		if !ok {
			continue
		}
		for _, part := range parts {
			pm, ok := part.(map[string]any)
			if !ok {
				continue
			}
			pt, _ := pm["type"].(string)
			switch mediaPartTypes[pt] {
			case "image":
				m.Image = true
			case "audio":
				m.Audio = true
			case "video":
				m.Video = true
			}
		}
	}
	return m
}

// PoolForMedia is the router's entry point. It applies the deterministic
// precedence — hint, then keyword heuristics on the last user turn, then media
// fallthrough, then default — and reports the media present anywhere in the
// conversation.
//
// mediaPool is the pool that serves media-carrying requests; pass "" to keep the
// pre-media behavior of sending them to def. Media ranks *below* heuristics on
// purpose: "explain this code" plus a screenshot still belongs in the code pool,
// where the vision describe-hop can turn the pixels into text for a code model.
func PoolForMedia(h Heuristics, msgs []any, def, hint, mediaPool string) (pool, rule, text string, media Media) {
	media = DetectMedia(msgs)
	lastText, _, ok := LastUserContent(msgs)
	if hint != "" {
		return hint, "hint", lastText, media
	}
	if ok && strings.TrimSpace(lastText) != "" {
		if p, found := Detect(h, lastText); found {
			return p, p + "-heuristic", lastText, media
		}
	}
	if media.Any() {
		if mediaPool != "" {
			return mediaPool, "media", lastText, media
		}
		return def, "media", lastText, media
	}
	return def, "default", lastText, media
}

// StripMedia returns a copy of msgs with every media part removed and each
// message's remaining text parts flattened into a plain string. Roles, order and
// text all survive; only the pixels (or audio, or video) are dropped. A message
// that held nothing but media is omitted entirely, since an empty content string
// is something several upstreams reject outright.
//
// This is what lets a describe-first hop hand a text-only model the whole
// conversation rather than just the description of the attachment.
func StripMedia(msgs []any) []any {
	out := make([]any, 0, len(msgs))
	for _, msg := range msgs {
		mm, ok := msg.(map[string]any)
		if !ok {
			continue
		}
		copied := make(map[string]any, len(mm))
		for k, v := range mm {
			copied[k] = v
		}
		switch c := mm["content"].(type) {
		case string:
			copied["content"] = c
		case []any:
			var sb strings.Builder
			for _, part := range c {
				pm, ok := part.(map[string]any)
				if !ok {
					continue
				}
				pt, _ := pm["type"].(string)
				if _, isMedia := mediaPartTypes[pt]; isMedia {
					continue
				}
				if pt == "text" {
					if s, ok := pm["text"].(string); ok && s != "" {
						if sb.Len() > 0 {
							sb.WriteString("\n")
						}
						sb.WriteString(s)
					}
				}
			}
			if sb.Len() == 0 {
				continue // media-only turn: nothing left to say
			}
			copied["content"] = sb.String()
		default:
			continue
		}
		out = append(out, copied)
	}
	return out
}

// AppendToLastUser folds extra into the text of the last user message, so added
// context arrives attached to the turn it describes instead of as a trailing
// message of its own — two consecutive user messages are rejected by some
// upstreams. If there is no user message, extra becomes one.
func AppendToLastUser(msgs []any, extra string) []any {
	out := append([]any(nil), msgs...)
	for i := len(out) - 1; i >= 0; i-- {
		mm, ok := out[i].(map[string]any)
		if !ok {
			continue
		}
		if role, _ := mm["role"].(string); role != "user" {
			continue
		}
		copied := make(map[string]any, len(mm))
		for k, v := range mm {
			copied[k] = v
		}
		text, _ := mm["content"].(string)
		copied["content"] = text + extra
		out[i] = copied
		return out
	}
	return append(out, map[string]any{"role": "user", "content": strings.TrimSpace(extra)})
}
