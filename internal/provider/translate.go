// Package provider: non-OpenAI endpoint support.
//
// The router's core contract is OpenAI-shaped: it accepts /v1/chat/completions
// requests and passes upstream bytes straight through to the client. To reach
// providers that speak Anthropic or Gemini wire formats, we translate at the
// boundary inside Do(): the outgoing request is built in the upstream's native
// shape, and the 2xx response is converted back to OpenAI shape so the rest of
// the router (fallback, token capture, dashboard passthrough) is untouched.
//
// Scope for now: text-only, non-streaming. When the client asks for streaming
// against a non-OpenAI provider, we force the upstream to non-stream and wrap
// the translated completion in a single OpenAI SSE chunk so the client still
// gets a valid stream. Tool calls and multimodal content are not yet mapped.
package provider

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// APIMode identifies the upstream wire format. Empty/unknown defaults to openai.
const (
	ModeOpenAI    = "openai"
	ModeAnthropic = "anthropic"
	ModeGemini    = "gemini"
)

// normalizeMode coerces a configured api_mode string to a known value.
func normalizeMode(m string) string {
	switch strings.ToLower(strings.TrimSpace(m)) {
	case ModeAnthropic:
		return ModeAnthropic
	case ModeGemini:
		return ModeGemini
	default:
		return ModeOpenAI
	}
}

// buildNativeRequest converts the OpenAI-shaped payload into the upstream's
// native request body. Returns the body bytes and the model id (Gemini embeds
// the model in the URL rather than the body).
func buildNativeRequest(mode string, payload map[string]any) (body []byte, model string, err error) {
	switch normalizeMode(mode) {
	case ModeAnthropic:
		return buildAnthropicRequest(payload)
	case ModeGemini:
		return buildGeminiRequest(payload)
	default:
		// OpenAI: pass the payload through unchanged (existing behaviour).
		b, err := json.Marshal(payload)
		return b, "", err
	}
}

// nativeEndpoint returns the request path (relative to base_url) for a mode.
func nativeEndpoint(mode string) string {
	switch normalizeMode(mode) {
	case ModeAnthropic:
		return "/v1/messages"
	case ModeGemini:
		return "/v1beta/models"
	default:
		return "/v1/chat/completions"
	}
}

// buildAnthropicRequest maps OpenAI messages to the Anthropic Messages API.
// system messages are hoisted into the top-level "system" field; the rest map
// to user/assistant turns with plain-text content. stream is forced off.
func buildAnthropicRequest(payload map[string]any) ([]byte, string, error) {
	model, _ := payload["model"].(string)
	msgs, _ := payload["messages"].([]any)

	var system strings.Builder
	anthropicMsgs := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		role, _ := mm["role"].(string)
		content := openAIContentToText(mm["content"])
		if role == "system" {
			if system.Len() > 0 {
				system.WriteString("\n\n")
			}
			system.WriteString(content)
			continue
		}
		// Anthropic only allows user/assistant roles.
		if role != "user" && role != "assistant" {
			role = "user"
		}
		anthropicMsgs = append(anthropicMsgs, map[string]any{
			"role":    role,
			"content": content,
		})
	}

	out := map[string]any{
		"model":    model,
		"messages": anthropicMsgs,
		"stream":   false,
	}
	if system.Len() > 0 {
		out["system"] = system.String()
	}
	if mt, ok := payload["max_tokens"].(float64); ok && int(mt) > 0 {
		out["max_tokens"] = int(mt)
	} else {
		// Anthropic requires max_tokens; default generously.
		out["max_tokens"] = 4096
	}
	if temp, ok := payload["temperature"].(float64); ok {
		out["temperature"] = temp
	}

	b, err := json.Marshal(out)
	return b, model, err
}

// buildGeminiRequest maps OpenAI messages to the Gemini generateContent shape.
// The model goes in the URL, so it's returned separately. stream is forced off.
func buildGeminiRequest(payload map[string]any) ([]byte, string, error) {
	model, _ := payload["model"].(string)
	msgs, _ := payload["messages"].([]any)

	contents := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		mm, ok := m.(map[string]any)
		if !ok {
			continue
		}
		role, _ := mm["role"].(string)
		if role == "system" {
			// Gemini has no system role; fold into a user turn with a marker.
			contents = append(contents, map[string]any{
				"role":  "user",
				"parts": []map[string]any{{"text": "System instruction: " + openAIContentToText(mm["content"])}},
			})
			continue
		}
		// Gemini roles: user / model.
		if role == "assistant" {
			role = "model"
		} else {
			role = "user"
		}
		contents = append(contents, map[string]any{
			"role":  role,
			"parts": []map[string]any{{"text": openAIContentToText(mm["content"])}},
		})
	}

	genCfg := map[string]any{}
	if mt, ok := payload["max_tokens"].(float64); ok && int(mt) > 0 {
		genCfg["maxOutputTokens"] = int(mt)
	}
	if temp, ok := payload["temperature"].(float64); ok {
		genCfg["temperature"] = temp
	}
	out := map[string]any{
		"contents":         contents,
		"generationConfig": genCfg,
	}

	b, err := json.Marshal(out)
	return b, model, err
}

// openAIContentToText flattens an OpenAI message content (string or part array)
// to plain text for Anthropic/Gemini.
func openAIContentToText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var sb strings.Builder
		for _, part := range c {
			if p, ok := part.(map[string]any); ok {
				if t, ok := p["text"].(string); ok {
					sb.WriteString(t)
				}
			}
		}
		return sb.String()
	default:
		return ""
	}
}

// translateNativeResponse converts a 2xx non-OpenAI upstream body into OpenAI
// shape. If the original client requested streaming, the result is wrapped as
// a single OpenAI SSE chunk; otherwise it's plain OpenAI chat.completion JSON.
func translateNativeResponse(mode string, body []byte, origPayload map[string]any) ([]byte, error) {
	var completion map[string]any
	switch normalizeMode(mode) {
	case ModeAnthropic:
		completion = anthropicToOpenAI(body)
	case ModeGemini:
		completion = geminiToOpenAI(body)
	default:
		return body, nil
	}
	if completion == nil {
		return nil, fmt.Errorf("failed to parse %s response", mode)
	}
	jsonBody, err := json.Marshal(completion)
	if err != nil {
		return nil, err
	}
	streaming, _ := origPayload["stream"].(bool)
	if !streaming {
		return jsonBody, nil
	}
	return wrapAsSSE(jsonBody), nil
}

// anthropicToOpenAI maps an Anthropic Messages response to OpenAI shape.
func anthropicToOpenAI(body []byte) map[string]any {
	var in struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &in); err != nil {
		return nil
	}
	var text strings.Builder
	for _, c := range in.Content {
		if c.Type == "text" {
			text.WriteString(c.Text)
		}
	}
	return map[string]any{
		"id":      in.ID,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   in.Model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": text.String()},
			"finish_reason": anthropicStopReason(in.StopReason),
		}},
		"usage": map[string]any{
			"prompt_tokens":     in.Usage.InputTokens,
			"completion_tokens": in.Usage.OutputTokens,
			"total_tokens":      in.Usage.InputTokens + in.Usage.OutputTokens,
		},
	}
}

func anthropicStopReason(r string) string {
	switch r {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	default:
		return "stop"
	}
}

// geminiToOpenAI maps a Gemini generateContent response to OpenAI shape.
func geminiToOpenAI(body []byte) map[string]any {
	var in struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
		} `json:"usageMetadata"`
		ModelVersion string `json:"modelVersion"`
	}
	if err := json.Unmarshal(body, &in); err != nil {
		return nil
	}
	var text strings.Builder
	for _, c := range in.Candidates {
		for _, p := range c.Content.Parts {
			text.WriteString(p.Text)
		}
	}
	model := in.ModelVersion
	if model == "" {
		model = "gemini"
	}
	return map[string]any{
		"id":      "gemini-" + fmt.Sprint(time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": text.String()},
			"finish_reason": geminiStopReason(in.Candidates),
		}},
		"usage": map[string]any{
			"prompt_tokens":     in.UsageMetadata.PromptTokenCount,
			"completion_tokens": in.UsageMetadata.CandidatesTokenCount,
			"total_tokens":      in.UsageMetadata.PromptTokenCount + in.UsageMetadata.CandidatesTokenCount,
		},
	}
}

func geminiStopReason(cands []struct {
	Content      struct {
		Parts []struct {
			Text string `json:"text"`
		} `json:"parts"`
	} `json:"content"`
	FinishReason string `json:"finishReason"`
}) string {
	if len(cands) == 0 {
		return "stop"
	}
	switch cands[0].FinishReason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION", "OTHER":
		return "content_filter"
	default:
		return "stop"
	}
}

// wrapAsSSE wraps a single OpenAI JSON completion as one SSE data chunk plus
// the [DONE] terminator, so clients that requested streaming get a valid stream.
func wrapAsSSE(jsonBody []byte) []byte {
	var sb bytes.Buffer
	sb.WriteString("data: ")
	sb.Write(jsonBody)
	sb.WriteString("\n\ndata: [DONE]\n\n")
	return sb.Bytes()
}

// syntheticResponse builds an *http.Response whose body is the translated
// OpenAI bytes, so the router's passthrough path sees a normal 200 response.
func syntheticResponse(body []byte, streaming bool) *http.Response {
	ct := "application/json"
	if streaming {
		ct = "text/event-stream"
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Status:     "200 OK",
		Header: http.Header{
			"Content-Type": []string{ct},
		},
		Body:          io.NopCloser(bytes.NewReader(body)),
		ContentLength: int64(len(body)),
	}
}
