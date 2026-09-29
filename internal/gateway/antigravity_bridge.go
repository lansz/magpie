package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/settings"
)

// bridgeRequest wraps Request with protocol translation and client profile metadata.
type bridgeRequest struct {
	*Request
	sourceProto   provider.Protocol
	clientProfile string
}

// newBridgeRequest returns a bridgeRequest wrapping the given Request with source protocol and client profile.
func newBridgeRequest(req *Request, proto provider.Protocol, profile string) *bridgeRequest {
	if profile == "" {
		profile = "unknown"
	}
	return &bridgeRequest{
		Request:       req,
		sourceProto:   proto,
		clientProfile: profile,
	}
}

// validateAntigravityTokenLimit validates output token limit fields for Antigravity target requests.
// It inspects raw field presence before parse to prevent codecs from conflating explicit null or 0 with absent fields.
func validateAntigravityTokenLimit(from provider.Protocol, body []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return err
	}
	checkField := func(field string) error {
		val, exists := raw[field]
		if !exists {
			return nil
		}
		trimmed := bytes.TrimSpace(val)
		if bytes.Equal(trimmed, []byte("null")) {
			return fmt.Errorf("%s must be a positive integer, got null", field)
		}
		var num int
		if err := json.Unmarshal(trimmed, &num); err != nil {
			return fmt.Errorf("%s must be a positive integer: %v", field, err)
		}
		if num <= 0 {
			return fmt.Errorf("%s must be greater than 0, got %d", field, num)
		}
		return nil
	}

	switch from {
	case provider.Anthropic:
		return checkField("max_tokens")
	case provider.Chat:
		if err := checkField("max_completion_tokens"); err != nil {
			return err
		}
		return checkField("max_tokens")
	case provider.Responses:
		return checkField("max_output_tokens")
	}
	return nil
}

// checkJsonObjectBytes checks whether b represents a valid JSON object without trailing data.
func checkJsonObjectBytes(b []byte, path string) error {
	trimmed := bytes.TrimSpace(b)
	if len(trimmed) == 0 || trimmed[0] != '{' || !json.Valid(trimmed) {
		return fmt.Errorf("%s must be a valid JSON object", path)
	}
	return nil
}

// checkStringArguments checks whether raw is a JSON string representing a valid JSON object.
func checkStringArguments(raw json.RawMessage, path string) error {
	if len(raw) == 0 {
		return fmt.Errorf("%s is required", path)
	}
	trimmed := bytes.TrimSpace(raw)
	if bytes.Equal(trimmed, []byte("null")) {
		return fmt.Errorf("%s cannot be null", path)
	}
	var s string
	if err := json.Unmarshal(trimmed, &s); err != nil {
		return fmt.Errorf("%s must be a JSON string", path)
	}
	trimmedStr := strings.TrimSpace(s)
	if trimmedStr == "" || trimmedStr == "null" {
		return fmt.Errorf("%s must be a non-empty JSON object string", path)
	}
	return checkJsonObjectBytes([]byte(trimmedStr), path)
}

// validateAntigravityToolArgs validates tool call arguments for Antigravity target requests before parse.
// Reason: Validating tool call arguments before parse prevents parseArgs from masking broken JSON
// into {"input": ...} and prevents argsOf from defaulting empty arguments to "{}", preserving exact
// business keys, numerical precision, and error positions for Antigravity requests.
func validateAntigravityToolArgs(from provider.Protocol, body []byte) error {
	switch from {
	case provider.Anthropic:
		var req struct {
			Messages []struct {
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			return err
		}
		for i, m := range req.Messages {
			if len(m.Content) == 0 {
				continue
			}
			trimmed := bytes.TrimSpace(m.Content)
			if bytes.HasPrefix(trimmed, []byte("[")) {
				var blocks []struct {
					Type  string          `json:"type"`
					Input json.RawMessage `json:"input"`
				}
				if err := json.Unmarshal(trimmed, &blocks); err != nil {
					return err
				}
				for j, b := range blocks {
					if b.Type == "tool_use" {
						path := fmt.Sprintf("messages[%d].content[%d].input", i, j)
						if len(b.Input) == 0 {
							return fmt.Errorf("%s is required", path)
						}
						trimmedInput := bytes.TrimSpace(b.Input)
						if bytes.Equal(trimmedInput, []byte("null")) {
							return fmt.Errorf("%s cannot be null", path)
						}
						if err := checkJsonObjectBytes(trimmedInput, path); err != nil {
							return err
						}
					}
				}
			}
		}
	case provider.Chat:
		var req struct {
			Messages []struct {
				Role      string `json:"role"`
				ToolCalls []struct {
					Function map[string]json.RawMessage `json:"function"`
				} `json:"tool_calls"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			return err
		}
		for i, m := range req.Messages {
			if m.Role != "assistant" {
				continue
			}
			for j, tc := range m.ToolCalls {
				path := fmt.Sprintf("messages[%d].tool_calls[%d].function.arguments", i, j)
				argRaw, exists := tc.Function["arguments"]
				if !exists {
					return fmt.Errorf("%s is required", path)
				}
				if err := checkStringArguments(argRaw, path); err != nil {
					return err
				}
			}
		}
	case provider.Responses:
		var req struct {
			Input json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			return err
		}
		if len(req.Input) > 0 {
			trimmed := bytes.TrimSpace(req.Input)
			if bytes.HasPrefix(trimmed, []byte("[")) {
				var items []struct {
					Type      string          `json:"type"`
					Arguments json.RawMessage `json:"arguments"`
				}
				if err := json.Unmarshal(trimmed, &items); err != nil {
					return err
				}
				for i, it := range items {
					if it.Type == "function_call" {
						path := fmt.Sprintf("input[%d].arguments", i)
						if err := checkStringArguments(it.Arguments, path); err != nil {
							return err
						}
					}
				}
			}
		}
	}
	return nil
}

// antigravityParallelBan is the error for a request that forbids parallel tool
// calls, or "" when it doesn't.
// Reason: Antigravity has no field for the ban, and dropping it would let the
// model call several tools at once against the caller's explicit wish.
func antigravityParallelBan(from provider.Protocol, r *Request) string {
	if r.Parallel == nil || *r.Parallel {
		return ""
	}
	field := "parallel_tool_calls"
	if from == provider.Anthropic {
		field = "tool_choice.disable_parallel_tool_use"
	}
	return field + ": Antigravity cannot forbid parallel tool calls"
}

const (
	AntigravityModeOff      = "off"
	AntigravityModeVerified = "verified"
	AntigravityModeStrict   = "strict"
)

var (
	antigravityModeMu      sync.RWMutex
	currentAntigravityMode = ""
)

func SetAntigravityMode(mode string) {
	antigravityModeMu.Lock()
	defer antigravityModeMu.Unlock()
	switch strings.ToLower(mode) {
	case "verified":
		currentAntigravityMode = AntigravityModeVerified
	case "strict":
		currentAntigravityMode = AntigravityModeStrict
	case "off":
		currentAntigravityMode = AntigravityModeOff
	default:
		currentAntigravityMode = AntigravityModeVerified
	}
}

func GetAntigravityMode() string {
	antigravityModeMu.RLock()
	defer antigravityModeMu.RUnlock()
	if currentAntigravityMode != "" {
		return currentAntigravityMode
	}
	switch strings.ToLower(settings.Load().AntigravityCompatibility) {
	case "off":
		return AntigravityModeOff
	case "strict":
		return AntigravityModeStrict
	default:
		return AntigravityModeVerified
	}
}

func effectiveAntigravityMode(in http.Header) string {
	serverMode := GetAntigravityMode()
	callerMode := ""
	if in != nil {
		if m := in.Get("X-Antigravity-Mode"); m != "" {
			switch strings.ToLower(m) {
			case "verified":
				callerMode = AntigravityModeVerified
			case "strict":
				callerMode = AntigravityModeStrict
			case "off":
				callerMode = AntigravityModeOff
			}
		}
	}
	if callerMode == AntigravityModeStrict || serverMode == AntigravityModeStrict {
		return AntigravityModeStrict
	}
	if callerMode == AntigravityModeVerified || serverMode == AntigravityModeVerified {
		return AntigravityModeVerified
	}
	return AntigravityModeOff
}

// validateAntigravityStrictMode checks whether requested capabilities have verified equivalent contracts.
func validateAntigravityStrictMode(in http.Header, r *Request) error {
	mode := effectiveAntigravityMode(in)
	if mode != AntigravityModeStrict || r == nil {
		return nil
	}
	for _, t := range r.Tools {
		name := strings.ToLower(t.Name)
		if name == "generate_image" || name == "image_gen" || strings.Contains(name, "subagent") {
			return fmt.Errorf("strict mode: capability %q lacks verified equivalent contract on Antigravity bridge", t.Name)
		}
	}
	return nil
}
