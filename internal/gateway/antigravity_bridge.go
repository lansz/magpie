package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/yetone/magpie/internal/provider"
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
