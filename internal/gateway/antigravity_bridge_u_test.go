package gateway

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB02AgentBridge is the unified entrypoint for B02 BDD scenarios.
func testB02AgentBridge(t *testing.T) {
	t.Run("Unit", testB02AgentBridgeU)
	t.Run("HLayer", testB02AgentBridgeH)
}

// testB02AgentBridgeU covers all unit-level B02 BDD scenarios.
func testB02AgentBridgeU(t *testing.T) {
	t.Run("EquivalentSemanticIR_ThreeProtocols", testB02EquivalentSemanticIR)
	t.Run("OrthogonalMatrix_ProtocolAndProfile", testB02OrthogonalMatrix)
	t.Run("UnknownProfile_Isolation_Unit", testB02UnknownProfileIsolationUnit)
}

// 1. Equivalent Semantic IR across Messages, Chat, and Responses.
func testB02EquivalentSemanticIR(t *testing.T) {
	const (
		systemPrompt = "You are a test coding assistant."
		userMessage  = "Please inspect main.go and summarize."
		toolName     = "custom_view_file"
		toolDesc     = "Inspects a local file content."
	)

	// Independent expected tool definition with its own memory copy (prevents self-comparison)
	expectedToolSchema := json.RawMessage([]byte(`{"properties":{"path":{"type":"string"}},"required":["path"],"type":"object"}`))
	expectedTool := Tool{
		Name:        toolName,
		Description: toolDesc,
		Schema:      expectedToolSchema,
	}

	anthropicBody := fmt.Sprintf(`{
		"model": "claude-sonnet-4-6",
		"system": %q,
		"max_tokens": 4096,
		"thinking": {"type": "enabled", "budget_tokens": 2000},
		"messages": [{"role": "user", "content": %q}],
		"tools": [{
			"name": %q,
			"description": %q,
			"input_schema": {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
		}]
	}`, systemPrompt, userMessage, toolName, toolDesc)

	chatBody := fmt.Sprintf(`{
		"model": "gpt-4o",
		"max_tokens": 4096,
		"reasoning_effort": "low",
		"messages": [
			{"role": "system", "content": %q},
			{"role": "user", "content": %q}
		],
		"tools": [{
			"type": "function",
			"function": {
				"name": %q,
				"description": %q,
				"parameters": {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
			}
		}]
	}`, systemPrompt, userMessage, toolName, toolDesc)

	responsesBody := fmt.Sprintf(`{
		"model": "gpt-4o",
		"instructions": %q,
		"max_output_tokens": 4096,
		"reasoning": {"effort": "low"},
		"input": [
			{"role": "user", "content": [{"type": "input_text", "text": %q}]}
		],
		"tools": [{
			"type": "function",
			"name": %q,
			"description": %q,
			"parameters": {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
		}]
	}`, systemPrompt, userMessage, toolName, toolDesc)

	reqAnthropic, err := parse(provider.Anthropic, []byte(anthropicBody))
	if err != nil {
		t.Fatalf("parse Anthropic failed: %v", err)
	}
	reqChat, err := parse(provider.Chat, []byte(chatBody))
	if err != nil {
		t.Fatalf("parse Chat failed: %v", err)
	}
	reqResponses, err := parse(provider.Responses, []byte(responsesBody))
	if err != nil {
		t.Fatalf("parse Responses failed: %v", err)
	}

	breqAnthropic := newBridgeRequest(reqAnthropic, provider.Anthropic, "test_profile")
	if breqAnthropic == nil || breqAnthropic.Request == nil {
		t.Fatalf("newBridgeRequest for Anthropic returned nil or nil Request")
	}

	breqChat := newBridgeRequest(reqChat, provider.Chat, "test_profile")
	if breqChat == nil || breqChat.Request == nil {
		t.Fatalf("newBridgeRequest for Chat returned nil or nil Request")
	}

	breqResponses := newBridgeRequest(reqResponses, provider.Responses, "test_profile")
	if breqResponses == nil || breqResponses.Request == nil {
		t.Fatalf("newBridgeRequest for Responses returned nil or nil Request")
	}

	requests := []*bridgeRequest{breqAnthropic, breqChat, breqResponses}
	names := []string{"Anthropic", "Chat", "Responses"}

	// Verify System prompt equivalence against independent expected constant
	for i, r := range requests {
		if r.System != systemPrompt {
			t.Errorf("[%s] system prompt mismatch: got %q, want %q", names[i], r.System, systemPrompt)
		}
	}

	// Verify User message equivalence against independent expected constant
	for i, r := range requests {
		if len(r.Messages) == 0 {
			t.Fatalf("[%s] expected non-empty messages", names[i])
		}
		lastMsg := r.Messages[len(r.Messages)-1]
		if lastMsg.Role != "user" {
			t.Errorf("[%s] expected last message role 'user', got %q", names[i], lastMsg.Role)
		}
		gotText := text(lastMsg.Parts)
		if gotText != userMessage {
			t.Errorf("[%s] message text mismatch: got %q, want %q", names[i], gotText, userMessage)
		}
	}

	// Verify Tool definitions against independent expectedTool
	for i, r := range requests {
		if len(r.Tools) != 1 {
			t.Fatalf("[%s] expected exactly 1 tool, got %d", names[i], len(r.Tools))
		}
		tool := r.Tools[0]
		if tool.Name != expectedTool.Name {
			t.Errorf("[%s] tool name mismatch: got %q, want %q", names[i], tool.Name, expectedTool.Name)
		}
		if tool.Description != expectedTool.Description {
			t.Errorf("[%s] tool description mismatch: got %q, want %q", names[i], tool.Description, expectedTool.Description)
		}
		eq, err := jsonEqualExact(tool.Schema, expectedTool.Schema)
		if err != nil || !eq {
			t.Errorf("[%s] tool schema mismatch: got %s, want %s (err: %v)", names[i], string(tool.Schema), string(expectedTool.Schema), err)
		}
	}

	// Verify MaxTokens across all three protocols
	for i, r := range requests {
		if r.MaxTokens != 4096 {
			t.Errorf("[%s] MaxTokens mismatch: got %d, want 4096", names[i], r.MaxTokens)
		}
	}

	// Verify reasoning effort / thinking intent representation in current IR:
	// Note: Anthropic thinking 2000 tokens is parsed as boolean Thinking=true;
	// current IR does not store numerical thinking token budget (handled in B03).
	if breqChat.Effort != "low" {
		t.Errorf("Chat effort mismatch: got %q, want 'low'", breqChat.Effort)
	}
	if breqResponses.Effort != "low" {
		t.Errorf("Responses effort mismatch: got %q, want 'low'", breqResponses.Effort)
	}
	if !breqAnthropic.Thinking {
		t.Errorf("Anthropic thinking boolean intent missing on bridgeRequest")
	}
}

// 2. Orthogonal Matrix: protocol × client profile must be completely uncoupled.
func testB02OrthogonalMatrix(t *testing.T) {
	protocols := []provider.Protocol{
		provider.Anthropic,
		provider.Chat,
		provider.Responses,
	}
	profiles := []string{
		"unknown",
		"claude_code",
		"opencode",
		"pi",
		"codex",
	}

	baseReq := &Request{
		Model:  "test-model",
		System: "System prompt",
		Messages: []Message{
			{Role: "user", Parts: []Part{{Kind: Text, Text: "ping"}}},
		},
	}

	for _, proto := range protocols {
		for _, prof := range profiles {
			t.Run(fmt.Sprintf("%s_x_%s", proto, prof), func(t *testing.T) {
				breq := newBridgeRequest(baseReq, proto, prof)
				if breq == nil || breq.Request == nil {
					t.Fatalf("newBridgeRequest returned nil or nil Request for proto=%s profile=%s", proto, prof)
				}
				if breq.Request != baseReq {
					t.Errorf("bridgeRequest must wrap existing *Request without cloning")
				}
				if breq.sourceProto != proto {
					t.Errorf("sourceProto mismatch: got %v, want %v", breq.sourceProto, proto)
				}
				if breq.clientProfile != prof {
					t.Errorf("clientProfile mismatch: got %q, want %q", breq.clientProfile, prof)
				}
			})
		}
	}

	// Default fallback: empty profile must default to "unknown"
	t.Run("DefaultUnknownProfileWhenEmpty", func(t *testing.T) {
		breq := newBridgeRequest(baseReq, provider.Chat, "")
		if breq == nil || breq.Request == nil {
			t.Fatalf("newBridgeRequest returned nil for empty profile")
		}
		if breq.clientProfile != "unknown" {
			t.Errorf("expected empty profile to default to 'unknown', got %q", breq.clientProfile)
		}
	})
}

// 3. Unknown Profile isolation: unit verification.
func testB02UnknownProfileIsolationUnit(t *testing.T) {
	// Independent expected tool definition
	expectedTool := Tool{
		Name:        "user_defined_calculator",
		Description: "Calculates mathematical expressions",
		Schema:      json.RawMessage([]byte(`{"properties":{"expr":{"type":"string"}},"required":["expr"],"type":"object"}`)),
	}

	// Distinct memory buffer for input tool definition (prevents reference sharing with expectedTool)
	inputToolSchema := json.RawMessage([]byte(`{"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}`))
	inputTool := Tool{
		Name:        "user_defined_calculator",
		Description: "Calculates mathematical expressions",
		Schema:      inputToolSchema,
	}

	unitReq := &Request{
		Model: "m1",
		Tools: []Tool{inputTool},
		Messages: []Message{
			{Role: "user", Parts: []Part{{Kind: Text, Text: "run calc"}}},
		},
	}

	breq := newBridgeRequest(unitReq, provider.Anthropic, "")
	if breq == nil || breq.Request == nil {
		t.Fatalf("newBridgeRequest returned nil or nil Request")
	}
	if breq.clientProfile != "unknown" {
		t.Fatalf("expected clientProfile to default to 'unknown', got %q", breq.clientProfile)
	}
	if len(breq.Tools) != 1 {
		t.Fatalf("tool count changed in bridgeRequest: got %d, want 1", len(breq.Tools))
	}
	if breq.Tools[0].Name != expectedTool.Name {
		t.Errorf("tool mismatch: got %q, want %q", breq.Tools[0].Name, expectedTool.Name)
	}
	if breq.Tools[0].Description != expectedTool.Description {
		t.Errorf("tool description mismatch: got %q, want %q", breq.Tools[0].Description, expectedTool.Description)
	}
	eq, err := jsonEqualExact(breq.Tools[0].Schema, expectedTool.Schema)
	if err != nil || !eq {
		t.Errorf("tool schema mismatch: got %s, want %s", string(breq.Tools[0].Schema), string(expectedTool.Schema))
	}
}
