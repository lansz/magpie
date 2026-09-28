package gateway

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03dBlockTypeValidationHValid covers all H-layer valid block type throughput scenarios.
func testB03dBlockTypeValidationHValid(t *testing.T) {
	t.Run("ValidTextImageTextAcrossProtocols", testB03dHValidTextImageTextAcrossProtocols)
	t.Run("MessagesUserToolUseAndThinkingAcknowledged", testB03dHMessagesUserToolUseAndThinkingAcknowledged)
	t.Run("ResponsesUserFourAcknowledgedTypesAndExplicitTypeMessage", testB03dHResponsesUserFourAcknowledgedTypesAndExplicitTypeMessage)
}

func assertValidTextImageTextWire(t *testing.T, wire []byte) {
	t.Helper()
	var env codeAssistWireEnvelope
	if err := json.Unmarshal(wire, &env); err != nil {
		t.Fatalf("unmarshal wire failed: %v", err)
	}
	if len(env.Request.Contents) != 1 || env.Request.Contents[0].Role != "user" {
		t.Fatalf("expected 1 user content on wire, got: %+v", env.Request.Contents)
	}
	parts := env.Request.Contents[0].Parts
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts on wire, got: %d", len(parts))
	}
	// Part 0: text pre
	if parts[0].Text != "pre" || parts[0].InlineData != nil || parts[0].FileData != nil {
		t.Errorf("part 0 mismatch: %+v", parts[0])
	}
	// Part 1: image
	if parts[1].Text != "" || parts[1].FileData != nil || parts[1].InlineData == nil {
		t.Fatalf("part 1 type exclusivity mismatch: %+v", parts[1])
	}
	if parts[1].InlineData.MimeType != "image/png" || parts[1].InlineData.Data != samplePNGBase64 {
		t.Errorf("part 1 data mismatch: %+v", parts[1].InlineData)
	}
	decoded, err := base64.StdEncoding.DecodeString(parts[1].InlineData.Data)
	if err != nil || len(decoded) == 0 {
		t.Errorf("part 1 base64 decode failed: %v", err)
	}
	// Part 2: text post
	if parts[2].Text != "post" || parts[2].InlineData != nil || parts[2].FileData != nil {
		t.Errorf("part 2 mismatch: %+v", parts[2])
	}
}

// 1. Valid text-image-text roundtrip across 3 protocols with shared wire validation.
func testB03dHValidTextImageTextAcrossProtocols(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello back"}]},"finishReason":"STOP"}]}}`)

	cases := []struct {
		name     string
		from     provider.Protocol
		model    string
		payload  string
		wantStop string
	}{
		{
			name:     "Messages",
			from:     provider.Anthropic,
			model:    "gemini-3.8-flash-high",
			payload:  fmt.Sprintf(`{"model":"gemini-3.8-flash-high","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"text","text":"pre"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":%q}},{"type":"text","text":"post"}]}]}`, samplePNGBase64),
			wantStop: "end_turn",
		},
		{
			name:     "Chat",
			from:     provider.Chat,
			model:    "claude-sonnet-4-6",
			payload:  fmt.Sprintf(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"pre"},{"type":"image_url","image_url":{"url":%q}},{"type":"text","text":"post"}]}]}`, samplePNGDataURL),
			wantStop: "stop",
		},
		{
			name:     "Responses",
			from:     provider.Responses,
			model:    "gemini-3.8-flash-high",
			payload:  fmt.Sprintf(`{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":[{"type":"input_text","text":"pre"},{"type":"input_image","image_url":%q},{"type":"input_text","text":"post"}]}]}`, samplePNGDataURL),
			wantStop: "completed",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := New()
			tr := &mockCaptureTransport{sseReply: sseChunk}
			s.client = &http.Client{Transport: tr}

			targetProvider := provider.Provider{
				ID:      "antigravity-test",
				Name:    "AntigravityProvider",
				Key:     "k",
				Models:  []string{tc.model},
				Account: &provider.Account{Agent: "antigravity"},
			}

			rec := httptest.NewRecorder()
			httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(tc.payload)))
			httpReq.Header.Set("Content-Type", "application/json")

			var u Usage
			status, errMsg := s.translate(rec, httpReq, targetProvider, tc.from, provider.CodeAssist, tc.model, []byte(tc.payload), &u)
			if status != http.StatusOK || rec.Code != http.StatusOK || errMsg != "" {
				t.Fatalf("[%s] expected HTTP 200, got status %d, rec.Code %d, errMsg: %s", tc.name, status, rec.Code, errMsg)
			}
			if atomic.LoadInt64(&tr.callCount) != 1 {
				t.Errorf("[%s] expected 1 upstream call, got %d", tc.name, tr.callCount)
			}

			assertClientTextAndStop(t, rec.Body.Bytes(), tc.from, "hello back", tc.wantStop)
			assertValidTextImageTextWire(t, tr.capturedBody)
		})
	}
}

// 2. Direct H reachability for Messages user-turn containing tool_use and thinking:
// verifies whitelist does not erroneously reject these acknowledged types in user messages.
func testB03dHMessagesUserToolUseAndThinkingAcknowledged(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello back"}]},"finishReason":"STOP"}]}}`)

	payload := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "start"},
				{"type": "thinking", "thinking": "user_think", "signature": "sig_usr"},
				{"type": "tool_use", "id": "c_usr", "name": "do_cmd", "input": {"x": 1}},
				{"type": "text", "text": "finish"}
			]}
		]
	}`

	s := New()
	tr := &mockCaptureTransport{sseReply: sseChunk}
	s.client = &http.Client{Transport: tr}

	targetProvider := provider.Provider{
		ID:      "antigravity-test",
		Name:    "AntigravityProvider",
		Key:     "k",
		Models:  []string{"claude-sonnet-4-6"},
		Account: &provider.Account{Agent: "antigravity"},
	}

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(payload)))
	httpReq.Header.Set("Content-Type", "application/json")

	var u Usage
	status, errMsg := s.translate(rec, httpReq, targetProvider, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(payload), &u)
	if status != http.StatusOK || rec.Code != http.StatusOK || errMsg != "" {
		t.Fatalf("expected HTTP 200 for user tool_use/thinking acknowledged types, got status %d, errMsg: %s", status, errMsg)
	}
	if atomic.LoadInt64(&tr.callCount) != 1 {
		t.Errorf("expected 1 upstream call, got %d", tr.callCount)
	}
	assertClientTextAndStop(t, rec.Body.Bytes(), provider.Anthropic, "hello back", "end_turn")
}

// 3. Direct H reachability for Responses user message containing all 4 acknowledged types:
// (input_text, text, output_text, input_image) with both explicit type=message and omitted type.
func testB03dHResponsesUserFourAcknowledgedTypesAndExplicitTypeMessage(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello back"}]},"finishReason":"STOP"}]}}`)

	cases := []struct {
		name    string
		payload string
	}{
		{
			name: "ExplicitTypeMessage",
			payload: fmt.Sprintf(`{
				"model": "gemini-3.8-flash-high",
				"input": [
					{
						"type": "message",
						"role": "user",
						"content": [
							{"type": "input_text", "text": "p1"},
							{"type": "text", "text": "p2"},
							{"type": "output_text", "text": "p3"},
							{"type": "input_image", "image_url": %q}
						]
					}
				]
			}`, samplePNGDataURL),
		},
		{
			name: "OmittedTypeWithRole",
			payload: fmt.Sprintf(`{
				"model": "gemini-3.8-flash-high",
				"input": [
					{
						"role": "user",
						"content": [
							{"type": "input_text", "text": "p1"},
							{"type": "text", "text": "p2"},
							{"type": "output_text", "text": "p3"},
							{"type": "input_image", "image_url": %q}
						]
					}
				]
			}`, samplePNGDataURL),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := New()
			tr := &mockCaptureTransport{sseReply: sseChunk}
			s.client = &http.Client{Transport: tr}

			targetProvider := provider.Provider{
				ID:      "antigravity-test",
				Name:    "AntigravityProvider",
				Key:     "k",
				Models:  []string{"gemini-3.8-flash-high"},
				Account: &provider.Account{Agent: "antigravity"},
			}

			rec := httptest.NewRecorder()
			httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(tc.payload)))
			httpReq.Header.Set("Content-Type", "application/json")

			var u Usage
			status, errMsg := s.translate(rec, httpReq, targetProvider, provider.Responses, provider.CodeAssist, "gemini-3.8-flash-high", []byte(tc.payload), &u)
			if status != http.StatusOK || rec.Code != http.StatusOK || errMsg != "" {
				t.Fatalf("[%s] expected HTTP 200, got status %d, errMsg: %s", tc.name, status, errMsg)
			}
			if atomic.LoadInt64(&tr.callCount) != 1 {
				t.Errorf("[%s] expected 1 upstream call, got %d", tc.name, tr.callCount)
			}
			assertClientTextAndStop(t, rec.Body.Bytes(), provider.Responses, "hello back", "completed")

			// Check wire text sequence [p1, p2, p3] and image part
			var env codeAssistWireEnvelope
			if err := json.Unmarshal(tr.capturedBody, &env); err != nil {
				t.Fatalf("unmarshal wire failed: %v", err)
			}
			if len(env.Request.Contents) != 1 || env.Request.Contents[0].Role != "user" {
				t.Fatalf("expected 1 user content, got: %+v", env.Request.Contents)
			}
			parts := env.Request.Contents[0].Parts
			if len(parts) != 4 {
				t.Fatalf("expected 4 parts on wire, got %d", len(parts))
			}
			if parts[0].Text != "p1" || parts[1].Text != "p2" || parts[2].Text != "p3" {
				t.Errorf("wire text sequence mismatch: got [%q, %q, %q], want [p1, p2, p3]", parts[0].Text, parts[1].Text, parts[2].Text)
			}
			if parts[3].InlineData == nil || parts[3].InlineData.MimeType != "image/png" {
				t.Errorf("wire image part mismatch: %+v", parts[3])
			}
		})
	}
}
