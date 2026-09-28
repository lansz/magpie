package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03dBlockTypeValidationHScope covers scope exclusion scenarios:
// developer, assistant messages, tool results with nested unmapped blocks, string inputs, and non-target legacy behaviors.
func testB03dBlockTypeValidationHScope(t *testing.T) {
	t.Run("ScopeExclusionsAndNestedUnmappedBlocksNotDamaged", testB03dHScopeExclusionsNotDamaged)
	t.Run("NonTargetLegacyBehaviorUnchanged", testB03dHNonTargetLegacyBehaviorUnchanged)
}

// 1. Scope exclusions: developer with unmapped blocks, tool_result with nested unmapped blocks, string inputs.
func testB03dHScopeExclusionsNotDamaged(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"verified ok"}]},"finishReason":"STOP"}]}}`)

	cases := []struct {
		name     string
		from     provider.Protocol
		model    string
		payload  string
		wantStop string
	}{
		// (a) Tool_result with nested unmapped block and tool_use with business type key: must not recurse into tool_result
		{
			name:     "Messages_ToolResultNestedUnmappedBlock_NotRejected",
			from:     provider.Anthropic,
			model:    "claude-sonnet-4-6",
			payload:  `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"query","input":{"type":"unmapped_type"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"c1","content":[{"type":"unmapped_inner_block"},{"type":"text","text":"ok"}]}]}]}`,
			wantStop: "end_turn",
		},
		// (b) Assistant message containing unmapped block type: rule checks role=user only
		{
			name:     "Messages_AssistantUnmappedBlock_NotRejected",
			from:     provider.Anthropic,
			model:    "gemini-3.8-flash-high",
			payload:  `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"custom_assistant_block","data":"raw"}]},{"role":"user","content":"next"}]}`,
			wantStop: "end_turn",
		},
		// (c) Responses non-message extension item: function_call_output with role=user and content:[{type:custom_unknown}]
		{
			name:     "Responses_FunctionCallOutputWithContentExtension_NotRejected",
			from:     provider.Responses,
			model:    "gemini-3.8-flash-high",
			payload:  `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"start"},{"type":"function_call","call_id":"c1","name":"fn","arguments":"{}"},{"type":"function_call_output","role":"user","call_id":"c1","output":"valid text","content":[{"type":"custom_unknown"}]}]}`,
			wantStop: "completed",
		},
		// (d) Developer message with unmapped block type: rule checks role=user only
		{
			name:     "Chat_DeveloperMessageWithUnmappedBlock_NotRejected",
			from:     provider.Chat,
			model:    "claude-sonnet-4-6",
			payload:  `{"model":"claude-sonnet-4-6","messages":[{"role":"developer","content":[{"type":"custom_dev_block","data":"raw"}]},{"role":"user","content":"hi"}]}`,
			wantStop: "stop",
		},
		// (e) String content forms across all 3 protocols: not arrays, not rejected
		{
			name:     "Messages_StringContent_NotRejected",
			from:     provider.Anthropic,
			model:    "claude-sonnet-4-6",
			payload:  `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"plain string text"}]}`,
			wantStop: "end_turn",
		},
		{
			name:     "Chat_StringContent_NotRejected",
			from:     provider.Chat,
			model:    "gemini-3.8-flash-high",
			payload:  `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"plain string text"}]}`,
			wantStop: "stop",
		},
		{
			name:     "Responses_StringInput_NotRejected",
			from:     provider.Responses,
			model:    "gemini-3.8-flash-high",
			payload:  `{"model":"gemini-3.8-flash-high","input":"plain string input"}`,
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

			assertClientTextAndStop(t, rec.Body.Bytes(), tc.from, "verified ok", tc.wantStop)
		})
	}
}

// 2. Non-target providers preserve legacy behavior: ignore unknown block types and preserve exact text sequence on wire.
func testB03dHNonTargetLegacyBehaviorUnchanged(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"id":"c","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`, `data: [DONE]`)

	t.Run("OrdinaryChat_IgnoresUnknownBlockPreservesText", func(t *testing.T) {
		s := New()
		tr := &mockCaptureTransport{sseReply: sseChunk}
		s.client = &http.Client{Transport: tr}

		p := provider.Provider{
			ID:     "plain-chat",
			Name:   "PlainChat",
			Key:    "k",
			Models: []string{"m1"},
			Chat:   "http://127.0.0.1/v1",
		}

		payload := `{"model":"m1","messages":[{"role":"user","content":[{"type":"text","text":"pre_text"},{"type":"custom_unknown","data":"ignored"},{"type":"text","text":"post_text"}]}]}`

		rec := httptest.NewRecorder()
		httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(payload)))
		httpReq.Header.Set("Content-Type", "application/json")

		var u Usage
		status, errMsg := s.translate(rec, httpReq, p, provider.Chat, provider.Chat, "m1", []byte(payload), &u)
		if status != http.StatusOK || rec.Code != http.StatusOK || errMsg != "" {
			t.Fatalf("ordinary chat failed: status=%d, rec.Code=%d, errMsg=%s", status, rec.Code, errMsg)
		}
		if atomic.LoadInt64(&tr.callCount) != 1 {
			t.Errorf("expected 1 upstream call, got %d", tr.callCount)
		}

		// Verify wire chat: Chat codec flushes adjacent text into single string content
		var wireChat struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(tr.capturedBody, &wireChat); err != nil {
			t.Fatalf("unmarshal wire chat failed: %v", err)
		}
		if len(wireChat.Messages) != 1 {
			t.Fatalf("expected 1 wire chat message, got %d", len(wireChat.Messages))
		}
		if wireChat.Messages[0].Content != "pre_textpost_text" {
			t.Errorf("wire content mismatch: got %q, want 'pre_textpost_text'", wireChat.Messages[0].Content)
		}

		assertClientTextAndStop(t, rec.Body.Bytes(), provider.Chat, "ok", "stop")
	})

	t.Run("GeminiCLI_IgnoresUnknownBlockPreservesText", func(t *testing.T) {
		codeAssistSSE := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"cli ok"}]},"finishReason":"STOP"}]}}`)

		s := New()
		tr := &mockCaptureTransport{sseReply: codeAssistSSE}
		s.client = &http.Client{Transport: tr}

		p := provider.Provider{
			ID:      "gemini-cli",
			Name:    "GeminiCLI",
			Key:     "k",
			Models:  []string{"gemini-2.5-pro"},
			Account: &provider.Account{Agent: "gemini"},
		}

		payload := `{"model":"gemini-2.5-pro","messages":[{"role":"user","content":[{"type":"text","text":"pre_text"},{"type":"custom_unknown","data":"ignored"},{"type":"text","text":"post_text"}]}]}`

		rec := httptest.NewRecorder()
		httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(payload)))
		httpReq.Header.Set("Content-Type", "application/json")

		var u Usage
		status, errMsg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "gemini-2.5-pro", []byte(payload), &u)
		if status != http.StatusOK || rec.Code != http.StatusOK || errMsg != "" {
			t.Fatalf("gemini CLI failed: status=%d, rec.Code=%d, errMsg=%s", status, rec.Code, errMsg)
		}
		if atomic.LoadInt64(&tr.callCount) != 1 {
			t.Errorf("expected 1 upstream call, got %d", tr.callCount)
		}

		var env codeAssistWireEnvelope
		if err := json.Unmarshal(tr.capturedBody, &env); err != nil {
			t.Fatalf("unmarshal wire failed: %v", err)
		}
		if len(env.Request.Contents) != 1 {
			t.Fatalf("expected 1 content on wire, got %d", len(env.Request.Contents))
		}
		parts := env.Request.Contents[0].Parts
		if len(parts) != 2 || parts[0].Text != "pre_text" || parts[1].Text != "post_text" {
			t.Errorf("gemini CLI wire parts mismatch: %+v", parts)
		}

		assertClientTextAndStop(t, rec.Body.Bytes(), provider.Anthropic, "cli ok", "end_turn")
	})
}
