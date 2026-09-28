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

// testB03bToolArgsValidationHNonTarget verifies non-target providers preserve legacy behavior:
// Ordinary Chat wraps non-JSON string into {"input": s}, Gemini CLI defaults empty args to {}.
func testB03bToolArgsValidationHNonTarget(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"id":"c","choices":[{"delta":{"content":"ok"},"finish_reason":null}]}`, `data: [DONE]`)

	// 1. Ordinary Chat Provider wraps invalid non-JSON string into {"input": s}
	t.Run("OrdinaryChat_WrapsInvalidString", func(t *testing.T) {
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

		payload := `{
			"model": "m1",
			"messages": [
				{"role": "user", "content": "run"},
				{"role": "assistant", "tool_calls": [
					{"id": "c1", "type": "function", "function": {"name": "fn", "arguments": "plain string not json"}}
				]}
			]
		}`

		rec := httptest.NewRecorder()
		httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(payload)))
		httpReq.Header.Set("Content-Type", "application/json")

		var u Usage
		status, errMsg := s.translate(rec, httpReq, p, provider.Chat, provider.Chat, "m1", []byte(payload), &u)
		if status != http.StatusOK {
			t.Fatalf("ordinary chat failed with status %d: %s", status, rec.Body.String())
		}
		if rec.Code != http.StatusOK {
			t.Errorf("expected rec.Code 200, got %d", rec.Code)
		}
		if errMsg != "" {
			t.Errorf("unexpected errMsg: %s", errMsg)
		}
		if atomic.LoadInt64(&tr.callCount) != 1 {
			t.Errorf("expected 1 upstream call, got %d", tr.callCount)
		}

		// Verify wire arguments accurately wrapped as {"input":"plain string not json"} using jsonEqualExact
		var wireChat struct {
			Messages []struct {
				ToolCalls []struct {
					Function struct {
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(tr.capturedBody, &wireChat); err != nil {
			t.Fatalf("unmarshal wire failed: %v", err)
		}
		if len(wireChat.Messages) < 2 || len(wireChat.Messages[1].ToolCalls) == 0 {
			t.Fatalf("wire missing assistant tool calls: %s", string(tr.capturedBody))
		}
		gotArgs := wireChat.Messages[1].ToolCalls[0].Function.Arguments
		eq, err := jsonEqualExact([]byte(gotArgs), []byte(`{"input":"plain string not json"}`))
		if err != nil || !eq {
			t.Errorf("ordinary chat wire arguments not wrapped accurately:\ngot:  %s\nwant: %s (err: %v)", gotArgs, `{"input":"plain string not json"}`, err)
		}
	})

	// 2. Gemini CLI defaults omitted args to {}
	t.Run("GeminiCLI_EmptyDefaultsToEmptyObject", func(t *testing.T) {
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

		payload := `{
			"model": "gemini-2.5-pro",
			"messages": [
				{"role": "user", "content": "run"},
				{"role": "assistant", "content": [
					{"type": "tool_use", "id": "c1", "name": "fn"}
				]}
			]
		}`

		rec := httptest.NewRecorder()
		httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(payload)))
		httpReq.Header.Set("Content-Type", "application/json")

		var u Usage
		status, errMsg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "gemini-2.5-pro", []byte(payload), &u)
		if status != http.StatusOK {
			t.Fatalf("gemini CLI failed with status %d: %s", status, rec.Body.String())
		}
		if rec.Code != http.StatusOK {
			t.Errorf("expected rec.Code 200, got %d", rec.Code)
		}
		if errMsg != "" {
			t.Errorf("unexpected errMsg: %s", errMsg)
		}
		if atomic.LoadInt64(&tr.callCount) != 1 {
			t.Errorf("expected 1 upstream call, got %d", tr.callCount)
		}

		// Verify wire functionCall args defaulted to {}
		args := extractFirstFunctionCallArgs(t, tr.capturedBody)
		eq, err := jsonEqualExact(args, []byte(`{}`))
		if err != nil || !eq {
			t.Errorf("gemini CLI wire args not defaulted to {}: got %s (err: %v)", string(args), err)
		}
	})
}
