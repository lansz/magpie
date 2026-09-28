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

// testB03cImageValidationHContext covers context exclusion scenarios:
// keywords in text, tool results containing images, tool args with image keys, and non-target legacy behaviors.
func testB03cImageValidationHContext(t *testing.T) {
	t.Run("NormalTextAndKeywordsNotDamaged", testB03cHNormalTextAndKeywordsNotDamaged)
	t.Run("NonTargetProvidersPreserveLegacyBehavior", testB03cHNonTargetProvidersPreserveLegacyBehavior)
}

// assertClientTextAndStop parses the client response for the given protocol and asserts the returned text and stop reason.
func assertClientTextAndStop(t *testing.T, body []byte, proto provider.Protocol, wantText, wantStop string) {
	t.Helper()
	switch proto {
	case provider.Anthropic:
		var rep struct {
			Content    []struct{ Text string } `json:"content"`
			StopReason string                  `json:"stop_reason"`
		}
		if err := json.Unmarshal(body, &rep); err != nil {
			t.Fatalf("unmarshal Messages client failed: %v", err)
		}
		if len(rep.Content) == 0 {
			t.Fatalf("Messages client response missing content: %s", string(body))
		}
		if rep.Content[0].Text != wantText || rep.StopReason != wantStop {
			t.Errorf("Messages client mismatch: %+v", rep)
		}
	case provider.Chat:
		var rep struct {
			Choices []struct {
				Message      struct{ Content string } `json:"message"`
				FinishReason string                   `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &rep); err != nil {
			t.Fatalf("unmarshal Chat client failed: %v", err)
		}
		if len(rep.Choices) == 0 {
			t.Fatalf("Chat client response missing choices: %s", string(body))
		}
		if rep.Choices[0].Message.Content != wantText || rep.Choices[0].FinishReason != wantStop {
			t.Errorf("Chat client mismatch: %+v", rep)
		}
	case provider.Responses:
		var rep struct {
			Status string `json:"status"`
			Output []struct {
				Content []struct{ Text string } `json:"content"`
			} `json:"output"`
		}
		if err := json.Unmarshal(body, &rep); err != nil {
			t.Fatalf("unmarshal Responses client failed: %v", err)
		}
		if rep.Status != wantStop || len(rep.Output) == 0 || len(rep.Output[0].Content) == 0 || rep.Output[0].Content[0].Text != wantText {
			t.Errorf("Responses client mismatch: %+v", rep)
		}
	}
}

// 1. Normal text, keywords, tool results with nested images, tool arguments with image keys, and assistant images are not damaged.
func testB03cHNormalTextAndKeywordsNotDamaged(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"verified ok"}]},"finishReason":"STOP"}]}}`)

	cases := []struct {
		name       string
		from       provider.Protocol
		model      string
		payload    string
		wantStop   string
		assertWire func(t *testing.T, wire []byte)
	}{
		{
			name:     "Messages_TextWithKeywords",
			from:     provider.Anthropic,
			model:    "gemini-3.8-flash-high",
			payload:  `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":"Discuss data:image/png;base64 and source format."}]}`,
			wantStop: "end_turn",
		},
		{
			name:     "Chat_TextWithKeywords",
			from:     provider.Chat,
			model:    "claude-sonnet-4-6",
			payload:  `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Explain image_url and base64 encoding."}]}`,
			wantStop: "stop",
		},
		{
			name:     "Responses_TextWithKeywords",
			from:     provider.Responses,
			model:    "gemini-3.8-flash-high",
			payload:  `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"Talk about input_image and image_url format."}]}`,
			wantStop: "completed",
		},
		// tool_result.content containing missing-payload image (which would be rejected in user message): proves rule does not recurse into tool_result
		{
			name:  "Messages_ToolResultContainingImageNode_NotDamaged",
			from:  provider.Anthropic,
			model: "gemini-3.8-flash-high",
			payload: `{
				"model": "gemini-3.8-flash-high",
				"max_tokens": 100,
				"messages": [
					{"role": "user", "content": "run tool"},
					{"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "take_shot", "input": {}}]},
					{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": [{"type": "image", "source": {"type": "base64", "data": ""}}]}]}
				]
			}`,
			wantStop: "end_turn",
		},
		// tool_use.input having business keys named source/image_url: must not be rejected
		{
			name:  "Chat_ToolUseWithImageKeys_NotDamaged",
			from:  provider.Chat,
			model: "claude-sonnet-4-6",
			payload: `{
				"model": "claude-sonnet-4-6",
				"messages": [
					{"role": "user", "content": "analyze"},
					{"role": "assistant", "tool_calls": [
						{"id": "c1", "type": "function", "function": {"name": "fetch", "arguments": "{\"image_url\":null,\"source\":\"db\"}"}}
					]}
				]
			}`,
			wantStop: "stop",
		},
		// assistant history message containing missing-payload image: must not be rejected by user message validator, and wire drops empty image preserving text [hi, next]
		{
			name:  "Messages_AssistantHistoryImage_NotDamaged",
			from:  provider.Anthropic,
			model: "gemini-3.8-flash-high",
			payload: `{
				"model": "gemini-3.8-flash-high",
				"max_tokens": 100,
				"messages": [
					{"role": "user", "content": "hi"},
					{"role": "assistant", "content": [{"type": "image", "source": {"type": "base64", "data": ""}}]},
					{"role": "user", "content": "next"}
				]
			}`,
			wantStop: "end_turn",
			assertWire: func(t *testing.T, wire []byte) {
				var env codeAssistWireEnvelope
				if err := json.Unmarshal(wire, &env); err != nil {
					t.Fatalf("unmarshal wire failed: %v", err)
				}
				var texts []string
				for _, c := range env.Request.Contents {
					for _, p := range c.Parts {
						if p.InlineData != nil || p.FileData != nil {
							t.Errorf("expected no image parts on wire for missing-payload assistant image, got: %+v", p)
						}
						if p.Text != "" {
							texts = append(texts, p.Text)
						}
					}
				}
				if len(texts) != 2 || texts[0] != "hi" || texts[1] != "next" {
					t.Errorf("wire text sequence mismatch: got %v, want [hi, next]", texts)
				}
			},
		},
		// Responses function_call_output carrying non-standard content with missing-image_url:
		// codec only consumes output string and ignores content, validator must not recurse into non-message item.
		{
			name:     "Responses_FunctionCallOutputWithContentExtension_NotDamaged",
			from:     provider.Responses,
			model:    "gemini-3.8-flash-high",
			payload:  `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"run task"},{"type":"function_call","call_id":"c1","name":"fn","arguments":"{}"},{"type":"function_call_output","role":"user","call_id":"c1","output":"valid output text","content":[{"type":"input_image"}]}]}`,
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
			if tc.assertWire != nil {
				tc.assertWire(t, tr.capturedBody)
			}
		})
	}
}

// 2. Non-target providers preserve legacy behavior for missing image payloads.
func testB03cHNonTargetProvidersPreserveLegacyBehavior(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	t.Run("OrdinaryChat_PreservesLegacyFallback", func(t *testing.T) {
		// Explicit finish_reason: "stop" instead of null
		chatSSE := sse(`data: {"id":"c","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`, `data: [DONE]`)

		s := New()
		tr := &mockCaptureTransport{sseReply: chatSSE}
		s.client = &http.Client{Transport: tr}

		p := provider.Provider{
			ID:     "plain-chat",
			Name:   "PlainChat",
			Key:    "k",
			Models: []string{"m1"},
			Chat:   "http://127.0.0.1/v1",
		}

		payload := `{"model":"m1","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":""}}]}]}`

		rec := httptest.NewRecorder()
		httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(payload)))
		httpReq.Header.Set("Content-Type", "application/json")

		var u Usage
		status, errMsg := s.translate(rec, httpReq, p, provider.Chat, provider.Chat, "m1", []byte(payload), &u)
		if status != http.StatusOK || rec.Code != http.StatusOK || errMsg != "" {
			t.Fatalf("ordinary chat failed with status %d: %s", status, rec.Body.String())
		}
		if atomic.LoadInt64(&tr.callCount) != 1 {
			t.Errorf("expected 1 upstream call, got %d", tr.callCount)
		}

		// Verify wire structured body for ordinary Chat:
		// Chat codec preserves empty url as image_url part with "data:;base64,"
		var wireChat struct {
			Messages []struct {
				Content []struct {
					Type     string `json:"type"`
					Text     string `json:"text"`
					ImageURL *struct {
						URL string `json:"url"`
					} `json:"image_url"`
				} `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(tr.capturedBody, &wireChat); err != nil {
			t.Fatalf("unmarshal wire chat failed: %v", err)
		}
		if len(wireChat.Messages) != 1 || len(wireChat.Messages[0].Content) != 2 {
			t.Fatalf("expected 2 content parts on wire chat, got %+v", wireChat.Messages)
		}
		c0 := wireChat.Messages[0].Content[0]
		c1 := wireChat.Messages[0].Content[1]
		if c0.Type != "text" || c0.Text != "hi" {
			t.Errorf("part 0 mismatch: %+v", c0)
		}
		if c1.Type != "image_url" || c1.ImageURL == nil || c1.ImageURL.URL != "data:;base64," {
			t.Errorf("part 1 mismatch: %+v (want URL \"data:;base64,\")", c1)
		}

		var clientRep struct {
			Choices []struct {
				Message      struct{ Content string } `json:"message"`
				FinishReason string                   `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &clientRep); err != nil {
			t.Fatalf("unmarshal client response failed: %v", err)
		}
		if len(clientRep.Choices) == 0 {
			t.Fatalf("client reply missing choices: %s", rec.Body.String())
		}
		if clientRep.Choices[0].Message.Content != "ok" {
			t.Errorf("client reply text mismatch: %s", rec.Body.String())
		}
		if clientRep.Choices[0].FinishReason != "stop" {
			t.Errorf("client finish_reason mismatch: got %q, want 'stop'", clientRep.Choices[0].FinishReason)
		}
	})

	t.Run("GeminiCLI_PreservesLegacyFallback", func(t *testing.T) {
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

		payload := `{"model":"gemini-2.5-pro","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64"}}]}]}`

		rec := httptest.NewRecorder()
		httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(payload)))
		httpReq.Header.Set("Content-Type", "application/json")

		var u Usage
		status, errMsg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "gemini-2.5-pro", []byte(payload), &u)
		if status != http.StatusOK || rec.Code != http.StatusOK || errMsg != "" {
			t.Fatalf("gemini CLI failed with status %d: %s", status, rec.Body.String())
		}
		if atomic.LoadInt64(&tr.callCount) != 1 {
			t.Errorf("expected 1 upstream call, got %d", tr.callCount)
		}

		// Verify wire structured body for Gemini CLI:
		// Empty image is dropped, but text 'hi' is preserved
		var env codeAssistWireEnvelope
		if err := json.Unmarshal(tr.capturedBody, &env); err != nil {
			t.Fatalf("unmarshal wire failed: %v", err)
		}
		if len(env.Request.Contents) != 1 || len(env.Request.Contents[0].Parts) != 1 {
			t.Fatalf("expected 1 part on wire for gemini CLI empty image, got: %+v", env.Request.Contents)
		}
		part := env.Request.Contents[0].Parts[0]
		if part.Text != "hi" || part.InlineData != nil || part.FileData != nil {
			t.Errorf("expected only text 'hi' on wire, got: %+v", part)
		}

		var clientRep struct {
			Content    []struct{ Text string } `json:"content"`
			StopReason string                  `json:"stop_reason"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &clientRep); err != nil {
			t.Fatalf("unmarshal client response failed: %v", err)
		}
		if len(clientRep.Content) == 0 {
			t.Fatalf("client reply missing content: %s", rec.Body.String())
		}
		if clientRep.Content[0].Text != "cli ok" {
			t.Errorf("client reply text mismatch: %s", rec.Body.String())
		}
		if clientRep.StopReason != "end_turn" {
			t.Errorf("client stop_reason mismatch: got %q, want 'end_turn'", clientRep.StopReason)
		}
	})
}
