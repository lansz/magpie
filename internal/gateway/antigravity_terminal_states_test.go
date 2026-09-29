package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB18TerminalStates covers B18: Valid terminal states across three protocols and
// both streaming/non-streaming modes; exact correspondence with actual finish reasons;
// elimination of stop-masking on MALFORMED_FUNCTION_CALL; and preventing corrupted frames
// from generating HTTP 200 success responses.
func testB18TerminalStates(t *testing.T) {
	t.Run("TextStopAcrossAllProtocolsAndModes", testB18TextStopAcrossAllProtocolsAndModes)
	t.Run("ToolCallTerminalStateAcrossProtocols", testB18ToolCallTerminalStateAcrossProtocols)
	t.Run("MaxTokensTerminalStateNoBudgetExpansion", testB18MaxTokensTerminalStateNoBudgetExpansion)
	t.Run("MalformedFunctionCallTreatedAsErrorNotStop", testB18MalformedFunctionCallTreatedAsErrorNotStop)
	t.Run("CorruptedFrameNeverProduces200Success", testB18CorruptedFrameNeverProduces200Success)
}

// 1. Text completion (STOP): All three protocols in both stream and non-stream modes
// must map accurately to their respective end_turn / stop / completed.
func testB18TextStopAcrossAllProtocolsAndModes(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	stopSSE := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]},"finishReason":"STOP"}]}}`)

	cases := []struct {
		name       string
		from       provider.Protocol
		stream     bool
		body       string
		assertResp func(t *testing.T, rec *httptest.ResponseRecorder)
	}{
		{
			name: "Anthropic_Stream", from: provider.Anthropic, stream: true,
			body: `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`,
			assertResp: func(t *testing.T, rec *httptest.ResponseRecorder) {
				out := rec.Body.String()
				if !strings.Contains(out, `"stop_reason":"end_turn"`) {
					t.Errorf("Anthropic stream missing stop_reason end_turn: %s", out)
				}
			},
		},
		{
			name: "Anthropic_NonStream", from: provider.Anthropic, stream: false,
			body: `{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}`,
			assertResp: func(t *testing.T, rec *httptest.ResponseRecorder) {
				var res struct {
					StopReason string `json:"stop_reason"`
				}
				json.Unmarshal(rec.Body.Bytes(), &res)
				if res.StopReason != "end_turn" {
					t.Errorf("Anthropic non-stream stop_reason = %q, want 'end_turn'", res.StopReason)
				}
			},
		},
		{
			name: "Chat_Stream", from: provider.Chat, stream: true,
			body: `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`,
			assertResp: func(t *testing.T, rec *httptest.ResponseRecorder) {
				out := rec.Body.String()
				if !strings.Contains(out, `"finish_reason":"stop"`) {
					t.Errorf("Chat stream missing finish_reason stop: %s", out)
				}
			},
		},
		{
			name: "Chat_NonStream", from: provider.Chat, stream: false,
			body: `{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}`,
			assertResp: func(t *testing.T, rec *httptest.ResponseRecorder) {
				var res struct {
					Choices []struct {
						FinishReason string `json:"finish_reason"`
					} `json:"choices"`
				}
				json.Unmarshal(rec.Body.Bytes(), &res)
				if len(res.Choices) == 0 || res.Choices[0].FinishReason != "stop" {
					t.Errorf("Chat non-stream finish_reason mismatch: %v", res.Choices)
				}
			},
		},
		{
			name: "Responses_Stream", from: provider.Responses, stream: true,
			body: `{"model":"m","stream":true,"input":[{"role":"user","content":"hi"}]}`,
			assertResp: func(t *testing.T, rec *httptest.ResponseRecorder) {
				out := rec.Body.String()
				if !strings.Contains(out, `"status":"completed"`) {
					t.Errorf("Responses stream missing status completed: %s", out)
				}
			},
		},
		{
			name: "Responses_NonStream", from: provider.Responses, stream: false,
			body: `{"model":"m","stream":false,"input":[{"role":"user","content":"hi"}]}`,
			assertResp: func(t *testing.T, rec *httptest.ResponseRecorder) {
				var res struct {
					Status string `json:"status"`
				}
				json.Unmarshal(rec.Body.Bytes(), &res)
				if res.Status != "completed" {
					t.Errorf("Responses non-stream status = %q, want 'completed'", res.Status)
				}
			},
		},
	}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := New()
			s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: stopSSE}}
			rec := httptest.NewRecorder()
			req := httptest.NewRequest("POST", "/dummy", strings.NewReader(tc.body))

			var u Usage
			status, msg := s.translate(rec, req, p, tc.from, provider.CodeAssist, "claude-sonnet-4-6", []byte(tc.body), &u)
			if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
				t.Fatalf("translate failed: %d/%d msg=%q", status, rec.Code, msg)
			}
			tc.assertResp(t, rec)
		})
	}
}

// 2. Tool call terminal state: Must NOT masquerade as end_turn.
func testB18ToolCallTerminalStateAcrossProtocols(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	toolSSE := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"c1","name":"read","args":{}}}]},"finishReason":"STOP"}]}}`)

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	// Anthropic Messages non-stream
	t.Run("Anthropic_ToolUse", func(t *testing.T) {
		s := New()
		s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: toolSSE}}
		rec := httptest.NewRecorder()
		body := `{"model":"m","tools":[{"name":"read","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`
		req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
		var u Usage
		s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

		var res struct {
			StopReason string `json:"stop_reason"`
		}
		json.Unmarshal(rec.Body.Bytes(), &res)
		if res.StopReason != "tool_use" {
			t.Errorf("tool call must result in stop_reason 'tool_use', got %q", res.StopReason)
		}
		if res.StopReason == "end_turn" {
			t.Errorf("tool call illegally masqueraded as 'end_turn'")
		}
	})

	// Chat non-stream
	t.Run("Chat_ToolCalls", func(t *testing.T) {
		s := New()
		s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: toolSSE}}
		rec := httptest.NewRecorder()
		body := `{"model":"m","tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"hi"}]}`
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
		var u Usage
		s.translate(rec, req, p, provider.Chat, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

		var res struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
			} `json:"choices"`
		}
		json.Unmarshal(rec.Body.Bytes(), &res)
		if len(res.Choices) == 0 || res.Choices[0].FinishReason != "tool_calls" {
			t.Errorf("tool call must result in finish_reason 'tool_calls', got %v", res.Choices)
		}
	})
}

// 3. MAX_TOKENS is a legitimate terminal state; magpie must NOT expand budget or retry.
func testB18MaxTokensTerminalStateNoBudgetExpansion(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	maxTokensSSE := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"cut off text"}]},"finishReason":"MAX_TOKENS"}]}}`)

	tr := &mockCaptureTransport{sseReply: maxTokensSSE}
	s := New()
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	body := `{"model":"m","max_tokens":128,"messages":[{"role":"user","content":"write an essay"}]}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))

	var u Usage
	status, msg := s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)
	if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
		t.Fatalf("translate returned status %d/%d msg=%q", status, rec.Code, msg)
	}

	// Must be exactly 1 upstream call (no automatic retry or continuation loop)
	if calls := atomic.LoadInt64(&tr.callCount); calls != 1 {
		t.Fatalf("expected exactly 1 call for MAX_TOKENS, got %d (budget expansion retry detected)", calls)
	}

	var res struct {
		StopReason string `json:"stop_reason"`
	}
	json.Unmarshal(rec.Body.Bytes(), &res)
	if res.StopReason != "max_tokens" {
		t.Errorf("expected stop_reason 'max_tokens', got %q", res.StopReason)
	}
}

// 4. MALFORMED_FUNCTION_CALL must be treated as a protocol error, NEVER masked as stop.
func testB18MalformedFunctionCallTreatedAsErrorNotStop(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	malformedSSE := sse(`data: {"response":{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL"}]}}`)

	s := New()
	s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: malformedSSE}}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	body := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))

	var u Usage
	status, msg := s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

	// Must be rejected as error, NOT success 200
	if status == http.StatusOK || rec.Code == http.StatusOK {
		t.Fatalf("MALFORMED_FUNCTION_CALL was illegally masked as HTTP 200 stop!")
	}
	if !strings.Contains(msg, "MALFORMED_FUNCTION_CALL") && !strings.Contains(rec.Body.String(), "MALFORMED_FUNCTION_CALL") && !strings.Contains(msg, "error") {
		t.Errorf("error output should indicate protocol error: msg=%q body=%s", msg, rec.Body.String())
	}
}

// 5. Corrupted frame / unparseable JSON must NEVER produce an HTTP 200 response.
func testB18CorruptedFrameNeverProduces200Success(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	// Upstream returns corrupted non-JSON chunk
	brokenSSE := sse(`data: {bad-json-missing-quotes: true`)

	s := New()
	s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: brokenSSE}}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	body := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))

	var u Usage
	status, msg := s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

	// Must NOT return 200 OK
	if status == http.StatusOK || rec.Code == http.StatusOK {
		t.Fatalf("corrupted frame illegally produced HTTP 200 response! msg=%q body=%s", msg, rec.Body.String())
	}
}
