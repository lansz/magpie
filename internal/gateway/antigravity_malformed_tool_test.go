package gateway

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB19MalformedToolAndBrokenFrame covers B19: Upstream broken frames and undeclared tool calls
// never generate successful client responses: undeclared tool calls are rejected as protocol errors;
// broken tool call arguments are flagged; mid-stream failures emit an error event instead of a
// successful stop frame; and corrupted turns are never committed to persistence.
func testB19MalformedToolAndBrokenFrame(t *testing.T) {
	t.Run("UndeclaredToolCallAbortsAsErrorNotSuccessH", testB19UndeclaredToolCallAbortsAsErrorNotSuccessH)
	t.Run("BrokenToolCallArgsRejectedH", testB19BrokenToolCallArgsRejectedH)
	t.Run("CorruptedStreamMidTurnAbortsWithoutStopEventH", testB19CorruptedStreamMidTurnAbortsWithoutStopEventH)
}

// 1. Undeclared tool calls from model must NOT be exposed or silently treated as text end_turn;
// it must be aborted as a protocol error without committing to persistence.
func testB19UndeclaredToolCallAbortsAsErrorNotSuccessH(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	// Upstream model unexpectedly calls "unauthorized_shell" when client only declared "read_file"
	undeclaredToolSSE := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"c_bad","name":"unauthorized_shell","args":{}}}]},"finishReason":"STOP"}]}}`)

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	// A. Non-stream mode: Must return HTTP 502 with error message
	t.Run("NonStream_Returns502", func(t *testing.T) {
		s := New()
		s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: undeclaredToolSSE}}

		body := `{
			"model": "claude-sonnet-4-6",
			"stream": false,
			"tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
			"messages": [{"role": "user", "content": "read please"}]
		}`

		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))

		var u Usage
		status, msg := s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

		if status == http.StatusOK || rec.Code == http.StatusOK {
			t.Fatalf("undeclared tool call illegally produced HTTP 200 success! body=%s", rec.Body.String())
		}
		if !strings.Contains(msg, "undeclared") && !strings.Contains(rec.Body.String(), "undeclared") && !strings.Contains(msg, "tool") {
			t.Errorf("error output should mention undeclared tool call: msg=%q body=%s", msg, rec.Body.String())
		}
	})

	// B. Stream mode: Must NOT emit normal message_stop; must emit error
	t.Run("Stream_EmitsErrorNoStop", func(t *testing.T) {
		s := New()
		s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: undeclaredToolSSE}}

		body := `{
			"model": "claude-sonnet-4-6",
			"stream": true,
			"tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
			"messages": [{"role": "user", "content": "read please"}]
		}`

		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))

		var u Usage
		_, msg := s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

		clientOutput := rec.Body.String()
		if strings.Contains(clientOutput, "message_stop") {
			t.Errorf("stream illegally emitted message_stop for undeclared tool call:\n%s", clientOutput)
		}
		if !strings.Contains(clientOutput, "error") && msg == "" {
			t.Errorf("stream failed to signal error for undeclared tool call:\n%s", clientOutput)
		}
	})
}

// 2. Broken / non-JSON tool call arguments from model must be flagged as protocol errors.
func testB19BrokenToolCallArgsRejectedH(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	// Upstream returns corrupted arguments
	brokenArgsSSE := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"c1","name":"read_file","args":{"path":broken_json}}}]},"finishReason":"STOP"}]}}`)

	s := New()
	s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: brokenArgsSSE}}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	body := `{
		"model": "claude-sonnet-4-6",
		"stream": false,
		"tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
		"messages": [{"role": "user", "content": "hi"}]
	}`

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))

	var u Usage
	status, msg := s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

	// Must fail, never 200
	if status == http.StatusOK || rec.Code == http.StatusOK {
		t.Fatalf("broken tool call args produced HTTP 200! msg=%q body=%s", msg, rec.Body.String())
	}
}

// 3. When an upstream stream breaks mid-turn after text was already sent,
// the gateway must emit an error event and NEVER emit a normal stop frame.
func testB19CorruptedStreamMidTurnAbortsWithoutStopEventH(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	// Stream starts with valid text, then sends broken JSON chunk
	midBreakSSE := sse(
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"Partial text start."}]}}]}}`,
		`data: {corrupted-unparseable-chunk`,
	)

	s := New()
	s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: midBreakSSE}}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	body := `{"model":"claude-sonnet-4-6","stream":true,"messages":[{"role":"user","content":"tell me"}]}`

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))

	var u Usage
	status, msg := s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)
	_ = status

	clientOutput := rec.Body.String()
	// Must NOT send message_stop
	if strings.Contains(clientOutput, "message_stop") {
		t.Errorf("stream illegally emitted normal message_stop after mid-turn corruption:\n%s", clientOutput)
	}

	// Must contain error event or notification
	if !strings.Contains(clientOutput, "error") && !strings.Contains(msg, "error") {
		t.Errorf("mid-stream failure did not deliver error event: msg=%q body=%s", msg, clientOutput)
	}
}
