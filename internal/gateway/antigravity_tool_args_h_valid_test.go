package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03bToolArgsValidationHValid covers positive H-layer scenarios:
// Antigravity valid multi-tool call roundtrip (with fidelity on wire) and normal text requests.
func testB03bToolArgsValidationHValid(t *testing.T) {
	t.Run("ValidMultiToolCallsRoundtrip", testB03bHValidMultiToolCallsRoundtrip)
	t.Run("NormalRequestsWithoutToolCallNotDamaged", testB03bHNormalRequestsNotDamaged)
}

// 1. Antigravity valid multi-tool calls through s.translate:
// covers {}, business keys (including arguments), and >2^53 integers in a single multi-call turn.
// Asserts exact wire count, order, name, and argument fidelity, plus client response.
func testB03bHValidMultiToolCallsRoundtrip(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"tools executed ok"}]},"finishReason":"STOP"}]}}`,
	)

	const (
		args1 = `{}`
		args2 = `{"arguments":"raw_cmd_args","cache_control":{"type":"ephemeral"},"default":"val","input":"data","title":"run"}`
		args3 = `{"id_above_53":9007199254740993}`
	)

	mustMarshalString := func(s string) string {
		b, err := json.Marshal(s)
		if err != nil {
			t.Fatalf("marshal string failed: %v", err)
		}
		return string(b)
	}

	cases := []struct {
		name         string
		from         provider.Protocol
		model        string
		payload      string
		assertClient func(t *testing.T, body []byte)
	}{
		// (a) Messages on Gemini
		{
			name:  "Messages_Gemini",
			from:  provider.Anthropic,
			model: "gemini-3.8-flash-high",
			payload: fmt.Sprintf(`{
				"model": "gemini-3.8-flash-high",
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": "do multi tools"},
					{"role": "assistant", "content": [
						{"type": "tool_use", "id": "c1", "name": "fn1", "input": %s},
						{"type": "tool_use", "id": "c2", "name": "fn2", "input": %s},
						{"type": "tool_use", "id": "c3", "name": "fn3", "input": %s}
					]}
				]
			}`, args1, args2, args3),
			assertClient: func(t *testing.T, body []byte) {
				var rep struct {
					Content []struct {
						Text string `json:"text"`
					} `json:"content"`
					StopReason string `json:"stop_reason"`
				}
				if err := json.Unmarshal(body, &rep); err != nil {
					t.Fatalf("unmarshal Messages client response failed: %v", err)
				}
				if len(rep.Content) == 0 || rep.Content[0].Text != "tools executed ok" || rep.StopReason != "end_turn" {
					t.Errorf("Messages client response mismatch: %+v", rep)
				}
			},
		},
		// (b) Chat on Claude
		{
			name:  "Chat_Claude",
			from:  provider.Chat,
			model: "claude-sonnet-4-6",
			payload: fmt.Sprintf(`{
				"model": "claude-sonnet-4-6",
				"messages": [
					{"role": "user", "content": "do multi tools"},
					{"role": "assistant", "tool_calls": [
						{"id": "c1", "type": "function", "function": {"name": "fn1", "arguments": %s}},
						{"id": "c2", "type": "function", "function": {"name": "fn2", "arguments": %s}},
						{"id": "c3", "type": "function", "function": {"name": "fn3", "arguments": %s}}
					]}
				]
			}`, mustMarshalString(args1), mustMarshalString(args2), mustMarshalString(args3)),
			assertClient: func(t *testing.T, body []byte) {
				var rep struct {
					Choices []struct {
						Message struct {
							Content string `json:"content"`
						} `json:"message"`
						FinishReason string `json:"finish_reason"`
					} `json:"choices"`
				}
				if err := json.Unmarshal(body, &rep); err != nil {
					t.Fatalf("unmarshal Chat client response failed: %v", err)
				}
				if len(rep.Choices) == 0 || rep.Choices[0].Message.Content != "tools executed ok" || rep.Choices[0].FinishReason != "stop" {
					t.Errorf("Chat client response mismatch: %+v", rep)
				}
			},
		},
		// (c) Responses on Gemini
		{
			name:  "Responses_Gemini",
			from:  provider.Responses,
			model: "gemini-3.8-flash-high",
			payload: fmt.Sprintf(`{
				"model": "gemini-3.8-flash-high",
				"input": [
					{"role": "user", "content": "do multi tools"},
					{"type": "function_call", "call_id": "c1", "name": "fn1", "arguments": %s},
					{"type": "function_call", "call_id": "c2", "name": "fn2", "arguments": %s},
					{"type": "function_call", "call_id": "c3", "name": "fn3", "arguments": %s}
				]
			}`, mustMarshalString(args1), mustMarshalString(args2), mustMarshalString(args3)),
			assertClient: func(t *testing.T, body []byte) {
				var rep struct {
					Status string `json:"status"`
					Output []struct {
						Content []struct {
							Text string `json:"text"`
						} `json:"content"`
					} `json:"output"`
				}
				if err := json.Unmarshal(body, &rep); err != nil {
					t.Fatalf("unmarshal Responses client response failed: %v", err)
				}
				if rep.Status != "completed" || len(rep.Output) == 0 || len(rep.Output[0].Content) == 0 || rep.Output[0].Content[0].Text != "tools executed ok" {
					t.Errorf("Responses client response mismatch: %+v", rep)
				}
			},
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
			if status != http.StatusOK {
				t.Fatalf("[%s] translate status %d: %s", tc.name, status, errMsg)
			}
			if rec.Code != http.StatusOK {
				t.Errorf("[%s] expected rec.Code 200, got %d", tc.name, rec.Code)
			}
			if errMsg != "" {
				t.Errorf("[%s] unexpected errMsg: %s", tc.name, errMsg)
			}
			if atomic.LoadInt64(&tr.callCount) != 1 {
				t.Fatalf("[%s] expected 1 upstream call, got %d", tc.name, tr.callCount)
			}

			tc.assertClient(t, rec.Body.Bytes())

			// Verify wire captured calls: exactly 3 calls in expected order with exact names and args
			calls := extractAllFunctionCalls(t, tr.capturedBody)
			if len(calls) != 3 {
				t.Fatalf("[%s] expected 3 function calls on wire, got %d", tc.name, len(calls))
			}
			expectedNames := []string{"fn1", "fn2", "fn3"}
			expectedArgs := []string{args1, args2, args3}
			for i := range expectedArgs {
				if calls[i].Name != expectedNames[i] {
					t.Errorf("[%s] wire call %d name mismatch: got %q, want %q", tc.name, i, calls[i].Name, expectedNames[i])
				}
				eq, err := jsonEqualExact(calls[i].Args, []byte(expectedArgs[i]))
				if err != nil || !eq {
					t.Errorf("[%s] wire call %d args mismatch:\ngot:  %s\nwant: %s (err: %v)", tc.name, i, string(calls[i].Args), expectedArgs[i], err)
				}
			}
		})
	}
}

// 2. Normal requests without tool calls and messages mentioning 'arguments'/'input' must pass undamaged.
func testB03bHNormalRequestsNotDamaged(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello back"}]},"finishReason":"STOP"}]}}`)

	cases := []struct {
		name         string
		from         provider.Protocol
		model        string
		payload      string
		assertClient func(t *testing.T, body []byte)
		assertWire   func(t *testing.T, wire []byte)
	}{
		{
			name:    "Messages_PlainWithKeywords",
			from:    provider.Anthropic,
			model:   "gemini-3.8-flash-high",
			payload: `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":"Please check arguments and input schema."}]}`,
			assertClient: func(t *testing.T, body []byte) {
				var rep struct {
					Content []struct {
						Text string `json:"text"`
					} `json:"content"`
					StopReason string `json:"stop_reason"`
				}
				if err := json.Unmarshal(body, &rep); err != nil {
					t.Fatalf("unmarshal Messages client failed: %v", err)
				}
				if len(rep.Content) == 0 || rep.Content[0].Text != "hello back" || rep.StopReason != "end_turn" {
					t.Errorf("Messages client reply mismatch: %+v", rep)
				}
			},
		},
		{
			name:    "Chat_PlainWithKeywords",
			from:    provider.Chat,
			model:   "claude-sonnet-4-6",
			payload: `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Review arguments and input carefully."}]}`,
			assertClient: func(t *testing.T, body []byte) {
				var rep struct {
					Choices []struct {
						Message struct {
							Content string `json:"content"`
						} `json:"message"`
						FinishReason string `json:"finish_reason"`
					} `json:"choices"`
				}
				if err := json.Unmarshal(body, &rep); err != nil {
					t.Fatalf("unmarshal Chat client failed: %v", err)
				}
				if len(rep.Choices) == 0 || rep.Choices[0].Message.Content != "hello back" || rep.Choices[0].FinishReason != "stop" {
					t.Errorf("Chat client reply mismatch: %+v", rep)
				}
			},
		},
		{
			name:    "Responses_PlainWithKeywords",
			from:    provider.Responses,
			model:   "gemini-3.8-flash-high",
			payload: `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"Examine the input and arguments."}]}`,
			assertClient: func(t *testing.T, body []byte) {
				var rep struct {
					Status string `json:"status"`
					Output []struct {
						Content []struct {
							Text string `json:"text"`
						} `json:"content"`
					} `json:"output"`
				}
				if err := json.Unmarshal(body, &rep); err != nil {
					t.Fatalf("unmarshal Responses client failed: %v", err)
				}
				if rep.Status != "completed" || len(rep.Output) == 0 || len(rep.Output[0].Content) == 0 || rep.Output[0].Content[0].Text != "hello back" {
					t.Errorf("Responses client reply mismatch: %+v", rep)
				}
			},
		},
		// Chat user message carrying unexpected tool_calls extension (legacy ignored): must not be rejected by assistant tool validation
		{
			name:    "Chat_UserMessageWithToolCallsExtension_NotRejected",
			from:    provider.Chat,
			model:   "gemini-3.8-flash-high",
			payload: `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"user text preserved","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"bad not json"}}]}]}`,
			assertClient: func(t *testing.T, body []byte) {
				var rep struct {
					Choices []struct {
						Message struct {
							Content string `json:"content"`
						} `json:"message"`
						FinishReason string `json:"finish_reason"`
					} `json:"choices"`
				}
				if err := json.Unmarshal(body, &rep); err != nil {
					t.Fatalf("unmarshal Chat client failed: %v", err)
				}
				if len(rep.Choices) == 0 || rep.Choices[0].Message.Content != "hello back" || rep.Choices[0].FinishReason != "stop" {
					t.Errorf("Chat client reply mismatch: %+v", rep)
				}
			},
			assertWire: func(t *testing.T, wire []byte) {
				// Assert wire does not produce functionCall from user message and preserves user text
				calls := extractAllFunctionCalls(t, wire)
				if len(calls) != 0 {
					t.Errorf("expected 0 functionCall on wire for user message tool_calls, got %d", len(calls))
				}
				if !strings.Contains(string(wire), "user text preserved") {
					t.Errorf("wire body does not contain user text: %s", string(wire))
				}
			},
		},
		// Responses top-level string input: must pass without error and preserve string content
		{
			name:    "Responses_TopLevelStringInput",
			from:    provider.Responses,
			model:   "claude-sonnet-4-6",
			payload: `{"model":"claude-sonnet-4-6","input":"just a top level string prompt"}`,
			assertClient: func(t *testing.T, body []byte) {
				var rep struct {
					Status string `json:"status"`
					Output []struct {
						Content []struct {
							Text string `json:"text"`
						} `json:"content"`
					} `json:"output"`
				}
				if err := json.Unmarshal(body, &rep); err != nil {
					t.Fatalf("unmarshal Responses client failed: %v", err)
				}
				if rep.Status != "completed" || len(rep.Output) == 0 || len(rep.Output[0].Content) == 0 || rep.Output[0].Content[0].Text != "hello back" {
					t.Errorf("Responses client reply mismatch: %+v", rep)
				}
			},
			assertWire: func(t *testing.T, wire []byte) {
				if !strings.Contains(string(wire), "just a top level string prompt") {
					t.Errorf("wire body does not contain top level string prompt: %s", string(wire))
				}
			},
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
			if status != http.StatusOK {
				t.Fatalf("[%s] expected HTTP 200, got status %d, errMsg: %s", tc.name, status, errMsg)
			}
			if rec.Code != http.StatusOK {
				t.Errorf("[%s] expected rec.Code 200, got %d", tc.name, rec.Code)
			}
			if errMsg != "" {
				t.Errorf("[%s] unexpected errMsg: %s", tc.name, errMsg)
			}
			if atomic.LoadInt64(&tr.callCount) != 1 {
				t.Errorf("[%s] expected 1 upstream call, got %d", tc.name, tr.callCount)
			}

			tc.assertClient(t, rec.Body.Bytes())
			if tc.assertWire != nil {
				tc.assertWire(t, tr.capturedBody)
			}
		})
	}
}
