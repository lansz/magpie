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

// testB03bToolArgsValidationHInvalid covers rejection scenarios of invalid tool arguments:
// ensures HTTP 400, zero upstream calls, and exact path index targeting.
func testB03bToolArgsValidationHInvalid(t *testing.T) {
	t.Run("InvalidToolArgsRejectionAndPathIndex", testB03bHInvalidToolArgsRejectionAndPathIndex)
	t.Run("MultipleCallsSecondBadArgsIndexTargeting", testB03bHMultipleCallsSecondBadArgsIndexTargeting)
}

func testB03bHInvalidToolArgsRejectionAndPathIndex(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	mustEscape := func(s string) string {
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
		wantPathNode string
	}{
		// (a) Anthropic Messages: messages[1].content[0].input
		{
			name:         "Messages_MissingInput",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"fn"}]}]}`,
			wantPathNode: "messages[1].content[0].input",
		},
		{
			name:         "Messages_InputNull",
			from:         provider.Anthropic,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"fn","input":null}]}]}`,
			wantPathNode: "messages[1].content[0].input",
		},
		{
			name:         "Messages_InputArray",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"fn","input":[1,2]}]}]}`,
			wantPathNode: "messages[1].content[0].input",
		},
		{
			name:         "Messages_InputScalarString",
			from:         provider.Anthropic,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"fn","input":"not_an_object"}]}]}`,
			wantPathNode: "messages[1].content[0].input",
		},
		{
			name:         "Messages_InputScalarNumber",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":"run"},{"role":"assistant","content":[{"type":"tool_use","id":"c1","name":"fn","input":123}]}]}`,
			wantPathNode: "messages[1].content[0].input",
		},

		// (b) Chat: messages[1].tool_calls[0].function.arguments
		{
			name:         "Chat_MissingArguments",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn"}}]}]}`,
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},
		{
			name:         "Chat_EmptyString",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":""}}]}]}`,
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},
		{
			name:         "Chat_WhitespaceOnly",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"   \t\n  "}}]}]}`,
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},
		{
			name:         "Chat_OuterNull",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":null}}]}]}`,
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},
		{
			name:         "Chat_InnerNullString",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":"null"}}]}]}`,
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},
		{
			name:         "Chat_InnerArrayString",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":%s}}]}]}`, mustEscape(`[1, 2]`)),
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},
		{
			name:         "Chat_InnerScalarNumberString",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      fmt.Sprintf(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":%s}}]}]}`, mustEscape(`1234`)),
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},
		{
			name:         "Chat_ObjectInsteadOfString",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":{"key":"val"}}}]}]}`,
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},
		{
			name:         "Chat_MalformedInnerJSON",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      fmt.Sprintf(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":%s}}]}]}`, mustEscape(`{"unclosed":`)),
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},
		{
			name:         "Chat_TrailingSecondValue",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"run"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"fn","arguments":%s}}]}]}`, mustEscape(`{"a":1} {"b":2}`)),
			wantPathNode: "messages[1].tool_calls[0].function.arguments",
		},

		// (c) Responses: input[1].arguments
		{
			name:         "Responses_MissingArguments",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"fn"}]}`,
			wantPathNode: "input[1].arguments",
		},
		{
			name:         "Responses_EmptyString",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"fn","arguments":""}]}`,
			wantPathNode: "input[1].arguments",
		},
		{
			name:         "Responses_WhitespaceOnly",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"fn","arguments":"   \n  "}]}`,
			wantPathNode: "input[1].arguments",
		},
		{
			name:         "Responses_OuterNull",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"fn","arguments":null}]}`,
			wantPathNode: "input[1].arguments",
		},
		{
			name:         "Responses_InnerNullString",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"fn","arguments":"null"}]}`,
			wantPathNode: "input[1].arguments",
		},
		{
			name:         "Responses_InnerArrayString",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"fn","arguments":%s}]}`, mustEscape(`[1, 2]`)),
			wantPathNode: "input[1].arguments",
		},
		{
			name:         "Responses_InnerScalarString",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      fmt.Sprintf(`{"model":"claude-sonnet-4-6","input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"fn","arguments":%s}]}`, mustEscape(`"scalar_str"`)),
			wantPathNode: "input[1].arguments",
		},
		{
			name:         "Responses_ObjectInsteadOfString",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"fn","arguments":{"key":"val"}}]}`,
			wantPathNode: "input[1].arguments",
		},
		{
			name:         "Responses_TrailingGarbage",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"run"},{"type":"function_call","call_id":"c1","name":"fn","arguments":%s}]}`, mustEscape(`{"a":1} trailing`)),
			wantPathNode: "input[1].arguments",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Pre-assertion: verify outer payload is 100% valid JSON so we only test inner args validation
			if !json.Valid([]byte(tc.payload)) {
				t.Fatalf("[%s] test payload itself is not valid JSON:\n%s", tc.name, tc.payload)
			}

			s := New()
			tr := &mockCaptureTransport{}
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

			// 1. Verify zero upstream calls without fatal
			if calls := atomic.LoadInt64(&tr.callCount); calls != 0 {
				t.Errorf("[%s] upstream call count must be 0, got %d", tc.name, calls)
			}

			// 2. Assert HTTP 400 rejection
			if status != http.StatusBadRequest {
				t.Errorf("[%s] expected status 400, got %d", tc.name, status)
			}
			if rec.Code != http.StatusBadRequest {
				t.Errorf("[%s] expected rec.Code 400, got %d", tc.name, rec.Code)
			}

			// 3. Verify client-visible error JSON contains the exact path node
			var clientErr struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &clientErr); err != nil {
				t.Errorf("[%s] unmarshal client error JSON failed: %v (raw body: %s)", tc.name, err, rec.Body.String())
			} else if !strings.Contains(clientErr.Error.Message, tc.wantPathNode) {
				t.Errorf("[%s] client error message %q does not contain expected path node %q", tc.name, clientErr.Error.Message, tc.wantPathNode)
			}

			// 4. Also verify internal errMsg contains the path node
			if !strings.Contains(errMsg, tc.wantPathNode) {
				t.Errorf("[%s] internal errMsg %q does not contain expected path node %q", tc.name, errMsg, tc.wantPathNode)
			}
		})
	}
}

// 3. Multiple calls where second call has bad args: index targeting must pinpoint index [1].
func testB03bHMultipleCallsSecondBadArgsIndexTargeting(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	cases := []struct {
		name         string
		from         provider.Protocol
		model        string
		payload      string
		wantPathNode string
	}{
		{
			name:  "Chat_SecondCallBadArgs",
			from:  provider.Chat,
			model: "gemini-3.8-flash-high",
			payload: `{
				"model": "gemini-3.8-flash-high",
				"messages": [
					{"role": "user", "content": "do both"},
					{"role": "assistant", "tool_calls": [
						{"id": "c0", "type": "function", "function": {"name": "fn0", "arguments": "{\"path\":\"a.txt\"}"}},
						{"id": "c1", "type": "function", "function": {"name": "fn1", "arguments": ""}}
					]}
				]
			}`,
			wantPathNode: "messages[1].tool_calls[1].function.arguments",
		},
		{
			name:  "Messages_SecondCallBadArgs",
			from:  provider.Anthropic,
			model: "claude-sonnet-4-6",
			payload: `{
				"model": "claude-sonnet-4-6",
				"max_tokens": 100,
				"messages": [
					{"role": "user", "content": "do both"},
					{"role": "assistant", "content": [
						{"type": "tool_use", "id": "c0", "name": "fn0", "input": {"path": "a.txt"}},
						{"type": "tool_use", "id": "c1", "name": "fn1", "input": null}
					]}
				]
			}`,
			wantPathNode: "messages[1].content[1].input",
		},
		{
			name:  "Responses_SecondCallBadArgs",
			from:  provider.Responses,
			model: "gemini-3.8-flash-high",
			payload: `{
				"model": "gemini-3.8-flash-high",
				"input": [
					{"role": "user", "content": "do both"},
					{"type": "function_call", "call_id": "c0", "name": "fn0", "arguments": "{\"path\":\"a.txt\"}"},
					{"type": "function_call", "call_id": "c1", "name": "fn1", "arguments": "   "}
				]
			}`,
			wantPathNode: "input[2].arguments",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !json.Valid([]byte(tc.payload)) {
				t.Fatalf("[%s] test payload is not valid JSON:\n%s", tc.name, tc.payload)
			}

			s := New()
			tr := &mockCaptureTransport{}
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

			if calls := atomic.LoadInt64(&tr.callCount); calls != 0 {
				t.Errorf("[%s] upstream call count must be 0, got %d", tc.name, calls)
			}
			if status != http.StatusBadRequest {
				t.Errorf("[%s] expected status 400, got %d", tc.name, status)
			}
			if rec.Code != http.StatusBadRequest {
				t.Errorf("[%s] expected rec.Code 400, got %d", tc.name, rec.Code)
			}

			var clientErr struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &clientErr); err != nil {
				t.Errorf("[%s] unmarshal client error JSON failed: %v", tc.name, err)
			} else if !strings.Contains(clientErr.Error.Message, tc.wantPathNode) {
				t.Errorf("[%s] client error %q does not contain targeted path node %q", tc.name, clientErr.Error.Message, tc.wantPathNode)
			}

			if !strings.Contains(errMsg, tc.wantPathNode) {
				t.Errorf("[%s] internal errMsg %q does not contain targeted path node %q", tc.name, errMsg, tc.wantPathNode)
			}
		})
	}
}
