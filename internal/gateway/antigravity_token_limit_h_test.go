package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03OutputTokenLimitH covers all H-layer B03a scenarios.
func testB03OutputTokenLimitH(t *testing.T) {
	t.Run("ValidLimitWireThroughput", testB03HValidLimitWireThroughput)
	t.Run("AbsentLimitWireThroughput", testB03HAbsentLimitWireThroughput)
	t.Run("ChatDualFieldWirePriority", testB03HChatDualFieldWirePriority)
	t.Run("InvalidLimitRejectionZeroCall", testB03HInvalidLimitRejectionZeroCall)
}

type mockCaptureTransport struct {
	capturedBody []byte
	callCount    int64
	sseReply     string
}

func (m *mockCaptureTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	atomic.AddInt64(&m.callCount, 1)
	if req.Body != nil {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		m.capturedBody = body
	}
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(m.sseReply)),
	}
	resp.Header.Set("Content-Type", "text/event-stream")
	return resp, nil
}

// 1. Valid token limits wire capture and client roundtrip through s.translate.
func testB03HValidLimitWireThroughput(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	// Legal CodeAssist SSE response with proper "data: " prefix
	sseChunk := sse(
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello back"}]},"finishReason":"STOP"}]}}`,
	)

	cases := []struct {
		name         string
		from         provider.Protocol
		model        string
		payload      string
		wantLimit    int
		assertClient func(t *testing.T, body []byte)
	}{
		{
			name:      "Messages_Gemini_2048",
			from:      provider.Anthropic,
			model:     "gemini-3.8-flash-high",
			payload:   `{"model":"gemini-3.8-flash-high","max_tokens":2048,"messages":[{"role":"user","content":"hi"}]}`,
			wantLimit: 2048,
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
				if len(rep.Content) == 0 || rep.Content[0].Text != "hello back" || rep.StopReason != "end_turn" {
					t.Errorf("Messages client response mismatch: %+v", rep)
				}
			},
		},
		{
			name:      "Chat_Claude_128",
			from:      provider.Chat,
			model:     "claude-sonnet-4-6",
			payload:   `{"model":"claude-sonnet-4-6","max_completion_tokens":128,"messages":[{"role":"user","content":"hi"}]}`,
			wantLimit: 128,
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
				if len(rep.Choices) == 0 || rep.Choices[0].Message.Content != "hello back" || rep.Choices[0].FinishReason != "stop" {
					t.Errorf("Chat client response mismatch: %+v", rep)
				}
			},
		},
		{
			name:      "Chat_SingleMaxTokens_Gemini_128",
			from:      provider.Chat,
			model:     "gemini-3.8-flash-high",
			payload:   `{"model":"gemini-3.8-flash-high","max_tokens":128,"messages":[{"role":"user","content":"hi"}]}`,
			wantLimit: 128,
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
				if len(rep.Choices) == 0 || rep.Choices[0].Message.Content != "hello back" || rep.Choices[0].FinishReason != "stop" {
					t.Errorf("Chat client response mismatch: %+v", rep)
				}
			},
		},
		{
			name:      "Responses_Gemini_2048",
			from:      provider.Responses,
			model:     "gemini-3.8-flash-high",
			payload:   `{"model":"gemini-3.8-flash-high","max_output_tokens":2048,"input":[{"role":"user","content":"hi"}]}`,
			wantLimit: 2048,
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
				if rep.Status != "completed" || len(rep.Output) == 0 || len(rep.Output[0].Content) == 0 || rep.Output[0].Content[0].Text != "hello back" {
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
				ID:     "antigravity-test",
				Name:   "AntigravityProvider",
				Key:    "dummy-key",
				Models: []string{tc.model},
				Account: &provider.Account{
					Agent: "antigravity",
				},
			}

			rec := httptest.NewRecorder()
			httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(tc.payload)))
			httpReq.Header.Set("Content-Type", "application/json")

			var u Usage
			status, errMsg := s.translate(rec, httpReq, targetProvider, tc.from, provider.CodeAssist, tc.model, []byte(tc.payload), &u)
			if status != http.StatusOK {
				t.Fatalf("translate returned status %d, errMsg: %s, body: %s", status, errMsg, rec.Body.String())
			}
			if rec.Code != http.StatusOK {
				t.Errorf("[%s] expected rec.Code 200, got %d", tc.name, rec.Code)
			}
			if errMsg != "" {
				t.Errorf("[%s] unexpected errMsg: %s", tc.name, errMsg)
			}
			if atomic.LoadInt64(&tr.callCount) != 1 {
				t.Fatalf("expected 1 upstream call, got %d", tr.callCount)
			}

			// Verify client response parsing
			tc.assertClient(t, rec.Body.Bytes())

			// Verify wire generationConfig.maxOutputTokens
			var typed typedGenConfigEnv
			if err := json.Unmarshal(tr.capturedBody, &typed); err != nil {
				t.Fatalf("unmarshal wire failed: %v\nBody: %s", err, string(tr.capturedBody))
			}
			if typed.Request.GenerationConfig.MaxOutputTokens == nil {
				t.Fatalf("upstream wire body missing maxOutputTokens")
			}
			if *typed.Request.GenerationConfig.MaxOutputTokens != tc.wantLimit {
				t.Errorf("wire maxOutputTokens mismatch: got %d, want %d", *typed.Request.GenerationConfig.MaxOutputTokens, tc.wantLimit)
			}
		})
	}
}

// 2. Absent limits across three protocols must not be present on wire.
func testB03HAbsentLimitWireThroughput(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}}`)

	cases := []struct {
		name    string
		from    provider.Protocol
		payload string
	}{
		{"Messages_Absent", provider.Anthropic, `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"hi"}]}`},
		{"Chat_Absent", provider.Chat, `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":"hi"}]}`},
		{"Responses_Absent", provider.Responses, `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":"hi"}]}`},
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
			status, errMsg := s.translate(rec, httpReq, targetProvider, tc.from, provider.CodeAssist, "gemini-3.8-flash-high", []byte(tc.payload), &u)
			if status != http.StatusOK {
				t.Fatalf("translate status %d: %s", status, errMsg)
			}
			if rec.Code != http.StatusOK {
				t.Errorf("[%s] expected rec.Code 200, got %d", tc.name, rec.Code)
			}
			if errMsg != "" {
				t.Errorf("[%s] unexpected errMsg: %s", tc.name, errMsg)
			}
			if calls := atomic.LoadInt64(&tr.callCount); calls != 1 {
				t.Errorf("[%s] expected 1 upstream call, got %d", tc.name, calls)
			}

			var raw rawGenConfigEnv
			if err := json.Unmarshal(tr.capturedBody, &raw); err != nil {
				t.Fatalf("unmarshal raw wire failed: %v", err)
			}
			if raw.Request.GenerationConfig != nil {
				if _, exists := raw.Request.GenerationConfig["maxOutputTokens"]; exists {
					t.Errorf("expected absent maxOutputTokens on wire for %s, but key exists", tc.name)
				}
			}
		})
	}
}

// 3. Chat dual field priority to wire: max_completion_tokens (128) beats max_tokens (2048).
func testB03HChatDualFieldWirePriority(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}}`)

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

	payload := `{"model":"gemini-3.8-flash-high","max_completion_tokens":128,"max_tokens":2048,"messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(payload)))
	httpReq.Header.Set("Content-Type", "application/json")

	var u Usage
	status, errMsg := s.translate(rec, httpReq, targetProvider, provider.Chat, provider.CodeAssist, "gemini-3.8-flash-high", []byte(payload), &u)
	if status != http.StatusOK {
		t.Fatalf("translate status %d: %s", status, errMsg)
	}
	if rec.Code != http.StatusOK {
		t.Errorf("expected rec.Code 200, got %d", rec.Code)
	}
	if errMsg != "" {
		t.Errorf("unexpected errMsg: %s", errMsg)
	}
	if calls := atomic.LoadInt64(&tr.callCount); calls != 1 {
		t.Errorf("expected 1 upstream call, got %d", calls)
	}

	var typed typedGenConfigEnv
	if err := json.Unmarshal(tr.capturedBody, &typed); err != nil {
		t.Fatalf("unmarshal wire failed: %v", err)
	}
	if typed.Request.GenerationConfig.MaxOutputTokens == nil || *typed.Request.GenerationConfig.MaxOutputTokens != 128 {
		t.Errorf("wire maxOutputTokens priority mismatch: got %v, want 128", typed.Request.GenerationConfig.MaxOutputTokens)
	}
}

// 4. Invalid token limit validation: guarantees HTTP 400, client-visible error field, and zero upstream calls.
func testB03HInvalidLimitRejectionZeroCall(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	cases := []struct {
		name      string
		from      provider.Protocol
		model     string
		field     string
		payload   string
		errSubstr string
	}{
		// Messages: 0, null, 负数
		{"Messages_Zero", provider.Anthropic, "gemini-3.8-flash-high", "max_tokens", `{"model":"gemini-3.8-flash-high","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},
		{"Messages_Null", provider.Anthropic, "claude-sonnet-4-6", "max_tokens", `{"model":"claude-sonnet-4-6","max_tokens":null,"messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},
		{"Messages_Negative", provider.Anthropic, "gemini-3.8-flash-high", "max_tokens", `{"model":"gemini-3.8-flash-high","max_tokens":-1,"messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},

		// Chat max_completion_tokens: 0, null, 负数
		{"Chat_MaxComp_Zero", provider.Chat, "gemini-3.8-flash-high", "max_completion_tokens", `{"model":"gemini-3.8-flash-high","max_completion_tokens":0,"messages":[{"role":"user","content":"hi"}]}`, "max_completion_tokens"},
		{"Chat_MaxComp_Null", provider.Chat, "claude-sonnet-4-6", "max_completion_tokens", `{"model":"claude-sonnet-4-6","max_completion_tokens":null,"messages":[{"role":"user","content":"hi"}]}`, "max_completion_tokens"},
		{"Chat_MaxComp_Negative", provider.Chat, "gemini-3.8-flash-high", "max_completion_tokens", `{"model":"gemini-3.8-flash-high","max_completion_tokens":-5,"messages":[{"role":"user","content":"hi"}]}`, "max_completion_tokens"},

		// Chat max_tokens: 0, null, 负数
		{"Chat_MaxTokens_Zero", provider.Chat, "gemini-3.8-flash-high", "max_tokens", `{"model":"gemini-3.8-flash-high","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},
		{"Chat_MaxTokens_Null", provider.Chat, "claude-sonnet-4-6", "max_tokens", `{"model":"claude-sonnet-4-6","max_tokens":null,"messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},
		{"Chat_MaxTokens_Negative", provider.Chat, "gemini-3.8-flash-high", "max_tokens", `{"model":"gemini-3.8-flash-high","max_tokens":-1,"messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},

		// Responses max_output_tokens: 0, null, 负数
		{"Responses_Zero", provider.Responses, "gemini-3.8-flash-high", "max_output_tokens", `{"model":"gemini-3.8-flash-high","max_output_tokens":0,"input":[{"role":"user","content":"hi"}]}`, "max_output_tokens"},
		{"Responses_Null", provider.Responses, "claude-sonnet-4-6", "max_output_tokens", `{"model":"claude-sonnet-4-6","max_output_tokens":null,"input":[{"role":"user","content":"hi"}]}`, "max_output_tokens"},
		{"Responses_Negative", provider.Responses, "gemini-3.8-flash-high", "max_output_tokens", `{"model":"gemini-3.8-flash-high","max_output_tokens":-1,"input":[{"role":"user","content":"hi"}]}`, "max_output_tokens"},

		// 代表性格式错误: 字符串、小数、溢出
		{"Messages_String", provider.Anthropic, "claude-sonnet-4-6", "max_tokens", `{"model":"claude-sonnet-4-6","max_tokens":"128","messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},
		{"Chat_Float", provider.Chat, "gemini-3.8-flash-high", "max_tokens", `{"model":"gemini-3.8-flash-high","max_tokens":128.5,"messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},
		{"Responses_Overflow", provider.Responses, "gemini-3.8-flash-high", "max_output_tokens", `{"model":"gemini-3.8-flash-high","max_output_tokens":99999999999999999999,"input":[{"role":"user","content":"hi"}]}`, "max_output_tokens"},

		// Chat 双字段交叉: 一合法一非法时显式拒绝 (双向组合)
		{"Chat_Dual_MaxCompInvalid_MaxTokensValid", provider.Chat, "gemini-3.8-flash-high", "max_completion_tokens", `{"model":"gemini-3.8-flash-high","max_completion_tokens":-1,"max_tokens":2048,"messages":[{"role":"user","content":"hi"}]}`, "max_completion_tokens"},
		{"Chat_Dual_MaxCompValid_MaxTokensInvalid", provider.Chat, "gemini-3.8-flash-high", "max_tokens", `{"model":"gemini-3.8-flash-high","max_completion_tokens":128,"max_tokens":-1,"messages":[{"role":"user","content":"hi"}]}`, "max_tokens"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
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

			// 1. Verify zero upstream calls without fatal exit
			calls := atomic.LoadInt64(&tr.callCount)
			if calls != 0 {
				t.Errorf("[%s] upstream call count must be 0, got %d", tc.name, calls)
			}

			// 2. Assert HTTP 400 rejection
			if status != http.StatusBadRequest {
				t.Errorf("[%s] expected status 400, got %d", tc.name, status)
			}
			if rec.Code != http.StatusBadRequest {
				t.Errorf("[%s] expected rec.Code 400, got %d", tc.name, rec.Code)
			}

			// 3. Verify error message points out the field directly on client-visible error JSON
			var clientErr struct {
				Error struct {
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &clientErr); err != nil {
				t.Errorf("[%s] failed to unmarshal client error JSON: %v (raw: %s)", tc.name, err, rec.Body.String())
			} else if !strings.Contains(clientErr.Error.Message, tc.errSubstr) {
				t.Errorf("[%s] client error message %q does not contain expected field %q", tc.name, clientErr.Error.Message, tc.errSubstr)
			}

			// 4. Also verify internal errMsg contains the field as supplementary check
			if !strings.Contains(errMsg, tc.errSubstr) {
				t.Errorf("[%s] internal errMsg %q does not contain expected field %q", tc.name, errMsg, tc.errSubstr)
			}
		})
	}
}
