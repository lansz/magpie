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

// testB03OutputTokenLimitNonTarget verifies non-target providers (Ordinary Chat and Gemini CLI)
// preserve their exact existing budget wire behaviors across positive (2000), 0, and absent.
func testB03OutputTokenLimitNonTarget(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	// 1. Non-target Ordinary Chat Provider (p.Account == nil)
	t.Run("OrdinaryChatProvider", func(t *testing.T) {
		sseChunk := sse(`data: {"id":"c","choices":[{"delta":{"content":"ok"},"finish_reason":null}]}`, `data: [DONE]`)

		cases := []struct {
			name       string
			payload    string
			assertWire func(t *testing.T, raw []byte)
		}{
			{
				name:    "Positive_2000",
				payload: `{"model":"m1","max_tokens":2000,"messages":[{"role":"user","content":"hi"}]}`,
				assertWire: func(t *testing.T, raw []byte) {
					var chatReq struct {
						MaxTokens *int `json:"max_tokens"`
					}
					if err := json.Unmarshal(raw, &chatReq); err != nil {
						t.Fatalf("unmarshal chat wire failed: %v", err)
					}
					if chatReq.MaxTokens == nil || *chatReq.MaxTokens != 2000 {
						t.Errorf("expected wire max_tokens 2000, got: %v", chatReq.MaxTokens)
					}
				},
			},
			{
				name:    "Zero_Omitted",
				payload: `{"model":"m1","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}`,
				assertWire: func(t *testing.T, raw []byte) {
					var rawMap map[string]json.RawMessage
					if err := json.Unmarshal(raw, &rawMap); err != nil {
						t.Fatalf("unmarshal chat wire failed: %v", err)
					}
					if _, exists := rawMap["max_tokens"]; exists {
						t.Errorf("expected max_tokens omitted on wire for zero value in ordinary chat, but exists")
					}
				},
			},
			{
				name:    "Absent_Omitted",
				payload: `{"model":"m1","messages":[{"role":"user","content":"hi"}]}`,
				assertWire: func(t *testing.T, raw []byte) {
					var rawMap map[string]json.RawMessage
					if err := json.Unmarshal(raw, &rawMap); err != nil {
						t.Fatalf("unmarshal chat wire failed: %v", err)
					}
					if _, exists := rawMap["max_tokens"]; exists {
						t.Errorf("expected max_tokens omitted on wire for absent request in ordinary chat, but exists")
					}
				},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
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

				rec := httptest.NewRecorder()
				httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(tc.payload)))
				httpReq.Header.Set("Content-Type", "application/json")

				var u Usage
				status, errMsg := s.translate(rec, httpReq, p, provider.Anthropic, provider.Chat, "m1", []byte(tc.payload), &u)
				if status != http.StatusOK {
					t.Fatalf("[%s] expected HTTP 200, got status %d: %s", tc.name, status, rec.Body.String())
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

				tc.assertWire(t, tr.capturedBody)
			})
		}
	})

	// 2. Non-target Gemini CLI (p.Account.Agent == "gemini")
	t.Run("GeminiCLI", func(t *testing.T) {
		codeAssistSSE := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"cli ok"}]},"finishReason":"STOP"}]}}`)

		cases := []struct {
			name       string
			payload    string
			assertWire func(t *testing.T, raw []byte)
		}{
			{
				name:    "Positive_2000",
				payload: `{"model":"gemini-2.5-pro","max_tokens":2000,"messages":[{"role":"user","content":"hi"}]}`,
				assertWire: func(t *testing.T, raw []byte) {
					var env typedGenConfigEnv
					if err := json.Unmarshal(raw, &env); err != nil {
						t.Fatalf("unmarshal gemini wire failed: %v", err)
					}
					if env.Request.GenerationConfig.MaxOutputTokens == nil || *env.Request.GenerationConfig.MaxOutputTokens != 2000 {
						t.Errorf("expected gemini CLI wire maxOutputTokens 2000, got: %v", env.Request.GenerationConfig.MaxOutputTokens)
					}
				},
			},
			{
				name:    "Zero_Omitted",
				payload: `{"model":"gemini-2.5-pro","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}`,
				assertWire: func(t *testing.T, raw []byte) {
					var env rawGenConfigEnv
					if err := json.Unmarshal(raw, &env); err != nil {
						t.Fatalf("unmarshal gemini wire failed: %v", err)
					}
					if env.Request.GenerationConfig != nil {
						if _, exists := env.Request.GenerationConfig["maxOutputTokens"]; exists {
							t.Errorf("expected maxOutputTokens omitted on wire for zero value in gemini CLI, but exists")
						}
					}
				},
			},
			{
				name:    "Absent_Omitted",
				payload: `{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"hi"}]}`,
				assertWire: func(t *testing.T, raw []byte) {
					var env rawGenConfigEnv
					if err := json.Unmarshal(raw, &env); err != nil {
						t.Fatalf("unmarshal gemini wire failed: %v", err)
					}
					if env.Request.GenerationConfig != nil {
						if _, exists := env.Request.GenerationConfig["maxOutputTokens"]; exists {
							t.Errorf("expected maxOutputTokens omitted on wire for absent request in gemini CLI, but exists")
						}
					}
				},
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
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

				rec := httptest.NewRecorder()
				httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(tc.payload)))
				httpReq.Header.Set("Content-Type", "application/json")

				var u Usage
				status, errMsg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "gemini-2.5-pro", []byte(tc.payload), &u)
				if status != http.StatusOK {
					t.Fatalf("[%s] expected HTTP 200, got status %d: %s", tc.name, status, rec.Body.String())
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

				tc.assertWire(t, tr.capturedBody)
			})
		}
	})
}
