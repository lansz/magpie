package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03cImageValidationHValid covers positive H-layer scenarios:
// valid alternating image roundtrip and remote URLs across 3 protocols.
func testB03cImageValidationHValid(t *testing.T) {
	t.Run("ValidAlternatingImageRoundtrip", testB03cHValidAlternatingImageRoundtrip)
	t.Run("RemoteURLThroughputAcrossProtocols", testB03cHRemoteURLThroughputAcrossProtocols)
}

// 1. Valid alternating [text, img, text, img, text] roundtrip through s.translate with full client parsing.
func testB03cHValidAlternatingImageRoundtrip(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"images processed ok"}]},"finishReason":"STOP"}]}}`,
	)

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
					{"role": "user", "content": [
						{"type": "text", "text": "t1"},
						{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": %q}},
						{"type": "text", "text": "t2"},
						{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": %q}},
						{"type": "text", "text": "t3"}
					]}
				]
			}`, samplePNGBase64, sampleJPEGBase64),
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
				if len(rep.Content) == 0 || rep.Content[0].Text != "images processed ok" || rep.StopReason != "end_turn" {
					t.Errorf("Messages client mismatch: %+v", rep)
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
					{"role": "user", "content": [
						{"type": "text", "text": "t1"},
						{"type": "image_url", "image_url": {"url": %q}},
						{"type": "text", "text": "t2"},
						{"type": "image_url", "image_url": {"url": %q}},
						{"type": "text", "text": "t3"}
					]}
				]
			}`, samplePNGDataURL, sampleJPEGDataURL),
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
				if len(rep.Choices) == 0 || rep.Choices[0].Message.Content != "images processed ok" || rep.Choices[0].FinishReason != "stop" {
					t.Errorf("Chat client mismatch: %+v", rep)
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
					{"role": "user", "content": [
						{"type": "input_text", "text": "t1"},
						{"type": "input_image", "image_url": %q},
						{"type": "input_text", "text": "t2"},
						{"type": "input_image", "image_url": %q},
						{"type": "input_text", "text": "t3"}
					]}
				]
			}`, samplePNGDataURL, sampleJPEGDataURL),
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
				if rep.Status != "completed" || len(rep.Output) == 0 || len(rep.Output[0].Content) == 0 || rep.Output[0].Content[0].Text != "images processed ok" {
					t.Errorf("Responses client mismatch: %+v", rep)
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

			var env codeAssistWireEnvelope
			if err := json.Unmarshal(tr.capturedBody, &env); err != nil {
				t.Fatalf("unmarshal wire body failed: %v", err)
			}
			if len(env.Request.Contents) != 1 || env.Request.Contents[0].Role != "user" {
				t.Fatalf("expected 1 user content on wire, got: %+v", env.Request.Contents)
			}
			if len(env.Request.Contents[0].Parts) != 5 {
				t.Fatalf("expected 5 wire parts, got: %+v", env.Request.Contents[0].Parts)
			}
			assertAlternatingWireParts(t, env.Request.Contents[0].Parts)
		})
	}
}

// 2. Remote ordinary URL table-driven throughput across all 3 protocols (no network fetch).
func testB03cHRemoteURLThroughputAcrossProtocols(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sseChunk := sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"remote ok"}]},"finishReason":"STOP"}]}}`)
	const remoteURL = "https://example.com/pic.png"

	cases := []struct {
		name         string
		from         provider.Protocol
		model        string
		payload      string
		assertClient func(t *testing.T, body []byte)
	}{
		{
			name:  "Messages_URL",
			from:  provider.Anthropic,
			model: "gemini-3.8-flash-high",
			payload: fmt.Sprintf(`{
				"model": "gemini-3.8-flash-high",
				"max_tokens": 1024,
				"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": %q}}]}]
			}`, remoteURL),
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
				if len(rep.Content) == 0 || rep.Content[0].Text != "remote ok" || rep.StopReason != "end_turn" {
					t.Errorf("Messages client mismatch: %+v", rep)
				}
			},
		},
		{
			name:  "Chat_URL",
			from:  provider.Chat,
			model: "claude-sonnet-4-6",
			payload: fmt.Sprintf(`{
				"model": "claude-sonnet-4-6",
				"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": %q}}]}]
			}`, remoteURL),
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
				if len(rep.Choices) == 0 || rep.Choices[0].Message.Content != "remote ok" || rep.Choices[0].FinishReason != "stop" {
					t.Errorf("Chat client mismatch: %+v", rep)
				}
			},
		},
		{
			name:  "Responses_URL",
			from:  provider.Responses,
			model: "gemini-3.8-flash-high",
			payload: fmt.Sprintf(`{
				"model": "gemini-3.8-flash-high",
				"input": [{"role": "user", "content": [{"type": "input_image", "image_url": %q}]}]
			}`, remoteURL),
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
				if rep.Status != "completed" || len(rep.Output) == 0 || len(rep.Output[0].Content) == 0 || rep.Output[0].Content[0].Text != "remote ok" {
					t.Errorf("Responses client mismatch: %+v", rep)
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
			if status != http.StatusOK || rec.Code != http.StatusOK || errMsg != "" {
				t.Fatalf("[%s] remote url failed: status=%d, rec.Code=%d, errMsg=%s", tc.name, status, rec.Code, errMsg)
			}
			if atomic.LoadInt64(&tr.callCount) != 1 {
				t.Errorf("[%s] expected 1 upstream call, got %d", tc.name, tr.callCount)
			}

			tc.assertClient(t, rec.Body.Bytes())

			var env codeAssistWireEnvelope
			if err := json.Unmarshal(tr.capturedBody, &env); err != nil {
				t.Fatalf("unmarshal wire failed: %v", err)
			}
			if len(env.Request.Contents) != 1 || len(env.Request.Contents[0].Parts) != 1 {
				t.Fatalf("expected exactly 1 part on wire, got %+v", env.Request.Contents)
			}
			part := env.Request.Contents[0].Parts[0]
			if part.Text != "" || part.InlineData != nil || part.FileData == nil {
				t.Fatalf("expected exclusive fileData, got: %+v", part)
			}
			if env.Request.Contents[0].Role != "user" {
				t.Errorf("[%s] role mismatch: got %q, want 'user'", tc.name, env.Request.Contents[0].Role)
			}
			if part.FileData.FileURI != remoteURL {
				t.Errorf("fileUri mismatch: got %q, want %q", part.FileData.FileURI, remoteURL)
			}
			if part.FileData.MimeType != "" {
				t.Errorf("[%s] mimeType mismatch: got %q, want empty", tc.name, part.FileData.MimeType)
			}
		})
	}
}
