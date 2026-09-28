package gateway

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03cImageValidationHInvalid covers rejection scenarios of invalid image payloads.
func testB03cImageValidationHInvalid(t *testing.T) {
	t.Run("InvalidImageRejectionAndPathIndex", testB03cHInvalidImageRejectionAndPathIndex)
}

// 1. Invalid image payload rejection: HTTP 400, 0 upstream calls, and exact path index targeting.
func testB03cHInvalidImageRejectionAndPathIndex(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	// Oracle verification: StdEncoding skips \r and \n entirely, decoding "\r\n" to empty bytes without error
	decodedCRLF, err := base64.StdEncoding.DecodeString("\r\n")
	if err != nil || len(decodedCRLF) != 0 {
		t.Fatalf("oracle precondition failed: base64.StdEncoding must decode \\r\\n to empty without error, got err=%v len=%d", err, len(decodedCRLF))
	}

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
		// (a) Anthropic Messages image node errors
		{
			name:         "Messages_MissingSource",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image"}]}]}`,
			wantPathNode: "messages[0].content[1].source",
		},
		{
			name:         "Messages_SourceNull",
			from:         provider.Anthropic,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":null}]}]}`,
			wantPathNode: "messages[0].content[1].source",
		},
		{
			name:         "Messages_Base64MissingData",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","media_type":"image/png"}}]}]}`,
			wantPathNode: "messages[0].content[1].source.data",
		},
		{
			name:         "Messages_Base64EmptyData",
			from:         provider.Anthropic,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":""}}]}]}`,
			wantPathNode: "messages[0].content[1].source.data",
		},
		{
			name:         "Messages_Base64InvalidData",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"not_base64!!!"}}]}]}`,
			wantPathNode: "messages[0].content[1].source.data",
		},
		{
			name:         "Messages_Base64WhitespaceOnlyDecodesEmpty",
			from:         provider.Anthropic,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"\r\n"}}]}]}`,
			wantPathNode: "messages[0].content[1].source.data",
		},
		{
			name:         "Messages_Base64MissingDataCannotFallbackToURL",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","media_type":"image/png","url":"https://example.com/pic.png"}}]}]}`,
			wantPathNode: "messages[0].content[1].source.data",
		},
		{
			name:         "Messages_URLMissingURLField",
			from:         provider.Anthropic,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"url"}}]}]}`,
			wantPathNode: "messages[0].content[1].source.url",
		},
		{
			name:         "Messages_URLEmptyURLField",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"url","url":""}}]}]}`,
			wantPathNode: "messages[0].content[1].source.url",
		},
		{
			name:         "Messages_MissingMediaType",
			from:         provider.Anthropic,
			model:        "claude-sonnet-4-6",
			payload:      fmt.Sprintf(`{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","data":%q}}]}]}`, samplePNGBase64),
			wantPathNode: "messages[0].content[1].source.media_type",
		},
		{
			name:         "Messages_EmptySubtypeMediaType",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","media_type":"image/","data":%q}}]}]}`, samplePNGBase64),
			wantPathNode: "messages[0].content[1].source.media_type",
		},
		{
			name:         "Messages_NonImageMediaType",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","media_type":"application/pdf","data":%q}}]}]}`, samplePNGBase64),
			wantPathNode: "messages[0].content[1].source.media_type",
		},
		{
			name:         "Messages_UnknownSourceType",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"custom_stream","url":"https://example.com"}}]}]}`,
			wantPathNode: "messages[0].content[1].source.type",
		},

		// (b) Chat image_url errors
		{
			name:         "Chat_MissingImageURL",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url"}]}]}`,
			wantPathNode: "messages[0].content[1].image_url",
		},
		{
			name:         "Chat_ImageURLNull",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":null}]}]}`,
			wantPathNode: "messages[0].content[1].image_url",
		},
		{
			name:         "Chat_MissingURLField",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{}}]}]}`,
			wantPathNode: "messages[0].content[1].image_url.url",
		},
		{
			name:         "Chat_EmptyURLField",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":""}}]}]}`,
			wantPathNode: "messages[0].content[1].image_url.url",
		},
		{
			name:         "Chat_DataURLMissingComma",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64"}}]}]}`,
			wantPathNode: "messages[0].content[1].image_url.url",
		},
		{
			name:         "Chat_DataURLMissingBase64Tag",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png,%s"}}]}]}`, samplePNGBase64),
			wantPathNode: "messages[0].content[1].image_url.url",
		},
		{
			name:         "Chat_DataURLEmptyData",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64,"}}]}]}`,
			wantPathNode: "messages[0].content[1].image_url.url",
		},
		{
			name:         "Chat_DataURLWhitespaceOnlyDecodesEmpty",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":%s}}]}]}`, mustEscape("data:image/png;base64,\r\n")),
			wantPathNode: "messages[0].content[1].image_url.url",
		},
		{
			name:         "Chat_DataURLBadBase64",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64,not_base64!!!"}}]}]}`,
			wantPathNode: "messages[0].content[1].image_url.url",
		},
		{
			name:         "Chat_DataURLNonImageMIME",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      fmt.Sprintf(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:application/pdf;base64,%s"}}]}]}`, samplePNGBase64),
			wantPathNode: "messages[0].content[1].image_url.url",
		},
		{
			name:         "Chat_DataURLTrailingSpace",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":%s}}]}]}`, mustEscape(samplePNGDataURL+" ")),
			wantPathNode: "messages[0].content[1].image_url.url",
		},

		// (c) Responses input_image errors: path must be input[i].content[j].image_url
		{
			name:         "Responses_MissingImageURL",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image"}]}]}`,
			wantPathNode: "input[0].content[1].image_url",
		},
		{
			name:         "Responses_EmptyImageURL",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":""}]}]}`,
			wantPathNode: "input[0].content[1].image_url",
		},
		{
			name:         "Responses_DataURLMissingComma",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":"data:image/png;base64"}]}]}`,
			wantPathNode: "input[0].content[1].image_url",
		},
		{
			name:         "Responses_DataURLMissingBase64Tag",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      fmt.Sprintf(`{"model":"claude-sonnet-4-6","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":"data:image/png,%s"}]}]}`, samplePNGBase64),
			wantPathNode: "input[0].content[1].image_url",
		},
		{
			name:         "Responses_DataURLEmptyData",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":"data:image/png;base64,"}]}]}`,
			wantPathNode: "input[0].content[1].image_url",
		},
		{
			name:         "Responses_DataURLWhitespaceOnlyDecodesEmpty",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      fmt.Sprintf(`{"model":"claude-sonnet-4-6","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":%s}]}]}`, mustEscape("data:image/png;base64,\r\n")),
			wantPathNode: "input[0].content[1].image_url",
		},
		{
			name:         "Responses_DataURLBadBase64",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":"data:image/png;base64,not_base64!!!"}]}]}`,
			wantPathNode: "input[0].content[1].image_url",
		},
		{
			name:         "Responses_DataURLNonImageMIME",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      fmt.Sprintf(`{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":"data:application/pdf;base64,%s"}]}]}`, samplePNGBase64),
			wantPathNode: "input[0].content[1].image_url",
		},
		{
			name:         "Responses_DataURLTrailingSpace",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      fmt.Sprintf(`{"model":"claude-sonnet-4-6","input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":%s}]}]}`, mustEscape(samplePNGDataURL+" ")),
			wantPathNode: "input[0].content[1].image_url",
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
				t.Errorf("[%s] unmarshal client error JSON failed: %v (raw body: %s)", tc.name, err, rec.Body.String())
			} else if !strings.Contains(clientErr.Error.Message, tc.wantPathNode) {
				t.Errorf("[%s] client error message %q does not contain expected path node %q", tc.name, clientErr.Error.Message, tc.wantPathNode)
			}

			if !strings.Contains(errMsg, tc.wantPathNode) {
				t.Errorf("[%s] internal errMsg %q does not contain expected path node %q", tc.name, errMsg, tc.wantPathNode)
			}
		})
	}
}
