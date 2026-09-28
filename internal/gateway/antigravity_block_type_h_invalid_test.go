package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03dBlockTypeValidationHInvalid covers rejection scenarios of unmapped or invalid block types in user content arrays.
func testB03dBlockTypeValidationHInvalid(t *testing.T) {
	t.Run("InvalidBlockTypeRejectionAndPathIndex", testB03dHInvalidBlockTypeRejectionAndPathIndex)
}

// 1. Invalid block types in user content arrays: HTTP 400, 0 upstream calls, and exact path index targeting.
func testB03dHInvalidBlockTypeRejectionAndPathIndex(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	cases := []struct {
		name         string
		from         provider.Protocol
		model        string
		payload      string
		wantPathNode string
	}{
		// (a) Anthropic Messages: messages[0].content[1].type
		{
			name:         "Messages_UnknownSpellingTxt",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"txt","text":"bad"},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},
		{
			name:         "Messages_UnmappedDocument",
			from:         provider.Anthropic,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"document","source":{}},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},
		{
			name:         "Messages_MissingType",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"text":"no_type"},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},
		{
			name:         "Messages_EmptyType",
			from:         provider.Anthropic,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"","text":"empty"},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},
		{
			name:         "Messages_NullType",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":null,"text":"null"},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},

		// (b) Chat: messages[0].content[1].type
		{
			name:         "Chat_UnknownSpellingTxt",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"txt","text":"bad"},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},
		{
			name:         "Chat_UnmappedInputAudio",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"input_audio","input_audio":{}},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},
		{
			name:         "Chat_MissingType",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"before"},{"text":"no_type"},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},
		{
			name:         "Chat_EmptyType",
			from:         provider.Chat,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"","text":"empty"},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},
		{
			name:         "Chat_NullType",
			from:         provider.Chat,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":null,"text":"null"},{"type":"text","text":"after"}]}]}`,
			wantPathNode: "messages[0].content[1].type",
		},

		// (c) Responses: input[0].content[1].type
		{
			name:         "Responses_UnknownSpellingTxt",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":[{"type":"input_text","text":"before"},{"type":"txt","text":"bad"},{"type":"input_text","text":"after"}]}]}`,
			wantPathNode: "input[0].content[1].type",
		},
		{
			name:         "Responses_UnmappedInputFile",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","input":[{"role":"user","content":[{"type":"input_text","text":"before"},{"type":"input_file","file":{}},{"type":"input_text","text":"after"}]}]}`,
			wantPathNode: "input[0].content[1].type",
		},
		{
			name:         "Responses_MissingType",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":[{"type":"input_text","text":"before"},{"text":"no_type"},{"type":"input_text","text":"after"}]}]}`,
			wantPathNode: "input[0].content[1].type",
		},
		{
			name:         "Responses_EmptyType",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","input":[{"role":"user","content":[{"type":"input_text","text":"before"},{"type":"","text":"empty"},{"type":"input_text","text":"after"}]}]}`,
			wantPathNode: "input[0].content[1].type",
		},
		{
			name:         "Responses_NullType",
			from:         provider.Responses,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","input":[{"role":"user","content":[{"type":"input_text","text":"before"},{"type":null,"text":"null"},{"type":"input_text","text":"after"}]}]}`,
			wantPathNode: "input[0].content[1].type",
		},

		// (d) Non-zero message and block index targeting
		{
			name:         "Messages_NonZeroIndex_Targeting",
			from:         provider.Anthropic,
			model:        "gemini-3.8-flash-high",
			payload:      `{"model":"gemini-3.8-flash-high","max_tokens":100,"messages":[{"role":"user","content":"msg0"},{"role":"user","content":[{"type":"text","text":"c0"},{"type":"text","text":"c1"},{"type":"unmapped_type_ext","data":"c2"},{"type":"text","text":"c3"}]}]}`,
			wantPathNode: "messages[1].content[2].type",
		},
		{
			name:         "Responses_NonZeroIndex_Targeting",
			from:         provider.Responses,
			model:        "claude-sonnet-4-6",
			payload:      `{"model":"claude-sonnet-4-6","input":[{"role":"user","content":"item0"},{"role":"assistant","content":"item1"},{"role":"user","content":[{"type":"input_text","text":"c0"},{"type":"unmapped_type_ext","data":"c1"},{"type":"input_text","text":"c2"}]}]}`,
			wantPathNode: "input[2].content[1].type",
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
				t.Errorf("[%s] unmarshal client error JSON failed: %v (raw: %s)", tc.name, err, rec.Body.String())
			} else if !strings.Contains(clientErr.Error.Message, tc.wantPathNode) {
				t.Errorf("[%s] client error message %q does not contain expected path node %q", tc.name, clientErr.Error.Message, tc.wantPathNode)
			}

			if !strings.Contains(errMsg, tc.wantPathNode) {
				t.Errorf("[%s] internal errMsg %q does not contain expected path node %q", tc.name, errMsg, tc.wantPathNode)
			}
		})
	}
}
