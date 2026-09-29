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

// testB13CarrierlessHistory covers B13: Restoring native history without client signature carriers:
// All three protocols (Messages, Chat, Responses) share a single native ledger to recover signatures;
// carrier-less clients do not lose native signatures; Google signatures are never leaked into OpenAI
// encrypted_content; and foreign/unrecognized history is rejected under strict mode with 0 upstream calls.
func testB13CarrierlessHistory(t *testing.T) {
	t.Run("ThreeProtocolsRecoverNativeSignatureFromSharedLedger", testB13ThreeProtocolsRecoverNativeSignatureFromSharedLedger)
	t.Run("NeverLeakGoogleSigToOpenAIEncryptedContent", testB13NeverLeakGoogleSigToOpenAIEncryptedContent)
	t.Run("ForeignOpaqueHistoryRejectedUnderStrict", testB13ForeignOpaqueHistoryRejectedUnderStrict)
}

// 1. Messages, Chat, and Responses clients without carriers all recover the native signature
// from the single shared native ledger.
func testB13ThreeProtocolsRecoverNativeSignatureFromSharedLedger(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
		nativeCallID  = "call_shared_native_01"
		clientCallID  = "toolu_shared_client_01"
		realNativeSig = "opaque_shared_native_sig_99999"
		model         = "claude-sonnet-4-6"
		userAlice     = "alice@example.com"
		projAlice     = "proj-alice"
	)

	// Register in the single shared ledger/binding store
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)
	if err := defaultToolBindingStore.Bind(AntigravityToolBinding{
		NativeID:        nativeCallID,
		NativeName:      "view_file",
		NativeArgs:      json.RawMessage(`{"path":"/code/main.go"}`),
		NativeSignature: realNativeSig,
		ClientID:        clientCallID,
		ClientName:      "view_file",
	}); err != nil {
		t.Fatalf("Bind: %v", err)
	}

	p := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-test")

	protocols := []struct {
		name string
		from provider.Protocol
		body string
	}{
		{
			name: "Messages_NoCarrier",
			from: provider.Anthropic,
			body: `{
				"model": "claude-sonnet-4-6",
				"messages": [
					{"role": "user", "content": "view main"},
					{"role": "assistant", "content": [{"type": "tool_use", "id": "` + clientCallID + `", "name": "view_file", "input": {"path":"/code/main.go"}}]},
					{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "` + clientCallID + `", "content": "package main"}]}
				]
			}`,
		},
		{
			name: "Chat_NoCarrier",
			from: provider.Chat,
			body: `{
				"model": "claude-sonnet-4-6",
				"messages": [
					{"role": "user", "content": "view main"},
					{"role": "assistant", "tool_calls": [{"id": "` + clientCallID + `", "type": "function", "function": {"name": "view_file", "arguments": "{\"path\":\"/code/main.go\"}"}}]},
					{"role": "tool", "tool_call_id": "` + clientCallID + `", "content": "package main"}
				]
			}`,
		},
		{
			name: "Responses_NoCarrier",
			from: provider.Responses,
			body: `{
				"model": "claude-sonnet-4-6",
				"input": [
					{"role": "user", "content": "view main"},
					{"type": "function_call", "call_id": "` + clientCallID + `", "name": "view_file", "arguments": "{\"path\":\"/code/main.go\"}"},
					{"type": "function_call_output", "call_id": "` + clientCallID + `", "output": "package main"}
				]
			}`,
		},
	}

	for _, tc := range protocols {
		t.Run(tc.name, func(t *testing.T) {
			tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}}`)}
			s := New()
			s.client = &http.Client{Transport: tr}

			rec := httptest.NewRecorder()
			httpReq := httptest.NewRequest("POST", "/dummy", strings.NewReader(tc.body))

			var u Usage
			status, msg := s.translate(rec, httpReq, p, tc.from, provider.CodeAssist, model, []byte(tc.body), &u)
			if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
				t.Fatalf("translate returned status %d/%d msg=%q body=%s", status, rec.Code, msg, rec.Body.String())
			}

			if atomic.LoadInt64(&tr.callCount) != 1 {
				t.Fatalf("callCount = %d, want 1", tr.callCount)
			}

			var env struct {
				Request struct {
					Contents []struct {
						Parts []struct {
							FunctionCall *struct {
								ID string `json:"id"`
							} `json:"functionCall"`
							ThoughtSignature string `json:"thoughtSignature"`
						} `json:"parts"`
					} `json:"contents"`
				} `json:"request"`
			}
			if err := json.Unmarshal(tr.capturedBody, &env); err != nil {
				t.Fatalf("unmarshal error: %v", err)
			}

			var recoveredSig string
			for _, c := range env.Request.Contents {
				for _, p := range c.Parts {
					if p.FunctionCall != nil {
						recoveredSig = p.ThoughtSignature
					}
				}
			}

			if recoveredSig != realNativeSig {
				t.Errorf("[%s] native signature not recovered from shared ledger: got %q, want %q", tc.name, recoveredSig, realNativeSig)
			}
		})
	}
}

// 2. Anti-pattern: Google thought signatures must NEVER be leaked or wrapped into
// OpenAI Responses encrypted_content.
func testB13NeverLeakGoogleSigToOpenAIEncryptedContent(t *testing.T) {
	// Verify that when rendering Responses output, encrypted_content is not fabricated from thoughtSignature
	res := renderResponses(Result{
		ID:    "resp_1",
		Model: "claude-sonnet-4-6",
		Parts: []Part{
			{Kind: Thinking, Text: "thought", Signature: "google_secret_thought_signature_xyz"},
			{Kind: Text, Text: "Answer."},
		},
		Stop: "stop",
	}, "claude-sonnet-4-6")
	resStr := string(res)

	if strings.Contains(resStr, "encrypted_content") && strings.Contains(resStr, "google_secret_thought_signature_xyz") {
		t.Errorf("Google thought signature was illegally leaked into OpenAI encrypted_content:\n%s", resStr)
	}
}

// 3. Foreign/unrecognized opaque history is rejected under strict resume with 0 upstream calls.
func testB13ForeignOpaqueHistoryRejectedUnderStrict(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"unexpected"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	// Foreign call ID and session reference completely absent from our native ledger
	foreignBody := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"metadata": {"user_id": "{\"session_id\":\"foreign_session_unknown_999\"}"},
		"messages": [
			{"role": "user", "content": "call tool"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "call_foreign_unknown_999", "name": "unknown_tool", "input": {}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_foreign_unknown_999", "content": "foreign result"}]}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(foreignBody))
	httpReq.Header.Set("X-Antigravity-Resume", "strict")

	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(foreignBody), &u)

	if status != http.StatusBadRequest || rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for foreign unverified history under strict, got status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}
	if calls := atomic.LoadInt64(&tr.callCount); calls != 0 {
		t.Fatalf("expected 0 upstream calls for rejected foreign history, got %d", calls)
	}
	if !strings.Contains(msg, "session") && !strings.Contains(rec.Body.String(), "session") {
		t.Errorf("expected rejection reason to indicate session scope failure, got msg=%q body=%s", msg, rec.Body.String())
	}
}
