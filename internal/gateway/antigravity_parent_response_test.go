package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB07ParentResponse covers B07: Responses API incremental requests with previous_response_id
// resolve their parent response within the scoped namespace, reconstructing the complete multi-turn
// history without duplicates or drops; unknown/cross-account parent IDs are rejected before upstream dispatch (0 calls);
// and previous_response_id is never mistaken for Antigravity's internal sessionId.
func testB07ParentResponse(t *testing.T) {
	t.Run("ReconstructsFullHistory", testB07ReconstructsFullHistory)
	t.Run("UnknownParentResponseRejected", testB07UnknownParentResponseRejected)
	t.Run("CrossAccountParentResponseRejected", testB07CrossAccountParentResponseRejected)
	t.Run("PreviousResponseIDNeverUsedAsSessionID", testB07PreviousResponseIDNeverUsedAsSessionID)
}

// 1. Reconstruct complete multi-turn conversation from parent response record.
func testB07ReconstructsFullHistory(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
		parentRespID = "resp_parent_test_001"
		userAlice    = "alice@example.com"
		projAlice    = "proj-alice"
		model        = "gemini-3.8-flash-high"
	)

	scope := antigravitySessionScope{
		Caller:  "unknown",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}

	// Parent history: turn 1 user, turn 1 assistant
	parentMessages := []Message{
		{Role: "user", Parts: []Part{{Kind: Text, Text: "What is 2+2?"}}},
		{Role: "assistant", Parts: []Part{{Kind: Text, Text: "4"}}},
	}

	defaultAntigravityParentStore.StoreParent(scope, parentRespID, "-12345678", parentMessages)
	t.Cleanup(func() {
		defaultAntigravityParentStore.Lock()
		delete(defaultAntigravityParentStore.records, buildParentStoreKey(scope, parentRespID))
		defaultAntigravityParentStore.Unlock()
	})

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"14"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	pAlice := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-alice")

	incrementalBody := `{
		"model": "gemini-3.8-flash-high",
		"previous_response_id": "` + parentRespID + `",
		"input": [
			{"role": "user", "content": [{"type": "input_text", "text": "Now add 10"}]}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader([]byte(incrementalBody)))

	var u Usage
	status, msg := s.translate(rec, httpReq, pAlice, provider.Responses, provider.CodeAssist, model, []byte(incrementalBody), &u)

	if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
		t.Fatalf("translate returned status %d/%d msg=%q body=%s", status, rec.Code, msg, rec.Body.String())
	}
	if atomic.LoadInt64(&tr.callCount) != 1 {
		t.Fatalf("expected 1 upstream call, got %d", tr.callCount)
	}

	// Verify upstream wire body contents contains all 3 turns in order:
	// Turn 1 user ("What is 2+2?"), Turn 1 model ("4"), Turn 2 user ("Now add 10")
	var env struct {
		Request struct {
			Contents []struct {
				Role  string `json:"role"`
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"contents"`
			SessionID string `json:"sessionId"`
		} `json:"request"`
	}
	if err := json.Unmarshal(tr.capturedBody, &env); err != nil {
		t.Fatalf("unmarshal upstream wire: %v\nbody: %s", err, tr.capturedBody)
	}

	contents := env.Request.Contents
	if len(contents) != 3 {
		t.Fatalf("expected 3 contents turns, got %d: %v", len(contents), contents)
	}

	if contents[0].Role != "user" || contents[0].Parts[0].Text != "What is 2+2?" {
		t.Errorf("contents[0] mismatch: %v", contents[0])
	}
	if contents[1].Role != "model" || contents[1].Parts[0].Text != "4" {
		t.Errorf("contents[1] mismatch: %v", contents[1])
	}
	if contents[2].Role != "user" || contents[2].Parts[0].Text != "Now add 10" {
		t.Errorf("contents[2] mismatch: %v", contents[2])
	}

	// sessionId must NOT be previous_response_id
	if env.Request.SessionID == parentRespID {
		t.Errorf("previous_response_id %q was erroneously used as Antigravity sessionId", parentRespID)
	}
}

// 2. When previous_response_id does not exist, the request must fail with 400 (0 upstream calls).
func testB07UnknownParentResponseRejected(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"unexpected"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@example.com", "proj-test", "tok-test")

	body := `{
		"model": "claude-sonnet-4-6",
		"previous_response_id": "resp_non_existent_123",
		"input": [
			{"role": "user", "content": "hello"}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader([]byte(body)))

	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Responses, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

	if status != http.StatusBadRequest || rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for unknown previous_response_id, got status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}
	if atomic.LoadInt64(&tr.callCount) != 0 {
		t.Fatalf("expected 0 upstream calls for unknown parent, got %d", tr.callCount)
	}
	if !strings.Contains(msg, "previous_response_id") && !strings.Contains(rec.Body.String(), "previous_response_id") {
		t.Errorf("error must mention previous_response_id, got msg=%q body=%s", msg, rec.Body.String())
	}
}

// 3. When a previous_response_id belongs to another account, it must be rejected (400, 0 upstream calls).
func testB07CrossAccountParentResponseRejected(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
		parentRespID = "resp_alice_confidential_007"
		userAlice    = "alice@example.com"
		userBob      = "bob@example.com"
		projAlice    = "proj-alice"
		projBob      = "proj-bob"
		model        = "claude-sonnet-4-6"
	)

	// Pre-seed in Alice's scope
	aliceScope := antigravitySessionScope{
		Caller:  "unknown",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}
	defaultAntigravityParentStore.StoreParent(aliceScope, parentRespID, "-999", []Message{
		{Role: "user", Parts: []Part{{Kind: Text, Text: "alice secret"}}},
	})
	t.Cleanup(func() {
		defaultAntigravityParentStore.Lock()
		delete(defaultAntigravityParentStore.records, buildParentStoreKey(aliceScope, parentRespID))
		defaultAntigravityParentStore.Unlock()
	})

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"unexpected"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	// Bob attempts to use Alice's parentRespID
	pBob := provider.AntigravityTestProvider("antigravity", userBob, projBob, "tok-bob")

	body := `{
		"model": "claude-sonnet-4-6",
		"previous_response_id": "` + parentRespID + `",
		"input": [
			{"role": "user", "content": "tell me alice's secret"}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader([]byte(body)))

	var u Usage
	status, msg := s.translate(rec, httpReq, pBob, provider.Responses, provider.CodeAssist, model, []byte(body), &u)

	if status != http.StatusBadRequest || rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for cross-account previous_response_id, got status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}
	if atomic.LoadInt64(&tr.callCount) != 0 {
		t.Fatalf("expected 0 upstream calls for cross-account parent, got %d", tr.callCount)
	}
}

// 4. Assert previous_response_id is never used as Antigravity sessionId.
func testB07PreviousResponseIDNeverUsedAsSessionID(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
		parentRespID = "resp_alpha_beta_123"
		userAlice    = "alice@example.com"
		projAlice    = "proj-alice"
		model        = "claude-sonnet-4-6"
	)

	scope := antigravitySessionScope{
		Caller:  "unknown",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}

	defaultAntigravityParentStore.StoreParent(scope, parentRespID, "-88888", []Message{
		{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}},
	})
	t.Cleanup(func() {
		defaultAntigravityParentStore.Lock()
		delete(defaultAntigravityParentStore.records, buildParentStoreKey(scope, parentRespID))
		defaultAntigravityParentStore.Unlock()
	})

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-alice")

	body := `{
		"model": "claude-sonnet-4-6",
		"previous_response_id": "` + parentRespID + `",
		"input": [
			{"role": "user", "content": "next"}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader([]byte(body)))
	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Responses, provider.CodeAssist, model, []byte(body), &u)
	if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
		t.Fatalf("translate failed: %d msg=%q", status, msg)
	}

	var env struct {
		Request struct {
			SessionID string `json:"sessionId"`
		} `json:"request"`
	}
	json.Unmarshal(tr.capturedBody, &env)

	if env.Request.SessionID == parentRespID || strings.HasPrefix(env.Request.SessionID, "resp_") {
		t.Errorf("Antigravity sessionId must not be a response ID, got %q", env.Request.SessionID)
	}

	sessionPattern := regexp.MustCompile(`^-[0-9]+$`)
	if !sessionPattern.MatchString(env.Request.SessionID) {
		t.Errorf("Antigravity sessionId must be a negative integer string, got %q", env.Request.SessionID)
	}
}
