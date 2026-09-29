package gateway

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB06SessionIsolation covers B06: The session ledger isolates records strictly
// by caller profile, account, project, and model. Identical opening messages do not
// cause distinct conversations to cross-contaminate; long session IDs are never truncated;
// private headers/metadata are only recognized within profile boundaries; and forged
// cross-account resume references are rejected before upstream dispatch (0 calls).
func testB06SessionIsolation(t *testing.T) {
	t.Run("Unit_NoHashAsFallbackKey", testB06NoHashAsFallbackKey)
	t.Run("Unit_NoTruncateLongSessionID", testB06NoTruncateLongSessionID)
	t.Run("Unit_ProfileBoundaryIsolation", testB06ProfileBoundaryIsolation)
	t.Run("Unit_NamespaceStrictIsolation", testB06NamespaceStrictIsolation)
	t.Run("HLayer_CrossAccountResumeRejected", testB06CrossAccountResumeRejected)
	t.Run("HLayer_ValidResumePasses", testB06ValidResumePasses)
}

// 1. Antigravity must never fall back to hashing the first message content as a session key.
func testB06NoHashAsFallbackKey(t *testing.T) {
	const commonFirstMessage = "Hello assistant, please start the task now."

	bodyA := []byte(`{"model":"m","messages":[{"role":"user","content":"` + commonFirstMessage + `"}]}`)
	bodyAWithTurn := []byte(`{"model":"m","messages":[{"role":"user","content":"` + commonFirstMessage + `"},{"role":"assistant","content":"ok"}]}`)

	// Without trusted session references, extractSessionReference must return empty, NOT a hash of the first message.
	refA := extractSessionReference("claude_code", http.Header{}, bodyA)
	refB := extractSessionReference("claude_code", http.Header{}, bodyAWithTurn)

	if refA != "" || refB != "" {
		t.Fatalf("extractSessionReference should be empty when no session reference provided, got refA=%q refB=%q", refA, refB)
	}

	scope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: "test@example.com",
		Project: "p1",
		Model:   "claude-sonnet-4-6",
	}

	keyA := buildSessionLedgerKey(scope, refA)
	keyB := buildSessionLedgerKey(scope, refB)
	if keyA != "" || keyB != "" {
		t.Errorf("buildSessionLedgerKey with empty sessionRef must be empty, got keyA=%q keyB=%q", keyA, keyB)
	}
}

// 2. Long session IDs must never be truncated (e.g. at 128 chars), preventing collisions.
func testB06NoTruncateLongSessionID(t *testing.T) {
	longPrefix := strings.Repeat("a", 130)
	longID_A := longPrefix + "_suffix_AAA"
	longID_B := longPrefix + "_suffix_BBB"

	scope := antigravitySessionScope{
		Caller:  "opencode",
		Account: "user@example.com",
		Project: "p-long",
		Model:   "gemini-3.8-flash-high",
	}

	keyA := buildSessionLedgerKey(scope, longID_A)
	keyB := buildSessionLedgerKey(scope, longID_B)

	if keyA == "" || keyB == "" {
		t.Fatalf("keys should not be empty: keyA=%q keyB=%q", keyA, keyB)
	}
	if keyA == keyB {
		t.Fatalf("long session IDs were truncated and collided: %s == %s", keyA, keyB)
	}
	if !strings.Contains(keyA, longID_A) || !strings.Contains(keyB, longID_B) {
		t.Errorf("keys must preserve entire session ID: keyA=%q keyB=%q", keyA, keyB)
	}
}

// 3. Private headers and body metadata are only recognized within corresponding profile boundaries.
func testB06ProfileBoundaryIsolation(t *testing.T) {
	const (
		openCodeSessionID = "opencode-session-987"
		claudeSessionID   = "claude-session-456"
		piSessionID       = "pi-session-123"
	)

	openCodeHeader := http.Header{}
	openCodeHeader.Set("x-opencode-session", openCodeSessionID)

	piHeader := http.Header{}
	piHeader.Set("x-session-id", piSessionID)

	claudeBody := []byte(`{
		"model": "claude-sonnet-4-6",
		"metadata": {"user_id": "{\"session_id\":\"` + claudeSessionID + `\"}"},
		"messages": [{"role":"user","content":"hi"}]
	}`)

	// A. OpenCode header
	if got := extractSessionReference("opencode", openCodeHeader, nil); got != openCodeSessionID {
		t.Errorf("opencode profile must extract x-opencode-session, got %q", got)
	}
	if got := extractSessionReference("unknown", openCodeHeader, nil); got != "" {
		t.Errorf("unknown profile must NOT guess x-opencode-session, got %q", got)
	}
	if got := extractSessionReference("claude_code", openCodeHeader, nil); got != "" {
		t.Errorf("claude_code profile must NOT guess x-opencode-session, got %q", got)
	}

	// B. Pi header
	if got := extractSessionReference("pi", piHeader, nil); got != piSessionID {
		t.Errorf("pi profile must extract x-session-id, got %q", got)
	}
	if got := extractSessionReference("unknown", piHeader, nil); got != "" {
		t.Errorf("unknown profile must NOT guess x-session-id, got %q", got)
	}

	// C. Claude Code metadata.user_id
	if got := extractSessionReference("claude_code", http.Header{}, claudeBody); got != claudeSessionID {
		t.Errorf("claude_code profile must extract session_id from metadata.user_id, got %q", got)
	}
	if got := extractSessionReference("unknown", http.Header{}, claudeBody); got != "" {
		t.Errorf("unknown profile must NOT inspect claude metadata, got %q", got)
	}

	// D. Universal explicit header X-Magpie-Session is respected by all profiles
	universalHeader := http.Header{}
	universalHeader.Set("X-Magpie-Session", "universal-sess-1")
	for _, prof := range []string{"unknown", "claude_code", "opencode", "pi"} {
		if got := extractSessionReference(prof, universalHeader, nil); got != "universal-sess-1" {
			t.Errorf("X-Magpie-Session must be accepted across profile %s, got %q", prof, got)
		}
	}
}

// 4. Ledger records are isolated strictly by Caller, Account, Project, and Model.
func testB06NamespaceStrictIsolation(t *testing.T) {
	ledger := &antigravitySessionLedger{records: make(map[string]*antigravityLedgerRecord)}

	const sharedSessionRef = "target-session-ref-001"

	baseScope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: "alice@example.com",
		Project: "proj-alpha",
		Model:   "claude-sonnet-4-6",
	}

	record := &antigravityLedgerRecord{
		SessionID: sharedSessionRef,
		Scope:     baseScope,
		TurnCount: 3,
		Data:      map[string]any{"saved": "alice_secret_data"},
	}

	if err := ledger.Store(baseScope, sharedSessionRef, record); err != nil {
		t.Fatalf("Store error: %v", err)
	}

	// 1. Same scope can lookup
	rec, ok := ledger.Lookup(baseScope, sharedSessionRef)
	if !ok || rec.Data["saved"] != "alice_secret_data" {
		t.Fatalf("Lookup in exact scope failed: ok=%v, rec=%v", ok, rec)
	}

	// 2. Cross-account attempt must fail
	diffAccountScope := baseScope
	diffAccountScope.Account = "bob@example.com"
	if _, ok := ledger.Lookup(diffAccountScope, sharedSessionRef); ok {
		t.Errorf("cross-account lookup succeeded unexpectedly: Bob saw Alice's session")
	}

	// 3. Cross-project attempt must fail
	diffProjectScope := baseScope
	diffProjectScope.Project = "proj-beta"
	if _, ok := ledger.Lookup(diffProjectScope, sharedSessionRef); ok {
		t.Errorf("cross-project lookup succeeded unexpectedly")
	}

	// 4. Cross-caller attempt must fail
	diffCallerScope := baseScope
	diffCallerScope.Caller = "opencode"
	if _, ok := ledger.Lookup(diffCallerScope, sharedSessionRef); ok {
		t.Errorf("cross-caller lookup succeeded unexpectedly")
	}

	// 5. Cross-model attempt must fail
	diffModelScope := baseScope
	diffModelScope.Model = "gemini-3.8-flash-high"
	if _, ok := ledger.Lookup(diffModelScope, sharedSessionRef); ok {
		t.Errorf("cross-model lookup succeeded unexpectedly")
	}
}

// 5. H-Layer: When a forged session resume reference is provided from another account,
// the gateway must reject it before sending upstream (400, 0 upstream calls).
func testB06CrossAccountResumeRejected(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
		targetSessionID = "alice-sensitive-session-123"
		userAlice       = "alice@example.com"
		userBob         = "bob@example.com"
		projAlice       = "proj-alice"
		projBob         = "proj-bob"
		model           = "claude-sonnet-4-6"
	)

	// Pre-seed Alice's session in defaultAntigravityLedger
	aliceScope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}
	defaultAntigravityLedger.Store(aliceScope, targetSessionID, &antigravityLedgerRecord{
		SessionID: targetSessionID,
		Scope:     aliceScope,
		TurnCount: 2,
		Data:      map[string]any{"data": "secret"},
	})
	t.Cleanup(func() {
		defaultAntigravityLedger.Lock()
		delete(defaultAntigravityLedger.records, buildSessionLedgerKey(aliceScope, targetSessionID))
		defaultAntigravityLedger.Unlock()
	})

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"should not reach here"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	// Request sent under Bob's account trying to resume Alice's session
	pBob := provider.AntigravityTestProvider("antigravity", userBob, projBob, "tok-bob")

	body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [{"role":"user","content":"resume please"}],
		"metadata": {"user_id": "{\"session_id\":\"` + targetSessionID + `\"}"}
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(body)))
	httpReq.Header.Set("X-Antigravity-Resume", "strict")

	var u Usage
	status, msg := s.translate(rec, httpReq, pBob, provider.Anthropic, provider.CodeAssist, model, []byte(body), &u)

	// Must reject with 400
	if status != http.StatusBadRequest || rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for forged session resume, got status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}

	// Upstream call count must be strictly 0
	if calls := atomic.LoadInt64(&tr.callCount); calls != 0 {
		t.Fatalf("expected 0 upstream calls for rejected resume, got %d", calls)
	}

	if !strings.Contains(msg, "session") && !strings.Contains(rec.Body.String(), "session") {
		t.Errorf("error response should indicate session reference failure, msg=%q body=%s", msg, rec.Body.String())
	}
}

// 6. H-Layer: Valid session resume under matching account scope passes cleanly with 1 upstream call.
func testB06ValidResumePasses(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
		validSessionID = "alice-valid-session-999"
		userAlice      = "alice@example.com"
		projAlice      = "proj-alice"
		model          = "claude-sonnet-4-6"
	)

	aliceScope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}
	defaultAntigravityLedger.Store(aliceScope, validSessionID, &antigravityLedgerRecord{
		SessionID: validSessionID,
		Scope:     aliceScope,
		TurnCount: 1,
		Data:      map[string]any{"data": "alice_ok"},
	})
	t.Cleanup(func() {
		defaultAntigravityLedger.Lock()
		delete(defaultAntigravityLedger.records, buildSessionLedgerKey(aliceScope, validSessionID))
		defaultAntigravityLedger.Unlock()
	})

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"resumed ok"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	pAlice := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-alice")

	body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [{"role":"user","content":"resume please"}],
		"metadata": {"user_id": "{\"session_id\":\"` + validSessionID + `\"}"}
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(body)))
	httpReq.Header.Set("X-Antigravity-Resume", "strict")

	var u Usage
	status, msg := s.translate(rec, httpReq, pAlice, provider.Anthropic, provider.CodeAssist, model, []byte(body), &u)

	if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
		t.Fatalf("expected 200 for valid session resume, got status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}

	if calls := atomic.LoadInt64(&tr.callCount); calls != 1 {
		t.Fatalf("expected 1 upstream call for valid resume, got %d", calls)
	}
}
