package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB16ProcessRestart covers B16: Committed turns are recoverable after process restart;
// corrupted or truncated records are rejected without pretending to succeed as empty sessions;
// and cleanup/quarantine of corrupted records never damages intact sessions.
func testB16ProcessRestart(t *testing.T) {
	t.Run("RestartPreservesCommittedHistoryAndBindingsH", testB16RestartPreservesCommittedHistoryAndBindingsH)
	t.Run("CorruptedTruncatedFileRejectedNotTreatedAsEmptyH", testB16CorruptedTruncatedFileRejectedNotTreatedAsEmptyH)
	t.Run("CorruptedSessionDoesNotAffectHealthySessionH", testB16CorruptedSessionDoesNotAffectHealthySessionH)
}

// 1. Process restart: Previous process committed a round; a new Server instance reloads it from disk.
func testB16RestartPreservesCommittedHistoryAndBindingsH(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	const (
		sessionID     = "restart-session-001"
		nativeCallID  = "call_native_persist_01"
		clientCallID  = "t_persist_01"
		realNativeSig = "opaque_sig_persisted_across_restart_xyz"
		userAlice     = "alice@example.com"
		projAlice     = "proj-alice"
		model         = "claude-sonnet-4-6"
	)

	scope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}

	persister1 := newFileAntigravityPersister(filepath.Join(cacheDir, "antigravity_rounds"))
	canonical := []canonicalPart{
		{Kind: ToolCall, CallID: nativeCallID, Name: "view_file", Args: `{"path":"/a.go"}`, ThoughtSignature: realNativeSig},
	}
	bindings := []AntigravityToolBinding{
		{
			NativeID:        nativeCallID,
			NativeName:      "view_file",
			NativeArgs:      json.RawMessage(`{"path":"/a.go"}`),
			NativeSignature: realNativeSig,
			ClientID:        clientCallID,
			ClientName:      "view_file",
		},
	}

	// Process 1 commits round to disk
	if err := persister1.CommitRound(scope, sessionID, canonical, bindings); err != nil {
		t.Fatalf("Process 1 commit failed: %v", err)
	}

	// SIMULATE PROCESS RESTART:
	// Wipe all in-memory caches and reinitialize persister from the exact same disk directory
	defaultToolBindingStore.clear()
	defaultAntigravityLedger.Lock()
	defaultAntigravityLedger.records = make(map[string]*antigravityLedgerRecord)
	defaultAntigravityLedger.Unlock()

	persister2 := newFileAntigravityPersister(filepath.Join(cacheDir, "antigravity_rounds"))
	oldPersister := currentAntigravityPersister
	currentAntigravityPersister = persister2
	t.Cleanup(func() { currentAntigravityPersister = oldPersister })

	// Process 2: Server instance 2 receives turn 2 from carrier-less client
	s2 := New()
	tr2 := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"resumed"}]},"finishReason":"STOP"}]}}`)}
	s2.client = &http.Client{Transport: tr2}

	p := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-test")

	turn2Body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "view a"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "` + clientCallID + `", "name": "view_file", "input": {"path":"/a.go"}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "` + clientCallID + `", "content": "package a"}]}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(turn2Body))
	httpReq.Header.Set("X-Magpie-Client-Profile", "claude_code")
	httpReq.Header.Set("X-Magpie-Session", sessionID)
	httpReq.Header.Set("X-Antigravity-Resume", "strict")

	var u Usage
	status, msg := s2.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, model, []byte(turn2Body), &u)

	if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
		t.Fatalf("Process 2 translate failed: status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}
	if atomic.LoadInt64(&tr2.callCount) != 1 {
		t.Fatalf("expected 1 upstream call, got %d", tr2.callCount)
	}

	// Verify upstream wire request has restored signature and native name from reloaded disk state
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
	json.Unmarshal(tr2.capturedBody, &env)

	var recoveredSig string
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				recoveredSig = p.ThoughtSignature
			}
		}
	}

	if recoveredSig != realNativeSig {
		t.Errorf("failed to recover native signature after restart: got %q, want %q", recoveredSig, realNativeSig)
	}
}

// 2. Corrupted / truncated file on disk must be rejected and NEVER treated as an empty session.
func testB16CorruptedTruncatedFileRejectedNotTreatedAsEmptyH(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	roundsDir := filepath.Join(cacheDir, "antigravity_rounds")
	_ = os.MkdirAll(roundsDir, 0755)

	const (
		corruptedSessionID = "corrupted-session-666"
		userAlice          = "alice@example.com"
		projAlice          = "proj-alice"
		model              = "claude-sonnet-4-6"
	)

	scope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}

	// Write an intentionally truncated/corrupted JSON file to disk
	key := buildSessionLedgerKey(scope, corruptedSessionID)
	corruptedFile := filepath.Join(roundsDir, sanitizeFilename(key)+".json")
	if err := os.WriteFile(corruptedFile, []byte(`{"scope": {"caller":"claude_code"}, "canonical": [broken_truncated_json...`), 0644); err != nil {
		t.Fatalf("failed to write corrupted file: %v", err)
	}

	persister := newFileAntigravityPersister(roundsDir)
	oldPersister := currentAntigravityPersister
	currentAntigravityPersister = persister
	t.Cleanup(func() { currentAntigravityPersister = oldPersister })

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"unexpected"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-test")

	body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [{"role": "user", "content": "resume corrupted"}]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	httpReq.Header.Set("X-Magpie-Client-Profile", "claude_code")
	httpReq.Header.Set("X-Magpie-Session", corruptedSessionID)
	httpReq.Header.Set("X-Antigravity-Resume", "strict")

	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, model, []byte(body), &u)

	// Must be rejected with 400 (never treated as empty session)
	if status != http.StatusBadRequest || rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for corrupted session file, got status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}
	if calls := atomic.LoadInt64(&tr.callCount); calls != 0 {
		t.Fatalf("expected 0 upstream calls for corrupted session, got %d", calls)
	}
}

// 3. Presence of a corrupted session must NOT damage or prevent recovery of other healthy sessions.
func testB16CorruptedSessionDoesNotAffectHealthySessionH(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	roundsDir := filepath.Join(cacheDir, "antigravity_rounds")
	_ = os.MkdirAll(roundsDir, 0755)

	const (
		healthySessionID   = "healthy-session-888"
		corruptedSessionID = "bad-session-444"
		userAlice          = "alice@example.com"
		projAlice          = "proj-alice"
		model              = "claude-sonnet-4-6"
	)

	scope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}

	persister := newFileAntigravityPersister(roundsDir)
	oldPersister := currentAntigravityPersister
	currentAntigravityPersister = persister
	t.Cleanup(func() { currentAntigravityPersister = oldPersister })

	// 1. Commit Healthy Session
	canonicalHealthy := []canonicalPart{{Kind: Text, Text: "Healthy committed text."}}
	if err := persister.CommitRound(scope, healthySessionID, canonicalHealthy, nil); err != nil {
		t.Fatalf("commit healthy session failed: %v", err)
	}

	// 2. Plant Corrupted Session on disk
	badKey := buildSessionLedgerKey(scope, corruptedSessionID)
	badFile := filepath.Join(roundsDir, sanitizeFilename(badKey)+".json")
	_ = os.WriteFile(badFile, []byte(`{broken-json-file`), 0644)

	// 3. Clear memory cache to force disk resolution
	defaultToolBindingStore.clear()
	defaultAntigravityLedger.Lock()
	defaultAntigravityLedger.records = make(map[string]*antigravityLedgerRecord)
	defaultAntigravityLedger.Unlock()

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"healthy answered"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}
	p := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-test")

	// Healthy session request
	healthyBody := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": "Healthy committed text."},
			{"role": "user", "content": "next task"}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(healthyBody))
	httpReq.Header.Set("X-Magpie-Client-Profile", "claude_code")
	httpReq.Header.Set("X-Magpie-Session", healthySessionID)
	httpReq.Header.Set("X-Antigravity-Resume", "strict")

	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, model, []byte(healthyBody), &u)

	if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
		t.Fatalf("healthy session failed to resume in presence of corrupted session: status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}
	if calls := atomic.LoadInt64(&tr.callCount); calls != 1 {
		t.Fatalf("expected 1 upstream call for healthy session, got %d", tr.callCount)
	}
}
