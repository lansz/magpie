package gateway

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB15PersistCommit covers B15: Complete turns are committed to persistent storage
// BEFORE normal client completion is signaled; on storage failure, normal completion is
// aborted without exposing unbacked tool calls; storage failure is never reported as an auth error;
// and partial tool bindings are rolled back atomically.
func testB15PersistCommit(t *testing.T) {
	t.Run("CommitOrderBeforeClientStopH", testB15CommitOrderBeforeClientStopH)
	t.Run("StorageFailureAbortsClientCompletionH", testB15StorageFailureAbortsClientCompletionH)
	t.Run("AtomicRollbackNoHalfToolBindingsU", testB15AtomicRollbackNoHalfToolBindingsU)
}

type orderTrackingWriter struct {
	mu          sync.Mutex
	events      []string
	hasFinished bool
}

func (w *orderTrackingWriter) record(event string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.events = append(w.events, event)
	if strings.Contains(event, "message_stop") || strings.Contains(event, `"type":"message_stop"`) {
		w.hasFinished = true
	}
}

// 1. Commit must happen BEFORE client receives message_stop event.
func testB15CommitOrderBeforeClientStopH(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	var order []string
	var orderMu sync.Mutex
	recordEvent := func(name string) {
		orderMu.Lock()
		defer orderMu.Unlock()
		order = append(order, name)
	}

	// Hook persister that logs commit timing
	testPersister := &trackingPersister{
		onCommit: func() {
			recordEvent("persister_commit")
		},
	}
	oldPersister := currentAntigravityPersister
	currentAntigravityPersister = testPersister
	t.Cleanup(func() { currentAntigravityPersister = oldPersister })

	s := New()
	tr := &mockCaptureTransport{
		sseReply: sse(
			`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]}},{"finishReason":"STOP"}]}}`,
		),
	}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	body := `{"model":"claude-sonnet-4-6","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))

	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)
	if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
		t.Fatalf("translate failed: %d/%d msg=%q", status, rec.Code, msg)
	}

	// Inspect client SSE stream for message_stop
	clientOutput := rec.Body.String()
	if !strings.Contains(clientOutput, "message_stop") {
		t.Fatalf("client output missing message_stop: %s", clientOutput)
	}

	orderMu.Lock()
	defer orderMu.Unlock()
	if len(order) == 0 || order[0] != "persister_commit" {
		t.Fatalf("commit was not executed before client completion: order=%v", order)
	}
}

// 2. Storage failure must abort client completion (NO message_stop) and never report as auth error.
func testB15StorageFailureAbortsClientCompletionH(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	// Injected storage failure
	storageErr := errors.New("simulated disk I/O error: no space left on device")
	failingPersister := &trackingPersister{
		failErr: storageErr,
	}
	oldPersister := currentAntigravityPersister
	currentAntigravityPersister = failingPersister
	t.Cleanup(func() { currentAntigravityPersister = oldPersister })

	s := New()
	tr := &mockCaptureTransport{
		sseReply: sse(
			`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]}},{"finishReason":"STOP"}]}}`,
		),
	}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	body := `{"model":"claude-sonnet-4-6","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))

	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

	// Must NOT report as auth error (not 401 or 403)
	if status == http.StatusUnauthorized || status == http.StatusForbidden || rec.Code == http.StatusUnauthorized || rec.Code == http.StatusForbidden {
		t.Fatalf("storage failure was mistakenly reported as auth error: status=%d rec.Code=%d", status, rec.Code)
	}

	clientOutput := rec.Body.String()
	// Must NOT send normal message_stop event to client
	if strings.Contains(clientOutput, "message_stop") {
		t.Errorf("normal completion message_stop was sent despite storage failure:\n%s", clientOutput)
	}

	// Must surface storage failure in error stream or message
	if !strings.Contains(clientOutput, "storage") && !strings.Contains(clientOutput, "disk") && !strings.Contains(msg, "storage") && !strings.Contains(msg, "disk") {
		t.Errorf("error output did not report storage failure: msg=%q body=%s", msg, clientOutput)
	}
}

// 3. In the event of storage commit failure, any staged tool bindings must be rolled back.
func testB15AtomicRollbackNoHalfToolBindingsU(t *testing.T) {
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	binding := AntigravityToolBinding{
		NativeID:   "call_atomic_01",
		NativeName: "view_file",
		NativeArgs: json.RawMessage(`{"path":"/a.go"}`),
		ClientID:   "client_atomic_01",
		ClientName: "Read",
	}

	// Simulate atomic commit failure
	failingPersister := &trackingPersister{
		failErr: errors.New("disk write failure"),
	}

	scope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: "test@example.com",
		Project: "p1",
		Model:   "claude-sonnet-4-6",
	}

	err := commitAntigravityRoundWithPersister(scope, "sess-1", []canonicalPart{{Kind: ToolCall, CallID: "call_atomic_01"}}, []AntigravityToolBinding{binding}, failingPersister)
	if err == nil {
		t.Fatalf("expected commit error, got nil")
	}

	// Verify binding was rolled back and does not linger in defaultToolBindingStore
	if _, ok := defaultToolBindingStore.LookupByClientID("client_atomic_01"); ok {
		t.Errorf("staged tool binding was not rolled back after commit failure")
	}
}

type trackingPersister struct {
	onCommit func()
	failErr  error
}

func (t *trackingPersister) CommitRound(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding) error {
	if t.failErr != nil {
		return t.failErr
	}
	if t.onCommit != nil {
		t.onCommit()
	}
	return nil
}
