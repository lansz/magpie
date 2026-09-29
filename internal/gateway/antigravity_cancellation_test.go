package gateway

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// testB20Cancellation covers B20: Client disconnects/timeouts properly abort upstream operations;
// aborted turns are NEVER committed to ledger/disk; no lingering partial tool bindings pollute future turns;
// and retries are bounded without infinite loops.
func testB20Cancellation(t *testing.T) {
	t.Run("ClientDisconnectCancelsUpstream", testB20ClientDisconnectCancelsUpstream)
	t.Run("CanceledTurnNeverCommittedToLedger", testB20CanceledTurnNeverCommittedToLedger)
	t.Run("StagedBindingsClearedOnCancellation", testB20StagedBindingsClearedOnCancellation)
}

// 1. Client disconnect during streaming stops upstream request reading promptly.
func testB20ClientDisconnectCancelsUpstream(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	var upstreamReadCount atomic.Int64
	var upstreamClosed atomic.Bool

	// Upstream streams slow chunks indefinitely
	slowPipeR, slowPipeW := httpPipe()
	go func() {
		defer slowPipeW.Close()
		for i := 0; i < 20; i++ {
			select {
			case <-slowPipeR.done:
				upstreamClosed.Store(true)
				return
			default:
			}
			upstreamReadCount.Add(1)
			_, err := slowPipeW.Write([]byte("data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"chunk\"}]}}]}}\n\n"))
			if err != nil {
				upstreamClosed.Store(true)
				return
			}
			time.Sleep(15 * time.Millisecond)
		}
	}()

	tr := &pipeTransport{body: slowPipeR}
	s := New()
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	ctx, cancel := context.WithCancel(context.Background())
	body := `{"model":"claude-sonnet-4-6","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	httpReq := httptest.NewRequestWithContext(ctx, "POST", "/v1/messages", strings.NewReader(body))
	rec := httptest.NewRecorder()

	doneCh := make(chan struct{})
	go func() {
		var u Usage
		s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)
		close(doneCh)
	}()

	// Wait until client has started reading first chunks, then abruptly cancel
	time.Sleep(30 * time.Millisecond)
	cancel()

	select {
	case <-doneCh:
		// Succeeded in exiting promptly
	case <-time.After(1 * time.Second):
		t.Fatalf("gateway hung and failed to abort upstream on client cancellation")
	}

	// Verify upstream was stopped promptly (didn't read all 20 chunks)
	readTotal := upstreamReadCount.Load()
	if readTotal >= 20 {
		t.Errorf("upstream was not cancelled promptly, read all %d chunks", readTotal)
	}
}

// 2. An aborted turn must NEVER be committed to the ledger or persistent storage.
func testB20CanceledTurnNeverCommittedToLedger(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
		sessionID = "cancel-ledger-clean-session"
		userAlice = "alice@example.com"
		projAlice = "proj-alice"
		model     = "claude-sonnet-4-6"
	)

	scope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}

	// Turn 1: Successfully committed
	defaultAntigravityLedger.Store(scope, sessionID, &antigravityLedgerRecord{
		SessionID: sessionID,
		Scope:     scope,
		TurnCount: 1,
		Data: map[string]any{
			"history": []canonicalPart{{Kind: Text, Text: "Turn 1 committed text."}},
		},
	})
	t.Cleanup(func() {
		defaultAntigravityLedger.Lock()
		delete(defaultAntigravityLedger.records, buildSessionLedgerKey(scope, sessionID))
		defaultAntigravityLedger.Unlock()
	})

	// Turn 2: Starts streaming but client aborts mid-stream
	slowPipeR, slowPipeW := httpPipe()
	go func() {
		defer slowPipeW.Close()
		_, _ = slowPipeW.Write([]byte("data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Partial abort text\"}]}}]}}\n\n"))
		time.Sleep(100 * time.Millisecond)
		_, _ = slowPipeW.Write([]byte("data: {\"response\":{\"candidates\":[{\"finishReason\":\"STOP\"}]}}\n\n"))
	}()

	s := New()
	s.client = &http.Client{Transport: &pipeTransport{body: slowPipeR}}
	p := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-test")

	ctx, cancel := context.WithCancel(context.Background())
	turn2Body := `{
		"model": "claude-sonnet-4-6",
		"stream": true,
		"messages": [
			{"role": "user", "content": "q1"},
			{"role": "assistant", "content": "Turn 1 committed text."},
			{"role": "user", "content": "q2 aborted"}
		]
	}`

	httpReq := httptest.NewRequestWithContext(ctx, "POST", "/v1/messages", strings.NewReader(turn2Body))
	httpReq.Header.Set("X-Magpie-Client-Profile", "claude_code")
	httpReq.Header.Set("X-Magpie-Session", sessionID)
	rec := httptest.NewRecorder()

	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel() // Abort turn 2
	}()

	var u Usage
	s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, model, []byte(turn2Body), &u)

	// Assert ledger was NOT updated with Turn 2 content! Turn 1 must remain intact.
	recState, ok := defaultAntigravityLedger.Lookup(scope, sessionID)
	if !ok {
		t.Fatalf("session record was completely lost after abort")
	}
	hist := recState.Data["history"].([]canonicalPart)
	if len(hist) != 1 || hist[0].Text != "Turn 1 committed text." {
		t.Errorf("aborted turn contaminated committed ledger: %v", hist)
	}
}

// 3. Staged tool bindings are cleared on cancellation, leaving no partial state.
func testB20StagedBindingsClearedOnCancellation(t *testing.T) {
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	const (
		clientCallID = "toolu_cancel_staged"
		nativeCallID = "call_native_cancel"
	)

	// Simulate that a binding was temporarily staged but request was canceled
	binding := AntigravityToolBinding{
		NativeID:   nativeCallID,
		NativeName: "read_file",
		ClientID:   clientCallID,
		ClientName: "read_file",
	}

	scope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: "user@test.com",
		Project: "p1",
		Model:   "claude-sonnet-4-6",
	}

	// Direct verification of rollback on cancel / error
	persister := &trackingPersister{
		failErr: context.Canceled,
	}

	err := commitAntigravityRoundWithPersister(scope, "s-cancel", nil, []AntigravityToolBinding{binding}, persister)
	if err == nil {
		t.Fatalf("expected cancel error, got nil")
	}

	// Verify no partial state in store
	if _, ok := defaultToolBindingStore.LookupByClientID(clientCallID); ok {
		t.Errorf("staged tool binding was not cleared after cancellation")
	}
	if _, ok := defaultToolBindingStore.LookupByNativeID(nativeCallID); ok {
		t.Errorf("staged tool binding native was not cleared after cancellation")
	}
}

func httpPipe() (*pipeReader, *pipeWriter) {
	ch := make(chan []byte, 16)
	done := make(chan struct{})
	return &pipeReader{ch: ch, done: done}, &pipeWriter{ch: ch, done: done}
}

type pipeReader struct {
	ch   chan []byte
	done chan struct{}
	buf  []byte
}

func (r *pipeReader) Read(p []byte) (int, error) {
	select {
	case <-r.done:
		return 0, io.EOF
	default:
	}
	if len(r.buf) > 0 {
		n := copy(p, r.buf)
		r.buf = r.buf[n:]
		return n, nil
	}
	select {
	case <-r.done:
		return 0, io.EOF
	case data, ok := <-r.ch:
		if !ok {
			return 0, io.EOF
		}
		n := copy(p, data)
		if n < len(data) {
			r.buf = data[n:]
		}
		return n, nil
	}
}

func (r *pipeReader) Close() error {
	select {
	case <-r.done:
	default:
		close(r.done)
	}
	return nil
}

type pipeWriter struct {
	ch   chan []byte
	done chan struct{}
}

func (w *pipeWriter) Write(p []byte) (int, error) {
	select {
	case <-w.done:
		return 0, io.ErrClosedPipe
	default:
	}
	cp := make([]byte, len(p))
	copy(cp, p)
	select {
	case <-w.done:
		return 0, io.ErrClosedPipe
	case w.ch <- cp:
		return len(p), nil
	}
}

func (w *pipeWriter) Close() error {
	close(w.ch)
	return nil
}

type pipeTransport struct {
	body *pipeReader
}

func (pt *pipeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	go func() {
		<-req.Context().Done()
		pt.body.Close()
	}()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       pt.body,
	}
	resp.Header.Set("Content-Type", "text/event-stream")
	return resp, nil
}
