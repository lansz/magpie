package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// testB17BranchConcurrency covers B17: Concurrent branches, fork revisions, and concurrent commits
// do not collide or cross-contaminate call IDs; stale revisions cannot overwrite newer states (optimistic
// concurrency check); and storage locks are NEVER held across slow network wait durations.
func testB17BranchConcurrency(t *testing.T) {
	t.Run("ConcurrentBranchesNoCallIDCollision", testB17ConcurrentBranchesNoCallIDCollision)
	t.Run("StaleRevisionCannotOverwriteNewState", testB17StaleRevisionCannotOverwriteNewState)
	t.Run("NoLockHeldDuringNetworkWait", testB17NoLockHeldDuringNetworkWait)
}

// 1. Concurrent branches from the same parent do not leak or cross-contaminate tool call IDs.
func testB17ConcurrentBranchesNoCallIDCollision(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	const (
		baseSession = "fork-root-session"
		userAlice   = "alice@example.com"
		projAlice   = "proj-alice"
		model       = "claude-sonnet-4-6"
	)

	scope := antigravitySessionScope{
		Caller:  "claude_code",
		Account: userAlice,
		Project: projAlice,
		Model:   model,
	}

	persister := newFileAntigravityPersister(filepath.Join(cacheDir, "antigravity_rounds"))
	oldPersister := currentAntigravityPersister
	currentAntigravityPersister = persister
	t.Cleanup(func() { currentAntigravityPersister = oldPersister })

	const branchCount = 5
	var wg sync.WaitGroup
	errCh := make(chan error, branchCount)

	for b := 0; b < branchCount; b++ {
		wg.Add(1)
		branchID := fmt.Sprintf("branch_%d", b)
		callID := fmt.Sprintf("call_branch_%d_tool_99", b)
		clientID := fmt.Sprintf("toolu_branch_%d", b)

		go func(bID, cID, clID string) {
			defer wg.Done()

			binding := AntigravityToolBinding{
				NativeID:   cID,
				NativeName: "tool_for_" + bID,
				NativeArgs: json.RawMessage(`{"branch":"` + bID + `"}`),
				ClientID:   clID,
				ClientName: "tool_for_" + bID,
			}

			canonical := []canonicalPart{
				{Kind: ToolCall, CallID: cID, Name: "tool_for_" + bID, Args: `{"branch":"` + bID + `"}`},
			}

			// Commit round under explicit branch
			sessionWithBranch := baseSession + ":" + bID
			if err := commitAntigravityRoundWithPersister(scope, sessionWithBranch, canonical, []AntigravityToolBinding{binding}, persister); err != nil {
				errCh <- fmt.Errorf("branch %s commit failed: %w", bID, err)
				return
			}

			// Verify in store
			loadedBinding, ok := defaultToolBindingStore.LookupByClientID(clID)
			if !ok || loadedBinding.NativeID != cID {
				errCh <- fmt.Errorf("branch %s binding mismatch: got %v, want nativeID=%s", bID, loadedBinding, cID)
				return
			}
		}(branchID, callID, clientID)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatalf("concurrent branch error: %v", err)
	}
}

// 2. Anti-pattern: Stale revision cannot overwrite newer state (last-write does NOT win blindly).
func testB17StaleRevisionCannotOverwriteNewState(t *testing.T) {
	cacheDir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", cacheDir)

	const (
		sessionID = "revision-conflict-session"
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

	persister := newFileAntigravityPersister(filepath.Join(cacheDir, "antigravity_rounds"))
	oldPersister := currentAntigravityPersister
	currentAntigravityPersister = persister
	t.Cleanup(func() { currentAntigravityPersister = oldPersister })

	// State 1: Commit Revision 1
	rev1Parts := []canonicalPart{{Kind: Text, Text: "Turn 1 answer."}}
	if err := persister.CommitRoundWithRevision(scope, sessionID, rev1Parts, nil, 1); err != nil {
		t.Fatalf("Rev 1 commit failed: %v", err)
	}

	// State 2: Commit Revision 2 (Newer state)
	rev2Parts := []canonicalPart{{Kind: Text, Text: "Turn 2 updated answer."}}
	if err := persister.CommitRoundWithRevision(scope, sessionID, rev2Parts, nil, 2); err != nil {
		t.Fatalf("Rev 2 commit failed: %v", err)
	}

	// Anti-pattern attempt: Stale response with Revision 1 tries to overwrite Revision 2!
	staleParts := []canonicalPart{{Kind: Text, Text: "Stale turn 1 overwrite attempt!"}}
	err := persister.CommitRoundWithRevision(scope, sessionID, staleParts, nil, 1)
	if err == nil {
		t.Fatalf("expected stale revision overwrite to be rejected, got nil")
	}

	// Verify State 2 remains intact
	round, err := persister.LoadRound(scope, sessionID)
	if err != nil {
		t.Fatalf("LoadRound failed: %v", err)
	}
	if len(round.Canonical) == 0 || round.Canonical[0].Text != "Turn 2 updated answer." {
		t.Errorf("newer state was corrupted by stale write: %v", round.Canonical)
	}
}

// 3. Storage lock is NEVER held across slow network wait durations.
func testB17NoLockHeldDuringNetworkWait(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
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

	// Slow upstream transport simulating network latency (200ms delay)
	slowTransport := &slowMockTransport{
		delay: 200 * time.Millisecond,
		sseReply: sse(
			`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"slow answer"}]}},{"finishReason":"STOP"}]}}`,
		),
	}

	s := New()
	s.client = &http.Client{Transport: slowTransport}

	p := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-test")

	var wg sync.WaitGroup
	wg.Add(1)

	var slowStarted atomic.Bool
	go func() {
		defer wg.Done()
		slowStarted.Store(true)
		body := `{"model":"claude-sonnet-4-6","stream":true,"messages":[{"role":"user","content":"slow question"}]}`
		rec := httptest.NewRecorder()
		req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
		var u Usage
		s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, model, []byte(body), &u)
	}()

	// Wait until slow request has definitely entered its network wait
	for !slowStarted.Load() {
		time.Sleep(10 * time.Millisecond)
	}
	time.Sleep(50 * time.Millisecond)

	// In parallel, another goroutine reads/commits to ledger. It must NOT be blocked!
	quickStart := time.Now()
	quickDone := make(chan bool, 1)

	go func() {
		// Quick storage read/write
		_ = defaultAntigravityLedger.Store(scope, "parallel-fast", &antigravityLedgerRecord{SessionID: "parallel-fast", Scope: scope})
		_, _ = defaultAntigravityLedger.Lookup(scope, "parallel-fast")
		quickDone <- true
	}()

	select {
	case <-quickDone:
		dur := time.Since(quickStart)
		if dur > 100*time.Millisecond {
			t.Errorf("parallel storage operation took %v, indicating it was blocked by slow request", dur)
		}
	case <-time.After(150 * time.Millisecond):
		t.Fatalf("parallel storage operation DEADLOCKED or was blocked by slow network request")
	}

	wg.Wait()
}

type slowMockTransport struct {
	delay    time.Duration
	sseReply string
}

func (s *slowMockTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	time.Sleep(s.delay)
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(s.sseReply)),
	}
	resp.Header.Set("Content-Type", "text/event-stream")
	return resp, nil
}
