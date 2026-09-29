package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/yetone/magpie/internal/provider"
)

// antigravitySessionScope captures the strict 4-tuple namespace for an Antigravity conversation.
type antigravitySessionScope struct {
	Caller  string // client profile: "claude_code", "opencode", "pi", "unknown"
	Account string // user email / identity
	Project string // Google Cloud project id
	Model   string // model id
}

// antigravityLedgerRecord represents a stored Antigravity conversation turn or canonical history.
type antigravityLedgerRecord struct {
	SessionID string
	Scope     antigravitySessionScope
	TurnCount int
	Data      map[string]any
}

// antigravitySessionLedger stores conversation records scoped strictly by (Caller, Account, Project, Model, SessionRef).
type antigravitySessionLedger struct {
	sync.RWMutex
	records map[string]*antigravityLedgerRecord
}

var defaultAntigravityLedger = &antigravitySessionLedger{
	records: make(map[string]*antigravityLedgerRecord),
}

// antigravityParentRecord represents a cached parent response output for Responses API incremental requests.
type antigravityParentRecord struct {
	ResponseID string
	SessionID  string
	Scope      antigravitySessionScope
	Messages   []Message
}

type antigravityParentStore struct {
	sync.RWMutex
	records map[string]*antigravityParentRecord
}

var defaultAntigravityParentStore = &antigravityParentStore{
	records: make(map[string]*antigravityParentRecord),
}

func buildParentStoreKey(scope antigravitySessionScope, responseID string) string {
	if responseID == "" {
		return ""
	}
	return scope.Caller + "\x00" + scope.Account + "\x00" + scope.Project + "\x00" + scope.Model + "\x00" + responseID
}

func (s *antigravityParentStore) StoreParent(scope antigravitySessionScope, responseID, sessionID string, msgs []Message) {
	key := buildParentStoreKey(scope, responseID)
	if key == "" {
		return
	}
	s.Lock()
	s.records[key] = &antigravityParentRecord{
		ResponseID: responseID,
		SessionID:  sessionID,
		Scope:      scope,
		Messages:   msgs,
	}
	s.Unlock()
}

func (s *antigravityParentStore) LookupParent(scope antigravitySessionScope, responseID string) (*antigravityParentRecord, bool) {
	key := buildParentStoreKey(scope, responseID)
	if key == "" {
		return nil, false
	}
	s.RLock()
	rec, ok := s.records[key]
	s.RUnlock()
	return rec, ok
}

// resolveAntigravityParentResponse resolves parent response history for Responses API incremental requests.
func resolveAntigravityParentResponse(scope antigravitySessionScope, req *Request) error {
	if req == nil || req.PreviousResponseID == "" {
		return nil
	}
	parent, ok := defaultAntigravityParentStore.LookupParent(scope, req.PreviousResponseID)
	if !ok {
		return fmt.Errorf("previous_response_id not found in scope: %s", req.PreviousResponseID)
	}
	// Prepend parent history messages to current incremental messages without duplicates
	full := append(slices.Clone(parent.Messages), req.Messages...)
	req.Messages = mergeTurns(full)
	return nil
}

// detectClientProfile detects the caller profile at the protocol/request boundary.
func detectClientProfile(r *http.Request, from provider.Protocol, body []byte) string {
	if r != nil {
		if p := strings.TrimSpace(r.Header.Get("X-Magpie-Client-Profile")); p != "" {
			return p
		}
		ua := strings.ToLower(r.Header.Get("User-Agent"))
		if strings.Contains(ua, "claude-code") || strings.Contains(ua, "claudecode") {
			return "claude_code"
		}
		if strings.Contains(ua, "opencode") {
			return "opencode"
		}
	}
	// Anthropic Messages payload carrying metadata.user_id is Claude Code's verified wire evidence
	if from == provider.Anthropic && len(body) > 0 {
		var req struct {
			Metadata struct {
				UserID string `json:"user_id"`
			} `json:"metadata"`
		}
		if json.Unmarshal(body, &req) == nil && req.Metadata.UserID != "" {
			return "claude_code"
		}
	}
	return "unknown"
}

// extractSessionReference extracts the logical session reference strictly based on the client profile boundary.
// Private headers and body metadata are only recognized within their corresponding profile.
func extractSessionReference(profile string, in http.Header, body []byte) string {
	// Universal explicit header accepted across all profiles
	if in != nil {
		if v := strings.TrimSpace(in.Get("X-Magpie-Session")); v != "" {
			return v
		}
	}

	switch profile {
	case "claude_code":
		if in != nil {
			if v := strings.TrimSpace(in.Get("x-claude-code-session-id")); v != "" {
				return v
			}
		}
		if len(body) > 0 {
			var req struct {
				Metadata struct {
					UserID string `json:"user_id"`
				} `json:"metadata"`
			}
			if json.Unmarshal(body, &req) == nil && req.Metadata.UserID != "" {
				var inner struct {
					SessionID string `json:"session_id"`
				}
				if json.Unmarshal([]byte(req.Metadata.UserID), &inner) == nil && inner.SessionID != "" {
					return inner.SessionID
				}
				return req.Metadata.UserID
			}
		}
	case "opencode":
		if in != nil {
			if v := strings.TrimSpace(in.Get("x-opencode-session")); v != "" {
				return v
			}
		}
	case "pi":
		if in != nil {
			if v := strings.TrimSpace(in.Get("x-session-affinity")); v != "" {
				return v
			}
			if v := strings.TrimSpace(in.Get("x-session-id")); v != "" {
				return v
			}
		}
	}
	return ""
}

// buildSessionLedgerKey computes the composite key for session storage.
// Empty session references produce an empty key (no default fallback to first message hash).
// Long session IDs are never truncated.
func buildSessionLedgerKey(scope antigravitySessionScope, sessionRef string) string {
	if sessionRef == "" {
		return ""
	}
	return scope.Caller + "\x00" + scope.Account + "\x00" + scope.Project + "\x00" + scope.Model + "\x00" + sessionRef
}

// Store saves a record in the ledger under the scoped key.
func (l *antigravitySessionLedger) Store(scope antigravitySessionScope, sessionRef string, rec *antigravityLedgerRecord) error {
	key := buildSessionLedgerKey(scope, sessionRef)
	if key == "" {
		return nil
	}
	l.Lock()
	l.records[key] = rec
	l.Unlock()
	return nil
}

// Lookup finds a record in the ledger under the scoped key.
func (l *antigravitySessionLedger) Lookup(scope antigravitySessionScope, sessionRef string) (*antigravityLedgerRecord, bool) {
	key := buildSessionLedgerKey(scope, sessionRef)
	if key == "" {
		return nil, false
	}
	l.RLock()
	rec, ok := l.records[key]
	l.RUnlock()
	if ok {
		return rec, true
	}

	// Fallback: recover committed turn from persistent storage across restarts
	if currentAntigravityPersister != nil {
		round, err := currentAntigravityPersister.LoadRound(scope, sessionRef)
		if err == nil && round != nil {
			if len(round.Bindings) > 0 {
				for i := range round.Bindings {
					if round.Bindings[i].Scope.Account == "" {
						round.Bindings[i].Scope = scope
					}
				}
				_ = defaultToolBindingStore.Bind(round.Bindings...)
			}
			newRec := &antigravityLedgerRecord{
				SessionID: sessionRef,
				Scope:     scope,
				Data: map[string]any{
					"history": round.Canonical,
				},
			}
			l.Lock()
			l.records[key] = newRec
			l.Unlock()
			return newRec, true
		}
	}
	return nil, false
}

// verifyCommittedHistory verifies that the submitted history matches committed canonical history without tampering.
func verifyCommittedHistory(scope antigravitySessionScope, committed []canonicalPart, req *Request) error {
	if len(committed) == 0 || req == nil {
		return nil
	}
	var submitted []canonicalPart
	for _, m := range req.Messages {
		if m.Role == "assistant" {
			for _, p := range m.Parts {
				switch p.Kind {
				case Thinking:
					submitted = append(submitted, canonicalPart{Kind: Thinking, Text: p.Text, ThoughtSignature: p.Signature})
				case Text:
					submitted = append(submitted, canonicalPart{Kind: Text, Text: p.Text, ThoughtSignature: p.Signature})
				case ToolCall:
					submitted = append(submitted, canonicalPart{Kind: ToolCall, CallID: p.ID, Name: p.Name, Args: string(p.Args)})
				}
			}
		} else if m.Role == "user" {
			for _, p := range m.Parts {
				if p.Kind == ToolResult {
					argsStr := ""
					if p.IsError {
						argsStr = `{"is_error":true}`
					}
					submitted = append(submitted, canonicalPart{Kind: ToolResult, CallID: p.CallID, Name: p.Name, Text: p.Text, Args: argsStr})
				}
			}
		}
	}

	if len(submitted) < len(committed) {
		return fmt.Errorf("submitted history is missing committed turns: got %d, want at least %d", len(submitted), len(committed))
	}

	for i, exp := range committed {
		act := submitted[i]
		if act.Kind != exp.Kind {
			return fmt.Errorf("history kind mismatch at part %d: committed %v != submitted %v", i, exp.Kind, act.Kind)
		}
		if exp.Kind == Text && act.Text != exp.Text {
			return fmt.Errorf("history text mismatch at part %d: committed %q != submitted %q", i, exp.Text, act.Text)
		}
		if exp.Kind == Thinking {
			if act.Text != exp.Text {
				return fmt.Errorf("history thinking text mismatch: committed %q != submitted %q", exp.Text, act.Text)
			}
			if exp.ThoughtSignature != "" && act.ThoughtSignature != "" && act.ThoughtSignature != exp.ThoughtSignature {
				return fmt.Errorf("history thinking signature mismatch: committed %q != submitted %q", exp.ThoughtSignature, act.ThoughtSignature)
			}
		}
		if exp.Kind == ToolCall {
			actCallID := act.CallID
			if b, ok := defaultToolBindingStore.LookupByClientID(act.CallID); ok {
				actCallID = b.NativeID
			}
			if exp.CallID != "" && actCallID != exp.CallID {
				return fmt.Errorf("history tool call ID mismatch: committed %q != submitted %q", exp.CallID, act.CallID)
			}
			if exp.Args != "" {
				match, err := jsonEqualExact([]byte(act.Args), []byte(exp.Args))
				if err != nil || !match {
					if b, ok := defaultToolBindingStore.LookupByClientID(act.CallID); ok && len(b.ClientArgs) > 0 {
						cm, _ := jsonEqualExact([]byte(act.Args), b.ClientArgs)
						if !cm {
							return fmt.Errorf("history tool args tampered for call %q: committed %s != submitted %s", exp.CallID, exp.Args, act.Args)
						}
					} else {
						return fmt.Errorf("history tool args tampered for call %q: committed %s != submitted %s", exp.CallID, exp.Args, act.Args)
					}
				}
			}
		}
		if exp.Kind == ToolResult {
			if exp.CallID != "" && act.CallID != exp.CallID {
				return fmt.Errorf("history tool result CallID mismatch: committed %q != submitted %q", exp.CallID, act.CallID)
			}
			if exp.Text != "" && act.Text != exp.Text {
				return fmt.Errorf("history tool result text tampered for call %q: committed %q != submitted %q", exp.CallID, exp.Text, act.Text)
			}
			if exp.Args == `{"is_error":true}` && act.Args != `{"is_error":true}` {
				return fmt.Errorf("history tool result is_error tampered for call %q", exp.CallID)
			}
		}
	}
	return nil
}

func hasAssistantTurn(messages []Message) bool {
	for _, m := range messages {
		if m.Role == "assistant" {
			return true
		}
	}
	return false
}

// validateAntigravitySessionResume validates session reference and history consistency when strict resume is requested.
func validateAntigravitySessionResume(scope antigravitySessionScope, in http.Header, body []byte, from provider.Protocol, req *Request) error {
	explicitResumeStrict := in != nil && strings.EqualFold(in.Get("X-Antigravity-Resume"), "strict")
	serverStrict := effectiveAntigravityMode(in) == AntigravityModeStrict
	if !explicitResumeStrict && !serverStrict {
		return nil
	}
	if !explicitResumeStrict && req != nil && !hasAssistantTurn(req.Messages) && req.PreviousResponseID == "" {
		return nil
	}
	sessionRef := extractSessionReference(scope.Caller, in, body)
	if sessionRef == "" {
		return fmt.Errorf("antigravity session resume requires a valid session reference")
	}
	rec, ok := defaultAntigravityLedger.Lookup(scope, sessionRef)
	if !ok {
		if currentAntigravityPersister != nil {
			if _, err := currentAntigravityPersister.LoadRound(scope, sessionRef); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("antigravity session record corrupted on disk: %w", err)
			}
		}
		return fmt.Errorf("antigravity session not found in scope: %s", sessionRef)
	}
	if hist, ok := rec.Data["history"].([]canonicalPart); ok && len(hist) > 0 {
		if req != nil {
			if err := verifyCommittedHistory(scope, hist, req); err != nil {
				return err
			}
		}
	}
	return nil
}
