package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
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
	return rec, ok
}

// validateAntigravitySessionResume validates session reference when strict resume is requested.
func validateAntigravitySessionResume(scope antigravitySessionScope, in http.Header, body []byte) error {
	strict := false
	if in != nil && strings.EqualFold(in.Get("X-Antigravity-Resume"), "strict") {
		strict = true
	}
	if !strict {
		return nil
	}
	sessionRef := extractSessionReference(scope.Caller, in, body)
	if sessionRef == "" {
		return fmt.Errorf("antigravity session resume requires a valid session reference")
	}
	if _, ok := defaultAntigravityLedger.Lookup(scope, sessionRef); !ok {
		return fmt.Errorf("antigravity session not found in scope: %s", sessionRef)
	}
	return nil
}
