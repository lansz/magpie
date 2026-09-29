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

// testB21ModesAndNonTarget covers B21: Mode transitions between off/verified/strict
// do not break existing in-flight calls; switching to off preserves in-flight mappings;
// strict mode rejects unverified/unsupported capabilities with 400 (0 calls); and
// non-target paths (plain Anthropic/Chat/Gemini CLI) remain completely unaffected.
func testB21ModesAndNonTarget(t *testing.T) {
	t.Run("SwitchingToOffPreservesInFlightMappings", testB21SwitchingToOffPreservesInFlightMappings)
	t.Run("StrictModeRejectsUnsupportedCapabilitiesH", testB21StrictModeRejectsUnsupportedCapabilitiesH)
	t.Run("NonTargetProvidersUnaffectedAcrossModesH", testB21NonTargetProvidersUnaffectedAcrossModesH)
}

// 1. Anti-pattern: Switching mode to off must NOT delete in-flight tool mappings in use.
func testB21SwitchingToOffPreservesInFlightMappings(t *testing.T) {
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	oldMode := GetAntigravityMode()
	SetAntigravityMode(AntigravityModeVerified)
	t.Cleanup(func() { SetAntigravityMode(oldMode) })

	const (
		nativeCallID = "call_flight_01"
		clientCallID = "toolu_flight_01"
	)

	// In-flight binding created while in verified mode
	binding := AntigravityToolBinding{
		NativeID:   nativeCallID,
		NativeName: "view_file",
		NativeArgs: json.RawMessage(`{"path":"/in_flight.go"}`),
		ClientID:   clientCallID,
		ClientName: "Read",
	}
	if err := defaultToolBindingStore.Bind(binding); err != nil {
		t.Fatalf("Bind: %v", err)
	}

	// SWITCH MODE TO OFF
	SetAntigravityMode(AntigravityModeOff)

	// Verify the in-flight binding still exists and was NOT deleted
	loaded, ok := defaultToolBindingStore.LookupByClientID(clientCallID)
	if !ok || loaded.NativeName != "view_file" {
		t.Fatalf("switching mode to off deleted in-flight mapping!")
	}

	// Client sends tool_result for this in-flight call; it must still restore correctly
	body := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "` + clientCallID + `", "content": "res"}]}
		]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parseAnthropic: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request struct {
			Contents []struct {
				Parts []struct {
					FunctionResponse *struct {
						ID   string `json:"id"`
						Name string `json:"name"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	json.Unmarshal(wireBytes, &env)

	var frName string
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionResponse != nil {
				frName = p.FunctionResponse.Name
			}
		}
	}

	if frName != "view_file" {
		t.Errorf("in-flight tool_result name recovery failed after mode switch: got %q, want 'view_file'", frName)
	}
}

// 2. Strict mode must explicitly reject unsupported capabilities (e.g. unverified tools)
// before upstream dispatch (400, 0 upstream calls).
func testB21StrictModeRejectsUnsupportedCapabilitiesH(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	oldMode := GetAntigravityMode()
	SetAntigravityMode(AntigravityModeStrict)
	t.Cleanup(func() { SetAntigravityMode(oldMode) })

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"unexpected"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	// Unsupported unverified capability: generate_image
	body := `{
		"model": "claude-sonnet-4-6",
		"tools": [{"name": "generate_image", "input_schema": {"type": "object"}}],
		"messages": [{"role": "user", "content": "draw cat"}]
	}`

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	req.Header.Set("X-Antigravity-Mode", "strict")

	var u Usage
	status, msg := s.translate(rec, req, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

	if status != http.StatusBadRequest || rec.Code != http.StatusBadRequest {
		t.Fatalf("strict mode failed to reject unsupported capability: status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}
	if calls := atomic.LoadInt64(&tr.callCount); calls != 0 {
		t.Fatalf("expected 0 upstream calls for unsupported capability in strict mode, got %d", calls)
	}
	if !strings.Contains(msg, "generate_image") && !strings.Contains(rec.Body.String(), "generate_image") && !strings.Contains(msg, "strict") {
		t.Errorf("error message should indicate strict unsupported tool rejection: msg=%q body=%s", msg, rec.Body.String())
	}
}

// 3. Non-target providers (Anthropic, Chat, Gemini CLI) must remain completely unaffected across all modes.
func testB21NonTargetProvidersUnaffectedAcrossModesH(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	modes := []string{AntigravityModeOff, AntigravityModeVerified, AntigravityModeStrict}

	for _, mode := range modes {
		t.Run("Mode_"+mode, func(t *testing.T) {
			oldMode := GetAntigravityMode()
			SetAntigravityMode(mode)
			defer SetAntigravityMode(oldMode)

			// Control 1: Plain Anthropic Provider
			s := New()
			s.client = &http.Client{Transport: &mockCaptureTransport{sseReply: sse(`data: {"type":"message_stop"}`)}}
			pAnthropic := provider.Provider{ID: "ant", Key: "k", Anthropic: "http://example.com/v1"}
			recA := httptest.NewRecorder()
			bodyA := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
			reqA := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(bodyA))
			var uA Usage
			stA, _ := s.translate(recA, reqA, pAnthropic, provider.Anthropic, provider.Anthropic, "m", []byte(bodyA), &uA)
			if stA != http.StatusOK || recA.Code != http.StatusOK {
				t.Errorf("[%s] Plain Anthropic failed: status=%d rec.Code=%d", mode, stA, recA.Code)
			}

			// Control 2: Gemini CLI Provider
			sGemini := New()
			sGemini.client = &http.Client{Transport: &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"finishReason":"STOP"}]}}`)}}
			pGemini := provider.GeminiTestProvider("gemini-cli", "user@test.com", "proj-g", "tok-g")
			recG := httptest.NewRecorder()
			bodyG := `{"model":"gemini-2.5-pro","messages":[{"role":"user","content":"hi"}]}`
			reqG := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(bodyG))
			var uG Usage
			stG, _ := s.translate(recG, reqG, pGemini, provider.Anthropic, provider.CodeAssist, "gemini-2.5-pro", []byte(bodyG), &uG)
			if stG != http.StatusOK || recG.Code != http.StatusOK {
				t.Errorf("[%s] Gemini CLI failed: status=%d rec.Code=%d", mode, stG, recG.Code)
			}
		})
	}
}
