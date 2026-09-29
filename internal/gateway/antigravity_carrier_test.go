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

// testB12Carrier covers B12: Client carriers act as transparent channels for signatures;
// signatures are byte-exact and not re-encoded; Claude thinking.signature is preserved onto
// the following semantic part; carrier-less clients transparently recover signatures from ledger;
// and corrupted carrier payloads are rejected before upstream dispatch (0 calls).
func testB12Carrier(t *testing.T) {
	t.Run("ByteExactTransparency", testB12ByteExactTransparency)
	t.Run("ClaudeThinkingSignatureAttachedToNextPartOnWire", testB12ClaudeThinkingSignatureAttachedToNextPartOnWire)
	t.Run("CarrierlessMultiTurnRecoversFromLedgerH", testB12CarrierlessMultiTurnRecoversFromLedgerH)
	t.Run("CorruptedCarrierFormatRejectedH", testB12CorruptedCarrierFormatRejectedH)
}

// 1. Signatures transmitted via carrier must remain byte-exact without re-encoding or trimming.
func testB12ByteExactTransparency(t *testing.T) {
	const exactOpaqueSig = "SIG_v1_raw+bytes/with=padding=="

	body := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "hello"},
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "analyzing...", "signature": "` + exactOpaqueSig + `"},
				{"type": "text", "text": "Here is the response."}
			]},
			{"role": "user", "content": "continue"}
		]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request struct {
			Contents []struct {
				Role  string `json:"role"`
				Parts []struct {
					Text             string `json:"text"`
					ThoughtSignature string `json:"thoughtSignature"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	var foundSig string
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.ThoughtSignature != "" {
				foundSig = p.ThoughtSignature
			}
		}
	}

	if foundSig != exactOpaqueSig {
		t.Errorf("signature on wire was modified: got %q, want %q", foundSig, exactOpaqueSig)
	}
}

// 2. Claude thinking.signature from client history must attach to the following semantic part on wire.
func testB12ClaudeThinkingSignatureAttachedToNextPartOnWire(t *testing.T) {
	const claudeSig = "opaque_claude_carrier_signature_999"

	body := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "step 1"},
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "step 1 plan", "signature": "` + claudeSig + `"},
				{"type": "tool_use", "id": "t_step1", "name": "do_task", "input": {"cmd": "run"}}
			]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t_step1", "content": "ok"}]}
		]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request struct {
			Contents []struct {
				Role  string `json:"role"`
				Parts []struct {
					FunctionCall *struct {
						ID   string `json:"id"`
						Name string `json:"name"`
					} `json:"functionCall"`
					ThoughtSignature string `json:"thoughtSignature"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	var callSig string
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				callSig = p.ThoughtSignature
			}
		}
	}

	if callSig != claudeSig {
		t.Errorf("thinking.signature did not attach to following functionCall: got %q, want %q", callSig, claudeSig)
	}
}

// 3. H-Layer: Carrier-less client multi-turn transparently recovers signature from ledger.
func testB12CarrierlessMultiTurnRecoversFromLedgerH(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
		sessionID = "carrierless-session-001"
		realSig   = "real_ledger_recovered_signature_abc_777"
		model     = "claude-sonnet-4-6"
		userAlice = "alice@example.com"
		projAlice = "proj-alice"
	)

	// Ledger pre-stores previous turn's tool call with signature
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)
	defaultToolBindingStore.Bind(AntigravityToolBinding{
		NativeID:        "call_native_777",
		NativeName:      "read_file",
		NativeArgs:      json.RawMessage(`{"path":"/a.txt"}`),
		NativeSignature: realSig,
		ClientID:        "t_carrierless",
		ClientName:      "read_file",
	})

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"completed"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", userAlice, projAlice, "tok-test")

	// Notice: Client sends carrier-less history (NO thinking block, NO signature field)
	body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "read please"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "t_carrierless", "name": "read_file", "input": {"path":"/a.txt"}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t_carrierless", "content": "data"}]}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(body))
	httpReq.Header.Set("X-Magpie-Session", sessionID)

	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, model, []byte(body), &u)

	if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
		t.Fatalf("translate returned status %d/%d msg=%q", status, rec.Code, msg)
	}
	if calls := atomic.LoadInt64(&tr.callCount); calls != 1 {
		t.Fatalf("expected 1 upstream call, got %d", calls)
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
	json.Unmarshal(tr.capturedBody, &env)

	var wireSig string
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				wireSig = p.ThoughtSignature
			}
		}
	}

	if wireSig != realSig {
		t.Errorf("carrier-less turn failed to recover real signature from ledger: got %q, want %q", wireSig, realSig)
	}
}

// 4. H-Layer: Corrupted carrier format (e.g. signature as object instead of string) is rejected with 400.
func testB12CorruptedCarrierFormatRejectedH(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"unexpected"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-1", "tok-1")

	// signature is an invalid type (object instead of string)
	corruptedBody := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": [
				{"type": "thinking", "thinking": "broken", "signature": {"corrupted": true}},
				{"type": "text", "text": "hello"}
			]}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(corruptedBody))

	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(corruptedBody), &u)

	if status != http.StatusBadRequest || rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for corrupted carrier format, got status=%d rec.Code=%d msg=%q", status, rec.Code, msg)
	}
	if calls := atomic.LoadInt64(&tr.callCount); calls != 0 {
		t.Fatalf("expected 0 upstream calls for corrupted carrier, got %d", calls)
	}
}
