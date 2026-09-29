package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

// testB11Signature covers B11: Real thought signatures attach and merge according
// to model-specific rules: Gemini tail signature attaches to preceding merged text;
// Claude detached signature attaches to the next semantic part (first text chunk)
// and merges subsequent text chunks without duplicating signatures; signatures remain
// opaque and equal to original bytes; and skip_thought_signature_validator is eliminated.
func testB11Signature(t *testing.T) {
	t.Run("GeminiTailSignatureAttachesToMergedText", testB11GeminiTailSignatureAttachesToMergedText)
	t.Run("ClaudeDetachedSignatureAttachesToNextSemanticPart", testB11ClaudeDetachedSignatureAttachesToNextSemanticPart)
	t.Run("NeverUseSkipThoughtSignatureValidator", testB11NeverUseSkipThoughtSignatureValidator)
	t.Run("RestoreRealSignatureFromBinding", testB11RestoreRealSignatureFromBinding)
}

// 1. Gemini: A trailing empty-text signature belongs to the preceding merged text part.
func testB11GeminiTailSignatureAttachesToMergedText(t *testing.T) {
	fix := fixtureF01GeminiTailSignature()

	canonical, err := canonicalizeAntigravityParts("gemini-3.8-flash-high", fix.RawParts)
	if err != nil {
		t.Fatalf("canonicalize: %v", err)
	}
	if err := compareAntigravityFixture(fix, canonical, nil); err != nil {
		t.Fatalf("Gemini tail signature canonicalization failed: %v", err)
	}

	if len(canonical) != 1 {
		t.Fatalf("expected 1 merged text part, got %d: %v", len(canonical), canonical)
	}
	if canonical[0].Kind != Text || canonical[0].Text != "Hello from Gemini." {
		t.Errorf("text mismatch: got %q", canonical[0].Text)
	}
	if canonical[0].ThoughtSignature != "opaque_sig_gemini_tail" {
		t.Errorf("signature mismatch: got %q, want 'opaque_sig_gemini_tail'", canonical[0].ThoughtSignature)
	}
}

// 2. Claude: Detached signature attaches to the next semantic part (first text chunk),
// and subsequent text chunks are merged without replicating the signature.
func testB11ClaudeDetachedSignatureAttachesToNextSemanticPart(t *testing.T) {
	fix := fixtureF02ClaudeThoughtTextSigAB()

	canonical, err := canonicalizeAntigravityParts("claude-sonnet-4-6", fix.RawParts)
	if err != nil {
		t.Fatalf("canonicalize: %v", err)
	}
	if err := compareAntigravityFixture(fix, canonical, nil); err != nil {
		t.Fatalf("Claude detached signature canonicalization failed: %v", err)
	}

	if len(canonical) != 4 {
		t.Fatalf("expected 4 canonical parts, got %d: %v", len(canonical), canonical)
	}

	// Part 0: Thinking (must NOT have the detached signature)
	if canonical[0].Kind != Thinking || canonical[0].ThoughtSignature != "" {
		t.Errorf("Part 0 (Thinking) must not hold signature: %v", canonical[0])
	}

	// Part 1: Text (holds the detached signature, merged text)
	if canonical[1].Kind != Text || canonical[1].ThoughtSignature != "opaque_sig_claude_thought" {
		t.Errorf("Part 1 (Text) must hold the detached signature: %v", canonical[1])
	}
	if canonical[1].Text != "I will view file a and then b." {
		t.Errorf("Part 1 text mismatch: got %q", canonical[1].Text)
	}

	// Parts 2 & 3: ToolCalls (no duplicate of thinking signature)
	if canonical[2].Kind != ToolCall || canonical[2].ThoughtSignature != "" {
		t.Errorf("Part 2 (ToolCall A) must not replicate thinking signature: %v", canonical[2])
	}
	if canonical[3].Kind != ToolCall || canonical[3].ThoughtSignature != "" {
		t.Errorf("Part 3 (ToolCall B) must not replicate thinking signature: %v", canonical[3])
	}
}

// 3. Eliminate skip_thought_signature_validator pseudo-signature entirely from buildCodeAssist.
func testB11NeverUseSkipThoughtSignatureValidator(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
		]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	wireStr := string(wireBytes)

	if strings.Contains(wireStr, "skip_thought_signature_validator") {
		t.Errorf("wire request illegally contains 'skip_thought_signature_validator':\n%s", wireStr)
	}
}

// 4. Restore real opaque signature from binding record onto wire request functionCall.
func testB11RestoreRealSignatureFromBinding(t *testing.T) {
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	const (
		nativeCallID = "call_signed_007"
		clientCallID = "toolu_signed_007"
		realSig      = "opaque_real_signature_gemini_xyz_12345"
	)

	binding := AntigravityToolBinding{
		NativeID:        nativeCallID,
		NativeName:      "view_file",
		NativeArgs:      json.RawMessage(`{"path":"/secure.go"}`),
		NativeSignature: realSig,
		ClientID:        clientCallID,
		ClientName:      "Read",
	}

	if err := defaultToolBindingStore.Bind(binding); err != nil {
		t.Fatalf("Bind: %v", err)
	}

	body := `{
		"model": "gemini-3.8-flash-high",
		"messages": [
			{"role": "user", "content": "read secure"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "` + clientCallID + `", "name": "Read", "input": {"file_path": "/secure.go"}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "` + clientCallID + `", "content": "secret data"}]}
		]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	wireBytes := buildCodeAssist(req, "gemini-3.8-flash-high", "antigravity")
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
	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal: %v\nwire: %s", err, wireBytes)
	}

	var foundCallID, foundSig string
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				foundCallID = p.FunctionCall.ID
				foundSig = p.ThoughtSignature
			}
		}
	}

	if foundCallID != nativeCallID {
		t.Errorf("functionCall ID = %q, want %q", foundCallID, nativeCallID)
	}
	if foundSig != realSig {
		t.Errorf("functionCall thoughtSignature = %q, want real signature %q", foundSig, realSig)
	}
	if strings.Contains(foundSig, "skip") {
		t.Errorf("signature must not contain 'skip': %q", foundSig)
	}
}
