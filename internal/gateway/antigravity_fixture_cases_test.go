package gateway

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func cloneCanonicalList(t *testing.T, in []canonicalPart) []canonicalPart {
	t.Helper()
	out := make([]canonicalPart, len(in))
	copy(out, in)
	return out
}

func cloneE2EEvidence(t *testing.T, in *offlineE2EEvidence) *offlineE2EEvidence {
	t.Helper()
	if in == nil {
		return nil
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("cloneE2EEvidence marshal error: %v", err)
	}
	var out offlineE2EEvidence
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("cloneE2EEvidence unmarshal error: %v", err)
	}
	return &out
}

func validateNonEmptyScenarios[T any](list []T, name string) error {
	if len(list) == 0 {
		return fmt.Errorf("scenarios list %q is empty", name)
	}
	return nil
}

func requireNonEmptyScenarios[T any](t *testing.T, list []T, name string) {
	t.Helper()
	if err := validateNonEmptyScenarios(list, name); err != nil {
		t.Fatal(err)
	}
}

func testB01FixtureValidation(t *testing.T) {
	// 1. Root scenario list validation
	positives := []struct {
		fix antigravityFixture
		can []canonicalPart
		e2e *offlineE2EEvidence
	}{
		{
			fix: fixtureF01GeminiTailSignature(),
			can: fixtureF01GeminiTailSignature().ExpectedCanonical,
		},
		{
			fix: fixtureF02GeminiTools(),
			can: fixtureF02GeminiTools().ExpectedCanonical,
		},
		{
			fix: fixtureF02ClaudeThoughtTextSigAB(),
			can: fixtureF02ClaudeThoughtTextSigAB().ExpectedCanonical,
		},
		{
			fix: fixtureF06GeminiOfflineE2E(false),
			can: fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: fixtureF06GeminiOfflineE2E(false).OfflineE2E,
		},
		{
			// Positive case: empty output is completely valid when declared as expected
			fix: fixtureF06GeminiOfflineE2E(true),
			can: fixtureF06GeminiOfflineE2E(true).ExpectedCanonical,
			e2e: fixtureF06GeminiOfflineE2E(true).OfflineE2E,
		},
	}

	requireNonEmptyScenarios(t, positives, "root_positives")

	for _, p := range positives {
		t.Run("Positive_"+p.fix.Name, func(t *testing.T) {
			if err := compareAntigravityFixture(p.fix, p.can, p.e2e); err != nil {
				t.Fatalf("expected valid fixture %q to pass, got: %v", p.fix.Name, err)
			}
		})
	}

	// 2. Corrupt variants: strictly single-variable mutated from deep-copied baselines
	negatives := []struct {
		name         string
		fix          antigravityFixture
		can          []canonicalPart
		e2e          *offlineE2EEvidence
		errSubstring string
	}{
		{
			name: "Negative_MissingTerminalState",
			fix: func() antigravityFixture {
				f := fixtureF01GeminiTailSignature()
				f.FinishReason = "UNKNOWN"
				return f
			}(),
			can:          fixtureF01GeminiTailSignature().ExpectedCanonical,
			errSubstring: `finish reason must be STOP, got "UNKNOWN"`,
		},
		{
			name: "Negative_EmptyRawParts",
			fix: func() antigravityFixture {
				f := fixtureF01GeminiTailSignature()
				f.RawParts = nil
				return f
			}(),
			can:          fixtureF01GeminiTailSignature().ExpectedCanonical,
			errSubstring: "raw parts cannot be empty",
		},
		{
			name: "Negative_EmptyExpectedCanonicalWithNonEmptyRaw",
			fix: func() antigravityFixture {
				f := fixtureF01GeminiTailSignature()
				f.ExpectedCanonical = nil
				return f
			}(),
			can:          nil,
			errSubstring: "expected canonical parts cannot be empty",
		},
		{
			name: "Negative_SignatureTransferredToToolB",
			fix:  fixtureF02GeminiTools(),
			can: func() []canonicalPart {
				c := cloneCanonicalList(t, fixtureF02GeminiTools().ExpectedCanonical)
				c[0].ThoughtSignature = ""
				c[1].ThoughtSignature = "opaque_sig_gemini_call_a"
				return c
			}(),
			errSubstring: "signature mismatch",
		},
		{
			name: "Negative_SignatureDuplicatedToToolB",
			fix:  fixtureF02GeminiTools(),
			can: func() []canonicalPart {
				c := cloneCanonicalList(t, fixtureF02GeminiTools().ExpectedCanonical)
				c[1].ThoughtSignature = "opaque_sig_gemini_call_a"
				return c
			}(),
			errSubstring: "signature mismatch",
		},
		{
			name: "Negative_SignatureLost",
			fix:  fixtureF02ClaudeThoughtTextSigAB(),
			can: func() []canonicalPart {
				c := cloneCanonicalList(t, fixtureF02ClaudeThoughtTextSigAB().ExpectedCanonical)
				c[1].ThoughtSignature = ""
				return c
			}(),
			errSubstring: "signature mismatch",
		},
		{
			name: "Negative_ToolIDReplaced",
			fix:  fixtureF02ClaudeThoughtTextSigAB(),
			can: func() []canonicalPart {
				c := cloneCanonicalList(t, fixtureF02ClaudeThoughtTextSigAB().ExpectedCanonical)
				c[2].CallID = "replaced_call_id_x"
				return c
			}(),
			errSubstring: "call ID mismatch",
		},
		{
			name: "Negative_ToolArgsMutated",
			fix:  fixtureF02GeminiTools(),
			can: func() []canonicalPart {
				c := cloneCanonicalList(t, fixtureF02GeminiTools().ExpectedCanonical)
				c[0].Args = `{"AbsolutePath":"/mutated/path","toolAction":"Viewing file","toolSummary":"View main"}`
				return c
			}(),
			errSubstring: "args mismatch",
		},
		{
			name: "Negative_ToolNameMutated",
			fix:  fixtureF02GeminiTools(),
			can: func() []canonicalPart {
				c := cloneCanonicalList(t, fixtureF02GeminiTools().ExpectedCanonical)
				c[0].Name = "bash"
				return c
			}(),
			errSubstring: "name mismatch",
		},
		// Targeted single-variable negatives covering offline E2E evidence verification
		{
			name:         "Negative_MissingOfflineE2EEvidence",
			fix:          fixtureF06GeminiOfflineE2E(false),
			can:          fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e:          nil,
			errSubstring: "offline E2E evidence is missing",
		},
		{
			name: "Negative_MissingClientExecution",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.ClientExecution = nil
				return e
			}(),
			errSubstring: "client execution count mismatch",
		},
		{
			name: "Negative_MissingNextTurnContents",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.NextRequestTurn = nil
				return e
			}(),
			errSubstring: "next turn contents count mismatch",
		},
		{
			name: "Negative_ClientExecutionOutputDifferentString",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.ClientExecution[0].Output = "different non-empty string"
				return e
			}(),
			errSubstring: "output mismatch",
		},
		{
			name: "Negative_FunctionResponseIDMismatch",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.NextRequestTurn[0].Parts[0].FunctionResponse.ID = "call_different_resp_id"
				return e
			}(),
			errSubstring: "functionResponse ID/Name mismatch",
		},
		{
			name: "Negative_UpstreamOutputWrongValue",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.NextRequestTurn[0].Parts[0].FunctionResponse.Response = json.RawMessage(`{"output":"wrong value"}`)
				return e
			}(),
			errSubstring: "functionResponse output mismatch",
		},
		{
			name: "Negative_OutputMistakenlyWrittenAsResult",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.NextRequestTurn[0].Parts[0].FunctionResponse.Response = json.RawMessage(`{"result":"some content"}`)
				return e
			}(),
			errSubstring: "wrongly used 'result' instead of 'output'",
		},
		{
			name: "Negative_ClientArgsChanged",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.ClientExecution[0].ClientArgs = json.RawMessage(`{"AbsolutePath":"/another/path"}`)
				return e
			}(),
			errSubstring: "client args mismatch",
		},
		{
			name: "Negative_IsErrorFlipped",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.ClientExecution[0].IsError = true // flipped from false
				return e
			}(),
			errSubstring: "isError mismatch: got true, want false",
		},
		{
			name: "Negative_InitialCallsChanged",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.InitialCalls[0].ID = "call_different_id"
				return e
			}(),
			errSubstring: "initial call 0: mismatch",
		},
		{
			name: "Negative_NextTurnRoleMismatch",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.NextRequestTurn[0].Role = "user" // wrong role: Gemini requires model
				return e
			}(),
			errSubstring: `role mismatch: got "user", want "model"`,
		},
		{
			name: "Negative_FollowupFinishNotStop",
			fix:  fixtureF06GeminiOfflineE2E(false),
			can:  fixtureF06GeminiOfflineE2E(false).ExpectedCanonical,
			e2e: func() *offlineE2EEvidence {
				e := cloneE2EEvidence(t, fixtureF06GeminiOfflineE2E(false).OfflineE2E)
				e.FollowupFinish = "LENGTH"
				return e
			}(),
			errSubstring: `followup finish reason must be STOP, got "LENGTH"`,
		},
	}

	for _, n := range negatives {
		t.Run(n.name, func(t *testing.T) {
			err := compareAntigravityFixture(n.fix, n.can, n.e2e)
			if err == nil {
				t.Fatalf("expected error for %q, but got nil", n.name)
			}
			if !strings.Contains(err.Error(), n.errSubstring) {
				t.Fatalf("expected error for %q to contain %q, got: %v", n.name, n.errSubstring, err)
			}
		})
	}

	// 3. Test empty/zero scenario handling
	t.Run("Negative_EmptyFixtureName", func(t *testing.T) {
		err := compareAntigravityFixture(antigravityFixture{}, nil, nil)
		if err == nil || !strings.Contains(err.Error(), "missing name") {
			t.Fatalf("expected missing name error, got: %v", err)
		}
	})

	// 4. Test empty root scenario list failure using the shared validation helper
	t.Run("Negative_ZeroScenariosList", func(t *testing.T) {
		var emptyList []struct {
			fix antigravityFixture
			can []canonicalPart
			e2e *offlineE2EEvidence
		}
		err := validateNonEmptyScenarios(emptyList, "empty_scenarios")
		if err == nil || !strings.Contains(err.Error(), `scenarios list "empty_scenarios" is empty`) {
			t.Fatalf("expected empty scenario list error, got: %v", err)
		}
	})
}

func testB01JSONEqualExact(t *testing.T) {
	t.Run("Negative_RejectTrailingSecondValueA", func(t *testing.T) {
		a := []byte(`{"a":1} {"b":2}`)
		b := []byte(`{"a":1}`)
		equal, err := jsonEqualExact(a, b)
		if err == nil && equal {
			t.Fatalf("expected error or mismatch for trailing second value in a, got equal=true")
		}
	})

	t.Run("Negative_RejectTrailingSecondValueB", func(t *testing.T) {
		a := []byte(`{"a":1}`)
		b := []byte(`{"a":1} {"b":2}`)
		equal, err := jsonEqualExact(a, b)
		if err == nil && equal {
			t.Fatalf("expected error or mismatch for trailing second value in b, got equal=true")
		}
	})

	t.Run("Negative_RejectTrailingGarbageA", func(t *testing.T) {
		a := []byte(`{"a":1} trailing garbage`)
		b := []byte(`{"a":1}`)
		equal, err := jsonEqualExact(a, b)
		if err == nil && equal {
			t.Fatalf("expected error for trailing garbage in a, got equal=true")
		}
	})

	t.Run("Negative_RejectTrailingGarbageB", func(t *testing.T) {
		a := []byte(`{"a":1}`)
		b := []byte(`{"a":1} trailing garbage`)
		equal, err := jsonEqualExact(a, b)
		if err == nil && equal {
			t.Fatalf("expected error for trailing garbage in b, got equal=true")
		}
	})

	t.Run("Negative_RejectInvalidJSONA", func(t *testing.T) {
		a := []byte(`{"a":`)
		b := []byte(`{"a":1}`)
		equal, err := jsonEqualExact(a, b)
		if err == nil && equal {
			t.Fatalf("expected error for invalid JSON in a, got equal=true")
		}
	})

	t.Run("Negative_RejectInvalidJSONB", func(t *testing.T) {
		a := []byte(`{"a":1}`)
		b := []byte(`{"a":`)
		equal, err := jsonEqualExact(a, b)
		if err == nil && equal {
			t.Fatalf("expected error for invalid JSON in b, got equal=true")
		}
	})

	t.Run("Positive_AcceptFieldOrderAndWhitespaceEquivalent", func(t *testing.T) {
		a := []byte(`{"b": 2, "a": 1}`)
		b := []byte(`{ "a" : 1 , "b" : 2 }`)
		equal, err := jsonEqualExact(a, b)
		if err != nil {
			t.Fatalf("expected equivalent JSON to parse without error, got: %v", err)
		}
		if !equal {
			t.Fatalf("expected equivalent JSON to match, got false")
		}
	})

	t.Run("Negative_RejectIntegersAbove2To53WhenDifferent", func(t *testing.T) {
		// 9007199254740992 is 2^53, 9007199254740993 is 2^53 + 1
		// In float64 they lose precision and become identical, but json.Number preserves exact string digits.
		a := []byte(`{"num": 9007199254740992}`)
		b := []byte(`{"num": 9007199254740993}`)
		equal, err := jsonEqualExact(a, b)
		if err != nil {
			t.Fatalf("unexpected error parsing integers > 2^53: %v", err)
		}
		if equal {
			t.Fatalf("expected integers > 2^53 with different least-significant digit to not match, but got equal=true")
		}
	})
}
