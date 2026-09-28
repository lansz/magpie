package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// rawPart models an opaque native response part from Antigravity.
type rawPart struct {
	Text             string       `json:"text,omitempty"`
	Thought          bool         `json:"thought,omitempty"`
	ThoughtSignature string       `json:"thoughtSignature,omitempty"`
	FunctionCall     *rawFuncCall `json:"functionCall,omitempty"`
	FunctionResponse *rawFuncResp `json:"functionResponse,omitempty"`
}

type rawFuncCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name,omitempty"`
	Args json.RawMessage `json:"args,omitempty"`
}

type rawFuncResp struct {
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name,omitempty"`
	Response json.RawMessage `json:"response,omitempty"`
}

type rawContent struct {
	Role  string    `json:"role"`
	Parts []rawPart `json:"parts"`
}

// canonicalPart models the expected normalized Antigravity part representation.
type canonicalPart struct {
	Kind             Kind   `json:"kind"` // Thinking, Text, ToolCall, ToolResult
	Text             string `json:"text,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	CallID           string `json:"callId,omitempty"`
	Name             string `json:"name,omitempty"`
	Args             string `json:"args,omitempty"`
}

type toolExecutionRecord struct {
	CallID     string          `json:"callId"`
	ToolName   string          `json:"toolName"`
	ClientArgs json.RawMessage `json:"clientArgs"`
	Output     string          `json:"output"`
	IsError    bool            `json:"isError"`
}

// offlineE2EEvidence holds real-structure offline evidence for tool roundtrip completion.
type offlineE2EEvidence struct {
	InitialCalls    []rawFuncCall         `json:"initialCalls"`
	ClientExecution []toolExecutionRecord `json:"clientExecution"`
	NextRequestTurn []rawContent          `json:"nextRequestTurn"`
	FollowupFinish  string                `json:"followupFinish"`
}

type antigravityFixture struct {
	Name              string
	Model             string
	RawParts          []rawPart
	FinishReason      string
	ExpectedCanonical []canonicalPart
	OfflineE2E        *offlineE2EEvidence
}

// jsonEqualExact compares two JSON byte slices for exact structural/value equality without losing number precision.
// It strictly validates JSON syntax and rejects any trailing data or secondary values.
func jsonEqualExact(a, b []byte) (bool, error) {
	if !json.Valid(a) {
		return false, fmt.Errorf("invalid json in first value")
	}
	if !json.Valid(b) {
		return false, fmt.Errorf("invalid json in second value")
	}

	var valA, valB any

	decA := json.NewDecoder(bytes.NewReader(a))
	decA.UseNumber()
	if err := decA.Decode(&valA); err != nil {
		return false, fmt.Errorf("invalid json in first value: %w", err)
	}
	var extraA any
	if err := decA.Decode(&extraA); err != io.EOF {
		return false, fmt.Errorf("first value contains trailing data or multiple JSON values")
	}

	decB := json.NewDecoder(bytes.NewReader(b))
	decB.UseNumber()
	if err := decB.Decode(&valB); err != nil {
		return false, fmt.Errorf("invalid json in second value: %w", err)
	}
	var extraB any
	if err := decB.Decode(&extraB); err != io.EOF {
		return false, fmt.Errorf("second value contains trailing data or multiple JSON values")
	}

	// Re-marshal both with sorted keys and normalized spacing to compare
	normA, err := json.Marshal(valA)
	if err != nil {
		return false, err
	}
	normB, err := json.Marshal(valB)
	if err != nil {
		return false, err
	}

	return bytes.Equal(normA, normB), nil
}

// compareAntigravityFixture compares actual test evidence against independent fixed expectations.
func compareAntigravityFixture(fix antigravityFixture, actualCanonical []canonicalPart, actualE2E *offlineE2EEvidence) error {
	if fix.Name == "" {
		return fmt.Errorf("fixture: missing name")
	}
	if len(fix.RawParts) == 0 {
		return fmt.Errorf("fixture %q: raw parts cannot be empty", fix.Name)
	}
	if len(fix.ExpectedCanonical) == 0 {
		return fmt.Errorf("fixture %q: expected canonical parts cannot be empty", fix.Name)
	}
	if fix.FinishReason != "STOP" {
		return fmt.Errorf("fixture %q: finish reason must be STOP, got %q", fix.Name, fix.FinishReason)
	}

	// 1. Verify canonical parts against expected canonical
	if len(actualCanonical) != len(fix.ExpectedCanonical) {
		return fmt.Errorf("fixture %q: canonical length mismatch: got %d, want %d",
			fix.Name, len(actualCanonical), len(fix.ExpectedCanonical))
	}

	for i := range fix.ExpectedCanonical {
		exp := fix.ExpectedCanonical[i]
		act := actualCanonical[i]

		if act.Kind != exp.Kind {
			return fmt.Errorf("fixture %q part %d: kind mismatch: got %v, want %v", fix.Name, i, act.Kind, exp.Kind)
		}
		if act.Text != exp.Text {
			return fmt.Errorf("fixture %q part %d: text mismatch: got %q, want %q", fix.Name, i, act.Text, exp.Text)
		}
		if act.ThoughtSignature != exp.ThoughtSignature {
			return fmt.Errorf("fixture %q part %d: signature mismatch: got %q, want %q",
				fix.Name, i, act.ThoughtSignature, exp.ThoughtSignature)
		}
		if act.CallID != exp.CallID {
			return fmt.Errorf("fixture %q part %d: call ID mismatch: got %q, want %q", fix.Name, i, act.CallID, exp.CallID)
		}
		if act.Name != exp.Name {
			return fmt.Errorf("fixture %q part %d: name mismatch: got %q, want %q", fix.Name, i, act.Name, exp.Name)
		}
		if act.Args != "" || exp.Args != "" {
			match, err := jsonEqualExact([]byte(act.Args), []byte(exp.Args))
			if err != nil {
				return fmt.Errorf("fixture %q part %d: args invalid json: %w", fix.Name, i, err)
			}
			if !match {
				return fmt.Errorf("fixture %q part %d: args mismatch: got %q, want %q", fix.Name, i, act.Args, exp.Args)
			}
		}
	}

	// 2. Check offline E2E evidence completeness if fixture defines it
	if fix.OfflineE2E != nil {
		if actualE2E == nil {
			return fmt.Errorf("fixture %q: offline E2E evidence is missing", fix.Name)
		}
		if actualE2E.FollowupFinish != "STOP" {
			return fmt.Errorf("fixture %q: followup finish reason must be STOP, got %q", fix.Name, actualE2E.FollowupFinish)
		}

		// Initial calls comparison
		if len(actualE2E.InitialCalls) != len(fix.OfflineE2E.InitialCalls) {
			return fmt.Errorf("fixture %q: initial calls count mismatch: got %d, want %d",
				fix.Name, len(actualE2E.InitialCalls), len(fix.OfflineE2E.InitialCalls))
		}
		for i, expCall := range fix.OfflineE2E.InitialCalls {
			actCall := actualE2E.InitialCalls[i]
			if actCall.ID != expCall.ID || actCall.Name != expCall.Name {
				return fmt.Errorf("fixture %q initial call %d: mismatch: got (%q, %q), want (%q, %q)",
					fix.Name, i, actCall.ID, actCall.Name, expCall.ID, expCall.Name)
			}
			match, err := jsonEqualExact(actCall.Args, expCall.Args)
			if err != nil || !match {
				return fmt.Errorf("fixture %q initial call %d: args mismatch: got %s, want %s",
					fix.Name, i, string(actCall.Args), string(expCall.Args))
			}
		}

		// Client execution records comparison
		if len(actualE2E.ClientExecution) != len(fix.OfflineE2E.ClientExecution) {
			return fmt.Errorf("fixture %q: client execution count mismatch: got %d, want %d",
				fix.Name, len(actualE2E.ClientExecution), len(fix.OfflineE2E.ClientExecution))
		}
		for i, expExec := range fix.OfflineE2E.ClientExecution {
			actExec := actualE2E.ClientExecution[i]
			if actExec.CallID != expExec.CallID {
				return fmt.Errorf("fixture %q execution %d: call ID mismatch: got %q, want %q",
					fix.Name, i, actExec.CallID, expExec.CallID)
			}
			if actExec.ToolName != expExec.ToolName {
				return fmt.Errorf("fixture %q execution %d: tool name mismatch: got %q, want %q",
					fix.Name, i, actExec.ToolName, expExec.ToolName)
			}
			if actExec.Output != expExec.Output {
				return fmt.Errorf("fixture %q execution %d: output mismatch: got %q, want %q",
					fix.Name, i, actExec.Output, expExec.Output)
			}
			if actExec.IsError != expExec.IsError {
				return fmt.Errorf("fixture %q execution %d: isError mismatch: got %v, want %v",
					fix.Name, i, actExec.IsError, expExec.IsError)
			}
			match, err := jsonEqualExact(actExec.ClientArgs, expExec.ClientArgs)
			if err != nil || !match {
				return fmt.Errorf("fixture %q execution %d: client args mismatch: got %s, want %s",
					fix.Name, i, string(actExec.ClientArgs), string(expExec.ClientArgs))
			}
		}

		// Next request turn comparison with role/parts
		if len(actualE2E.NextRequestTurn) != len(fix.OfflineE2E.NextRequestTurn) {
			return fmt.Errorf("fixture %q: next turn contents count mismatch: got %d, want %d",
				fix.Name, len(actualE2E.NextRequestTurn), len(fix.OfflineE2E.NextRequestTurn))
		}
		for cIdx, expContent := range fix.OfflineE2E.NextRequestTurn {
			actContent := actualE2E.NextRequestTurn[cIdx]
			if actContent.Role != expContent.Role {
				return fmt.Errorf("fixture %q next turn content %d: role mismatch: got %q, want %q",
					fix.Name, cIdx, actContent.Role, expContent.Role)
			}
			if len(actContent.Parts) != len(expContent.Parts) {
				return fmt.Errorf("fixture %q next turn content %d: parts count mismatch: got %d, want %d",
					fix.Name, cIdx, len(actContent.Parts), len(expContent.Parts))
			}
			for pIdx, expPart := range expContent.Parts {
				actPart := actContent.Parts[pIdx]
				if expPart.FunctionResponse != nil {
					if actPart.FunctionResponse == nil {
						return fmt.Errorf("fixture %q content %d part %d: missing functionResponse", fix.Name, cIdx, pIdx)
					}
					expResp := expPart.FunctionResponse
					actResp := actPart.FunctionResponse
					if actResp.ID != expResp.ID || actResp.Name != expResp.Name {
						return fmt.Errorf("fixture %q content %d part %d: functionResponse ID/Name mismatch: got (%q, %q), want (%q, %q)",
							fix.Name, cIdx, pIdx, actResp.ID, actResp.Name, expResp.ID, expResp.Name)
					}

					// Verify exact response structure and output field
					var expObj, actObj struct {
						Output *string `json:"output"`
						Result *string `json:"result"`
					}
					if err := json.Unmarshal(expResp.Response, &expObj); err != nil {
						return fmt.Errorf("fixture %q content %d part %d: invalid expected response JSON: %w", fix.Name, cIdx, pIdx, err)
					}
					if err := json.Unmarshal(actResp.Response, &actObj); err != nil {
						return fmt.Errorf("fixture %q content %d part %d: invalid actual response JSON: %w", fix.Name, cIdx, pIdx, err)
					}

					if actObj.Result != nil {
						return fmt.Errorf("fixture %q content %d part %d: functionResponse response wrongly used 'result' instead of 'output'", fix.Name, cIdx, pIdx)
					}
					if actObj.Output == nil {
						return fmt.Errorf("fixture %q content %d part %d: functionResponse response missing 'output' field", fix.Name, cIdx, pIdx)
					}
					if expObj.Output == nil || *actObj.Output != *expObj.Output {
						expVal := "<nil>"
						if expObj.Output != nil {
							expVal = *expObj.Output
						}
						return fmt.Errorf("fixture %q content %d part %d: functionResponse output mismatch: got %q, want %q",
							fix.Name, cIdx, pIdx, *actObj.Output, expVal)
					}
				}
			}
		}
	}

	return nil
}

func fixtureF01GeminiTailSignature() antigravityFixture {
	return antigravityFixture{
		Name:         "F01_GeminiTrailingSignature",
		Model:        "gemini-3.8-flash-high",
		FinishReason: "STOP",
		RawParts: []rawPart{
			{Text: "Hello from Gemini."},
			{Text: "", ThoughtSignature: "opaque_sig_gemini_tail"},
		},
		ExpectedCanonical: []canonicalPart{
			{Kind: Text, Text: "Hello from Gemini.", ThoughtSignature: "opaque_sig_gemini_tail"},
		},
	}
}

func fixtureF02GeminiTools() antigravityFixture {
	argsA := `{"AbsolutePath":"/workspace/main.go","toolAction":"Viewing file","toolSummary":"View main"}`
	argsB := `{"CommandLine":"ls -la","Cwd":"/workspace","WaitMsBeforeAsync":1000,"toolAction":"Listing files","toolSummary":"List dir"}`
	return antigravityFixture{
		Name:         "F02_GeminiTools_A_with_sig_B_without",
		Model:        "gemini-3.8-flash-high",
		FinishReason: "STOP",
		RawParts: []rawPart{
			{
				FunctionCall:     &rawFuncCall{ID: "call_view_01", Name: "view_file", Args: json.RawMessage(argsA)},
				ThoughtSignature: "opaque_sig_gemini_call_a",
			},
			{
				FunctionCall: &rawFuncCall{ID: "call_cmd_02", Name: "run_command", Args: json.RawMessage(argsB)},
			},
		},
		ExpectedCanonical: []canonicalPart{
			{Kind: ToolCall, CallID: "call_view_01", Name: "view_file", Args: argsA, ThoughtSignature: "opaque_sig_gemini_call_a"},
			{Kind: ToolCall, CallID: "call_cmd_02", Name: "run_command", Args: argsB},
		},
	}
}

func fixtureF02ClaudeThoughtTextSigAB() antigravityFixture {
	argsA := `{"AbsolutePath":"/workspace/a.txt","toolAction":"Viewing file","toolSummary":"View a"}`
	argsB := `{"AbsolutePath":"/workspace/b.txt","toolAction":"Viewing file","toolSummary":"View b"}`
	return antigravityFixture{
		Name:         "F02_Claude_Thought_TextSig_A_B",
		Model:        "claude-sonnet-4-6",
		FinishReason: "STOP",
		RawParts: []rawPart{
			{Text: "Checking both files...", Thought: true},
			{Text: "", Thought: true, ThoughtSignature: "opaque_sig_claude_thought"},
			{Text: "I will view file a and then b."},
			{FunctionCall: &rawFuncCall{ID: "toolu_01_a", Name: "view_file", Args: json.RawMessage(argsA)}},
			{FunctionCall: &rawFuncCall{ID: "toolu_02_b", Name: "view_file", Args: json.RawMessage(argsB)}},
		},
		ExpectedCanonical: []canonicalPart{
			{Kind: Thinking, Text: "Checking both files..."},
			{Kind: Text, Text: "I will view file a and then b.", ThoughtSignature: "opaque_sig_claude_thought"},
			{Kind: ToolCall, CallID: "toolu_01_a", Name: "view_file", Args: argsA},
			{Kind: ToolCall, CallID: "toolu_02_b", Name: "view_file", Args: argsB},
		},
	}
}

// fixtureF06GeminiOfflineE2E builds offline E2E evidence for a single Gemini call with signature.
// Native protocol requirement: In Gemini, tool responses are submitted with role=model.
func fixtureF06GeminiOfflineE2E(emptyOutput bool) antigravityFixture {
	args := `{"AbsolutePath":"/workspace/empty.txt","toolAction":"Viewing file","toolSummary":"View empty"}`
	rawOutput := ""
	if !emptyOutput {
		rawOutput = "content: line1\nline2\n"
	}

	funcCall := rawFuncCall{ID: "call_view_99", Name: "view_file", Args: json.RawMessage(args)}
	respJSON := fmt.Sprintf(`{"output":%q}`, rawOutput)

	return antigravityFixture{
		Name:         "F06_Gemini_OfflineE2E_Evidence",
		Model:        "gemini-3.8-flash-high",
		FinishReason: "STOP",
		RawParts: []rawPart{
			{FunctionCall: &funcCall, ThoughtSignature: "opaque_sig_gemini_view"},
		},
		ExpectedCanonical: []canonicalPart{
			{Kind: ToolCall, CallID: "call_view_99", Name: "view_file", Args: args, ThoughtSignature: "opaque_sig_gemini_view"},
		},
		OfflineE2E: &offlineE2EEvidence{
			InitialCalls: []rawFuncCall{funcCall},
			ClientExecution: []toolExecutionRecord{
				{
					CallID:     "call_view_99",
					ToolName:   "view_file", // synthetic passthrough (same name)
					ClientArgs: json.RawMessage(args),
					Output:     rawOutput,
					IsError:    false,
				},
			},
			NextRequestTurn: []rawContent{
				{
					Role: "model",
					Parts: []rawPart{
						{
							FunctionResponse: &rawFuncResp{
								ID:       "call_view_99",
								Name:     "view_file",
								Response: json.RawMessage(respJSON),
							},
						},
					},
				},
			},
			FollowupFinish: "STOP",
		},
	}
}
