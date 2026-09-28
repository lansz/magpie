package gateway

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03bToolArgsValidation is the unified entrypoint for all B03b BDD scenarios.
func testB03bToolArgsValidation(t *testing.T) {
	t.Run("Unit", testB03bToolArgsValidationU)
	t.Run("HLayer_Valid", testB03bToolArgsValidationHValid)
	t.Run("HLayer_Invalid", testB03bToolArgsValidationHInvalid)
	t.Run("HLayer_NonTarget", testB03bToolArgsValidationHNonTarget)
}

// testB03bToolArgsValidationU covers all U-layer B03b tool arguments fidelity scenarios.
func testB03bToolArgsValidationU(t *testing.T) {
	t.Run("ValidArgsFidelity_GeminiAndClaude", testB03bUValidArgsFidelity)
	t.Run("BusinessKeysNotDamaged", testB03bUBusinessKeysNotDamaged)
	t.Run("WhitespaceAroundObjectPreserved", testB03bUWhitespaceAroundObjectPreserved)
}

type capturedFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

// extractAllFunctionCalls parses built CodeAssist JSON and returns all function calls in order.
func extractAllFunctionCalls(t *testing.T, built []byte) []capturedFunctionCall {
	t.Helper()
	var env struct {
		Request struct {
			Contents []struct {
				Parts []struct {
					FunctionCall *capturedFunctionCall `json:"functionCall"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	if err := json.Unmarshal(built, &env); err != nil {
		t.Fatalf("unmarshal built CodeAssist request failed: %v\nBody: %s", err, string(built))
	}
	var out []capturedFunctionCall
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				out = append(out, *p.FunctionCall)
			}
		}
	}
	return out
}

// extractFirstFunctionCallArgs returns the args of the first function call.
func extractFirstFunctionCallArgs(t *testing.T, built []byte) json.RawMessage {
	t.Helper()
	calls := extractAllFunctionCalls(t, built)
	if len(calls) == 0 {
		t.Fatalf("no functionCall with args found in built CodeAssist request: %s", string(built))
	}
	return calls[0].Args
}

// 1. Valid args fidelity: {}, nested array/obj, null values, booleans, number strings, and >2^53 integers.
func testB03bUValidArgsFidelity(t *testing.T) {
	models := []string{"gemini-3.8-flash-high", "claude-sonnet-4-6"}

	cases := []struct {
		name      string
		argsJSON  string
		wantExact string
	}{
		{
			name:      "ExplicitEmptyObject",
			argsJSON:  `{}`,
			wantExact: `{}`,
		},
		{
			name:      "NestedStructuresAndNulls",
			argsJSON:  `{"flag":true,"nested":{"items":[1,"two",null,false],"opt":null},"str_num":"12345"}`,
			wantExact: `{"flag":true,"nested":{"items":[1,"two",null,false],"opt":null},"str_num":"12345"}`,
		},
		{
			name:      "IntegerAbove2To53Preserved",
			argsJSON:  `{"id_above_53":9007199254740993}`,
			wantExact: `{"id_above_53":9007199254740993}`,
		},
	}

	for _, model := range models {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s_%s", model, tc.name), func(t *testing.T) {
				// (a) Anthropic Messages: tool_use.input
				t.Run("Messages", func(t *testing.T) {
					payload := fmt.Sprintf(`{
						"model": %q,
						"max_tokens": 1024,
						"messages": [
							{"role": "user", "content": "run tool"},
							{"role": "assistant", "content": [
								{"type": "tool_use", "id": "call_1", "name": "do_task", "input": %s}
							]}
						]
					}`, model, tc.argsJSON)

					req, err := parse(provider.Anthropic, []byte(payload))
					if err != nil {
						t.Fatalf("parse failed: %v", err)
					}
					built := buildCodeAssist(req, model, "antigravity")
					gotArgs := extractFirstFunctionCallArgs(t, built)
					eq, err := jsonEqualExact(gotArgs, []byte(tc.wantExact))
					if err != nil || !eq {
						t.Errorf("wire args mismatch:\ngot:  %s\nwant: %s (err: %v)", string(gotArgs), tc.wantExact, err)
					}
				})

				// (b) Chat: assistant.tool_calls[].function.arguments (escaped JSON string)
				t.Run("Chat", func(t *testing.T) {
					escapedArgs, err := json.Marshal(tc.argsJSON)
					if err != nil {
						t.Fatalf("marshal string failed: %v", err)
					}
					payload := fmt.Sprintf(`{
						"model": %q,
						"messages": [
							{"role": "user", "content": "run tool"},
							{"role": "assistant", "tool_calls": [
								{"id": "call_1", "type": "function", "function": {"name": "do_task", "arguments": %s}}
							]}
						]
					}`, model, string(escapedArgs))

					req, err := parse(provider.Chat, []byte(payload))
					if err != nil {
						t.Fatalf("parse failed: %v", err)
					}
					built := buildCodeAssist(req, model, "antigravity")
					gotArgs := extractFirstFunctionCallArgs(t, built)
					eq, err := jsonEqualExact(gotArgs, []byte(tc.wantExact))
					if err != nil || !eq {
						t.Errorf("wire args mismatch:\ngot:  %s\nwant: %s (err: %v)", string(gotArgs), tc.wantExact, err)
					}
				})

				// (c) Responses: input[type=function_call].arguments (escaped JSON string)
				t.Run("Responses", func(t *testing.T) {
					escapedArgs, err := json.Marshal(tc.argsJSON)
					if err != nil {
						t.Fatalf("marshal string failed: %v", err)
					}
					payload := fmt.Sprintf(`{
						"model": %q,
						"input": [
							{"role": "user", "content": "run tool"},
							{"type": "function_call", "call_id": "call_1", "name": "do_task", "arguments": %s}
						]
					}`, model, string(escapedArgs))

					req, err := parse(provider.Responses, []byte(payload))
					if err != nil {
						t.Fatalf("parse failed: %v", err)
					}
					built := buildCodeAssist(req, model, "antigravity")
					gotArgs := extractFirstFunctionCallArgs(t, built)
					eq, err := jsonEqualExact(gotArgs, []byte(tc.wantExact))
					if err != nil || !eq {
						t.Errorf("wire args mismatch:\ngot:  %s\nwant: %s (err: %v)", string(gotArgs), tc.wantExact, err)
					}
				})
			})
		}
	}
}

// 2. Sensitive business keys inside object (input, title, default, cache_control) must be preserved.
func testB03bUBusinessKeysNotDamaged(t *testing.T) {
	const businessArgs = `{"cache_control":{"type":"ephemeral"},"default":"std_val","input":"raw_input_data","title":"my_task"}`

	payload := fmt.Sprintf(`{
		"model": "gemini-3.8-flash-high",
		"max_tokens": 1024,
		"messages": [
			{"role": "user", "content": "run"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "call_b1", "name": "complex_task", "input": %s}
			]}
		]
	}`, businessArgs)

	req, err := parse(provider.Anthropic, []byte(payload))
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	built := buildCodeAssist(req, "gemini-3.8-flash-high", "antigravity")
	gotArgs := extractFirstFunctionCallArgs(t, built)

	eq, err := jsonEqualExact(gotArgs, []byte(businessArgs))
	if err != nil || !eq {
		t.Errorf("business keys damaged in args:\ngot:  %s\nwant: %s (err: %v)", string(gotArgs), businessArgs, err)
	}
}

// 3. Whitespace around JSON object must be trimmed safely and content preserved.
func testB03bUWhitespaceAroundObjectPreserved(t *testing.T) {
	const rawWithSpaces = "  \n\t {\"key\": \"value\"}  \t\r\n "
	escaped, err := json.Marshal(rawWithSpaces)
	if err != nil {
		t.Fatalf("marshal whitespace string failed: %v", err)
	}

	payload := fmt.Sprintf(`{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "run"},
			{"role": "assistant", "tool_calls": [
				{"id": "call_sp", "type": "function", "function": {"name": "do_space", "arguments": %s}}
			]}
		]
	}`, string(escaped))

	req, err := parse(provider.Chat, []byte(payload))
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	built := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	gotArgs := extractFirstFunctionCallArgs(t, built)

	eq, err := jsonEqualExact(gotArgs, []byte(`{"key":"value"}`))
	if err != nil || !eq {
		t.Errorf("whitespace around object not handled properly:\ngot:  %s\nwant: %s", string(gotArgs), `{"key":"value"}`)
	}
}
