package gateway

import (
	"encoding/json"
	"testing"
)

// testB10ToolResult covers B10: Tool results are strictly attributed to their correct calls,
// maintaining authenticity without error-masking or ID借用; Gemini puts results in a separate
// role=model group (never merged with functionCall); Claude puts results in a separate role=user group;
// tool result images are cleanly embedded in functionResponse.parts; and out-of-order results are paired correctly.
func testB10ToolResult(t *testing.T) {
	t.Run("GeminiSeparateModelResultGroup", testB10GeminiSeparateModelResultGroup)
	t.Run("ClaudeSeparateUserResultGroup", testB10ClaudeSeparateUserResultGroup)
	t.Run("ToolResultEmbeddedImages", testB10ToolResultEmbeddedImages)
	t.Run("OutofOrderResultsAttributedCorrectly", testB10OutofOrderResultsAttributedCorrectly)
	t.Run("ErrorResultRealOutputPreserved", testB10ErrorResultRealOutputPreserved)
}

// 1. Gemini: Tool results belong to an independent role=model content group,
// NEVER merged into the preceding functionCall content group.
func testB10GeminiSeparateModelResultGroup(t *testing.T) {
	body := `{
		"model": "gemini-3.8-flash-high",
		"messages": [
			{"role": "user", "content": "read file"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "a.txt"}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "content of a"}]}
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
				Role  string `json:"role"`
				Parts []struct {
					FunctionCall     any `json:"functionCall"`
					FunctionResponse any `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	contents := env.Request.Contents
	if len(contents) < 3 {
		t.Fatalf("expected at least 3 distinct content items (user -> model call -> model result), got %d: %v", len(contents), contents)
	}

	// Turn 0: user question
	if contents[0].Role != "user" {
		t.Errorf("contents[0].Role = %q, want 'user'", contents[0].Role)
	}

	// Turn 1: model tool call
	callTurn := contents[1]
	if callTurn.Role != "model" {
		t.Errorf("callTurn.Role = %q, want 'model'", callTurn.Role)
	}
	if len(callTurn.Parts) != 1 || callTurn.Parts[0].FunctionCall == nil {
		t.Errorf("callTurn must contain only functionCall: %v", callTurn.Parts)
	}

	// Turn 2: model tool result (Gemini uses role=model, separate from call)
	resultTurn := contents[2]
	if resultTurn.Role != "model" {
		t.Errorf("resultTurn.Role = %q, want 'model' for Gemini", resultTurn.Role)
	}
	if len(resultTurn.Parts) != 1 || resultTurn.Parts[0].FunctionResponse == nil {
		t.Errorf("resultTurn must contain functionResponse: %v", resultTurn.Parts)
	}

	// Anti-pattern assertion: functionCall and functionResponse must NOT be in the same content!
	for i, c := range contents {
		hasCall, hasResp := false, false
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				hasCall = true
			}
			if p.FunctionResponse != nil {
				hasResp = true
			}
		}
		if hasCall && hasResp {
			t.Errorf("contents[%d] illegally merged functionCall and functionResponse in single content", i)
		}
	}
}

// 2. Claude: Tool results belong to an independent role=user content group.
func testB10ClaudeSeparateUserResultGroup(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "read file"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "a.txt"}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "content of a"}]}
		]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request struct {
			Contents []struct {
				Role  string `json:"role"`
				Parts []struct {
					FunctionCall     any `json:"functionCall"`
					FunctionResponse any `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	contents := env.Request.Contents
	if len(contents) < 3 {
		t.Fatalf("expected at least 3 distinct content items, got %d", len(contents))
	}

	// Turn 1: model call
	if contents[1].Role != "model" || contents[1].Parts[0].FunctionCall == nil {
		t.Errorf("call turn mismatch: %v", contents[1])
	}
	// Turn 2: user result
	if contents[2].Role != "user" || contents[2].Parts[0].FunctionResponse == nil {
		t.Errorf("result turn mismatch: %v", contents[2])
	}
}

// 3. Tool result with embedded image must be nested cleanly inside functionResponse.parts.
func testB10ToolResultEmbeddedImages(t *testing.T) {
	const dummyBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

	body := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "take screenshot"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "t_snap", "name": "screenshot", "input": {}}]},
			{"role": "user", "content": [
				{
					"type": "tool_result",
					"tool_use_id": "t_snap",
					"content": [
						{"type": "text", "text": "captured screen"},
						{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "` + dummyBase64 + `"}}
					]
				}
			]}
		]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request struct {
			Contents []struct {
				Parts []struct {
					FunctionResponse *struct {
						ID       string `json:"id"`
						Name     string `json:"name"`
						Response struct {
							Output string `json:"output"`
						} `json:"response"`
						Parts []struct {
							InlineData *struct {
								MimeType string `json:"mimeType"`
								Data     string `json:"data"`
							} `json:"inlineData"`
						} `json:"parts"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal: %v\nwire: %s", err, wireBytes)
	}

	var foundFR *struct {
		ID       string `json:"id"`
		Name     string `json:"name"`
		Response struct {
			Output string `json:"output"`
		} `json:"response"`
		Parts []struct {
			InlineData *struct {
				MimeType string `json:"mimeType"`
				Data     string `json:"data"`
			} `json:"inlineData"`
		} `json:"parts"`
	}

	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionResponse != nil {
				foundFR = p.FunctionResponse
			}
		}
	}

	if foundFR == nil {
		t.Fatalf("functionResponse not found on wire: %s", wireBytes)
	}
	if foundFR.Response.Output != "captured screen" {
		t.Errorf("output text mismatch: got %q, want 'captured screen'", foundFR.Response.Output)
	}
	if len(foundFR.Parts) != 1 || foundFR.Parts[0].InlineData == nil {
		t.Fatalf("embedded image in functionResponse.parts missing: %v", foundFR.Parts)
	}
	img := foundFR.Parts[0].InlineData
	if img.MimeType != "image/png" || img.Data != dummyBase64 {
		t.Errorf("embedded image payload mismatch: mime=%q dataLen=%d", img.MimeType, len(img.Data))
	}
}

// 4. Out-of-order tool results in the same round must pair with their exact calls,
// never borrowing an adjacent tool name or ID.
func testB10OutofOrderResultsAttributedCorrectly(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "do both"},
			{"role": "assistant", "content": [
				{"type": "tool_use", "id": "call_alpha", "name": "tool_alpha", "input": {"val": 1}},
				{"type": "tool_use", "id": "call_beta", "name": "tool_beta", "input": {"val": 2}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "call_beta", "content": "res_beta"},
				{"type": "tool_result", "tool_use_id": "call_alpha", "content": "res_alpha"}
			]}
		]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request struct {
			Contents []struct {
				Parts []struct {
					FunctionResponse *struct {
						ID       string `json:"id"`
						Name     string `json:"name"`
						Response struct {
							Output string `json:"output"`
						} `json:"response"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	var resps []struct{ id, name, output string }
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionResponse != nil {
				resps = append(resps, struct{ id, name, output string }{
					id:     p.FunctionResponse.ID,
					name:   p.FunctionResponse.Name,
					output: p.FunctionResponse.Response.Output,
				})
			}
		}
	}

	if len(resps) != 2 {
		t.Fatalf("expected 2 functionResponses, got %d: %v", len(resps), resps)
	}

	// 1st returned result is beta
	if resps[0].id != "call_beta" || resps[0].name != "tool_beta" || resps[0].output != "res_beta" {
		t.Errorf("resps[0] attribution mismatch: %v", resps[0])
	}
	// 2nd returned result is alpha
	if resps[1].id != "call_alpha" || resps[1].name != "tool_alpha" || resps[1].output != "res_alpha" {
		t.Errorf("resps[1] attribution mismatch: %v", resps[1])
	}
}

// 5. Tool result with is_error=true preserves the real error message in response.output.
func testB10ErrorResultRealOutputPreserved(t *testing.T) {
	const errMsg = "permission denied: cannot access /root/secret.key"

	body := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "read file"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "c_err", "name": "read", "input": {"path": "/root/secret.key"}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c_err", "content": "` + errMsg + `", "is_error": true}]}
		]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request struct {
			Contents []struct {
				Parts []struct {
					FunctionResponse *struct {
						Response map[string]any `json:"response"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	json.Unmarshal(wireBytes, &env)

	var outputVal any
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionResponse != nil {
				outputVal = p.FunctionResponse.Response["output"]
			}
		}
	}

	if outputVal != errMsg {
		t.Errorf("error output was corrupted or masked: got %v, want %q", outputVal, errMsg)
	}
}
