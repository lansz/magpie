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

// testB09SchemaChoice covers B09: JSON schema and tool choice conversions do not
// expand input ranges; lossless constraints (required, enum) are preserved;
// circular references are safely handled; business arguments named title/default/const
// are never stripped; allowedFunctionNames are synchronized on remap; and parallel bans
// are never silently allowed.
func testB09SchemaChoice(t *testing.T) {
	t.Run("AllowedFunctionNamesUpdatedOnRemap", testB09AllowedFunctionNamesUpdatedOnRemap)
	t.Run("BusinessFieldsInArgsNeverStripped", testB09BusinessFieldsInArgsNeverStripped)
	t.Run("CircularReferenceHandledSafely", testB09CircularReferenceHandledSafely)
	t.Run("RequiredAndEnumPreservedLosslessly", testB09RequiredAndEnumPreservedLosslessly)
	t.Run("ToolChoiceNoneOmitsTools", testB09ToolChoiceNoneOmitsTools)
	t.Run("ParallelBanRejectedNotSilentlyAllowed", testB09ParallelBanRejectedNotSilentlyAllowed)
}

// 1. When a client tool is mapped/renamed to a native tool, allowedFunctionNames
// must be synchronized to the native name, not left as the client name.
func testB09AllowedFunctionNamesUpdatedOnRemap(t *testing.T) {
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	const (
		clientToolName = "Read"
		nativeToolName = "view_file"
	)

	binding := AntigravityToolBinding{
		NativeID:   "call_native_01",
		NativeName: nativeToolName,
		NativeArgs: json.RawMessage(`{"path":"/a.go"}`),
		ClientID:   "client_id_01",
		ClientName: clientToolName,
	}
	if err := defaultToolBindingStore.Bind(binding); err != nil {
		t.Fatalf("Bind: %v", err)
	}

	body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"tool_choice": {"type": "tool", "name": "` + clientToolName + `"},
		"tools": [{"name": "` + clientToolName + `", "input_schema": {"type": "object"}}],
		"messages": [{"role": "user", "content": "hi"}]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request struct {
			ToolConfig struct {
				FunctionCallingConfig struct {
					Mode                 string   `json:"mode"`
					AllowedFunctionNames []string `json:"allowedFunctionNames"`
				} `json:"functionCallingConfig"`
			} `json:"toolConfig"`
		} `json:"request"`
	}
	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	fc := env.Request.ToolConfig.FunctionCallingConfig
	if fc.Mode != "ANY" {
		t.Errorf("expected mode ANY, got %q", fc.Mode)
	}
	if len(fc.AllowedFunctionNames) != 1 {
		t.Fatalf("expected 1 allowed function name, got %v", fc.AllowedFunctionNames)
	}
	if fc.AllowedFunctionNames[0] != nativeToolName {
		t.Errorf("allowedFunctionNames was not updated on remap: got %q, want %q", fc.AllowedFunctionNames[0], nativeToolName)
	}
}

// 2. Business arguments named title, default, const, format in toolCall.args must NEVER
// be stripped or corrupted by schema cleaning passes.
func testB09BusinessFieldsInArgsNeverStripped(t *testing.T) {
	const businessArgsJSON = `{"const":"IMMUTABLE_VAL","default":42,"format":"pdf_report","title":"Quarterly Results"}`

	body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [
			{"role": "user", "content": "generate doc"},
			{"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "custom_doc", "input": ` + businessArgsJSON + `}]}
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
					FunctionCall *struct {
						Args json.RawMessage `json:"args"`
					} `json:"functionCall"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	var gotArgs json.RawMessage
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				gotArgs = p.FunctionCall.Args
			}
		}
	}

	eq, err := jsonEqualExact(gotArgs, json.RawMessage(businessArgsJSON))
	if err != nil || !eq {
		t.Fatalf("business fields in args were corrupted: got %s, want %s", string(gotArgs), businessArgsJSON)
	}
}

// 3. Schema circular references must terminate safely without stack overflow or corrupted output.
func testB09CircularReferenceHandledSafely(t *testing.T) {
	circularSchema := json.RawMessage(`{
		"$defs": {
			"Node": {
				"type": "object",
				"properties": {
					"name": {"type": "string"},
					"next": {"$ref": "#/$defs/Node"}
				},
				"required": ["name"]
			}
		},
		"type": "object",
		"properties": {
			"root": {"$ref": "#/$defs/Node"}
		}
	}`)

	res := plainSchema(circularSchema)
	if len(res) == 0 {
		t.Fatalf("plainSchema returned empty for circular schema")
	}

	var root map[string]any
	if err := json.Unmarshal(res, &root); err != nil {
		t.Fatalf("result is not valid JSON: %v", err)
	}

	if root["type"] != "object" {
		t.Errorf("expected type 'object', got %v", root["type"])
	}
	props, ok := root["properties"].(map[string]any)
	if !ok || props["root"] == nil {
		t.Fatalf("properties.root missing from converted schema: %v", root)
	}
}

// 4. Required fields and enum values must be preserved losslessly.
func testB09RequiredAndEnumPreservedLosslessly(t *testing.T) {
	inputSchema := json.RawMessage(`{
		"type": "object",
		"properties": {
			"action": {
				"type": "string",
				"enum": ["READ", "WRITE", "EXECUTE"]
			},
			"path": {
				"type": "string"
			}
		},
		"required": ["action", "path"]
	}`)

	res := plainSchema(inputSchema)
	var root struct {
		Type       string `json:"type"`
		Properties struct {
			Action struct {
				Type string   `json:"type"`
				Enum []string `json:"enum"`
			} `json:"action"`
			Path struct {
				Type string `json:"type"`
			} `json:"path"`
		} `json:"properties"`
		Required []string `json:"required"`
	}

	if err := json.Unmarshal(res, &root); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if len(root.Required) != 2 || root.Required[0] != "action" || root.Required[1] != "path" {
		t.Errorf("required mismatch: %v", root.Required)
	}
	if len(root.Properties.Action.Enum) != 3 || root.Properties.Action.Enum[0] != "READ" {
		t.Errorf("enum values were lost or modified: %v", root.Properties.Action.Enum)
	}
}

// 5. Tool choice "none" must completely omit the tools array from wire request.
func testB09ToolChoiceNoneOmitsTools(t *testing.T) {
	body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"tool_choice": {"type": "none"},
		"tools": [{"name": "calculator", "input_schema": {"type": "object"}}],
		"messages": [{"role": "user", "content": "no tools please"}]
	}`

	req, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request map[string]any `json:"request"`
	}
	json.Unmarshal(wireBytes, &env)

	if _, hasTools := env.Request["tools"]; hasTools {
		t.Errorf("tool_choice=none must omit tools from wire, got: %v", env.Request["tools"])
	}
}

// 6. When caller bans parallel tool use, Antigravity must reject it with 400 (never silently allow).
func testB09ParallelBanRejectedNotSilentlyAllowed(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"unexpected"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@example.com", "proj-1", "tok-1")

	banBody := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
		"tools": [{"name": "read", "input_schema": {"type": "object"}}],
		"messages": [{"role": "user", "content": "hi"}]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(banBody))
	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(banBody), &u)

	if status != http.StatusBadRequest || rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for parallel ban, got status=%d rec.Code=%d", status, rec.Code)
	}
	if calls := atomic.LoadInt64(&tr.callCount); calls != 0 {
		t.Fatalf("expected 0 upstream calls for parallel ban, got %d (must not silently allow)", calls)
	}
	if !strings.Contains(msg, "parallel") && !strings.Contains(rec.Body.String(), "parallel") {
		t.Errorf("error message must mention parallel ban, msg=%q body=%s", msg, rec.Body.String())
	}
}
